#!/bin/bash
# =============================================================================
# measure_bandwidth.sh — Measure inter-node bandwidth using iperf3
# =============================================================================
# Outputs bandwidth matrix in YAML format for cluster configuration.
#
# Usage:
#   ./measure_bandwidth.sh /path/to/asteroid_default.yaml
#
# Requires:
#   - iperf3 installed on all nodes
#   - SSH access to all nodes (via ansible inventory)
#   - Python venv with ansible
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="$BASELINES_DIR/deploy_asteroid"
VENV_DIR="$BASELINES_DIR/.venv"

# Activate venv
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
fi

CONFIG_FILE="${1:-$BASELINES_DIR/configs/asteroid_default.yaml}"
INVENTORY="$DEPLOY_DIR/inventory.ini"
SECRETS="$DEPLOY_DIR/secrets.yml"
IPERF_PORT=5201
DURATION=3

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

if [[ ! -f "$INVENTORY" ]]; then
    echo "[ERROR] Inventory not found: $INVENTORY"
    echo "       Run deploy_asteroid.sh first to generate it"
    exit 1
fi

echo "=========================================="
echo " Inter-Node Bandwidth Measurement"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Inventory: $INVENTORY"
echo ""

# Extract node IPs from inventory
NODES=()
NODE_NAMES=()
while IFS= read -r line; do
    if [[ "$line" =~ ansible_host=([0-9.]+) ]]; then
        ip="${BASH_REMATCH[1]}"
        # Extract node name (before ansible_host)
        name=$(echo "$line" | awk '{print $1}')
        NODES+=("$ip")
        NODE_NAMES+=("$name")
    fi
done < "$INVENTORY"

NUM_NODES=${#NODES[@]}
echo "Found $NUM_NODES nodes:"
for i in "${!NODES[@]}"; do
    echo "  [$i] ${NODE_NAMES[$i]} = ${NODES[$i]}"
done
echo ""

# Results matrix
declare -A BW_MATRIX

measure_bandwidth() {
    local src_name="$1"
    local src_ip="$2"
    local dst_ip="$3"
    local dst_name="$4"

    # Start iperf3 server on destination
    ansible "$dst_name" -i "$INVENTORY" -m shell \
        -a "pkill iperf3 2>/dev/null; iperf3 -s -D -p $IPERF_PORT" \
        --extra-vars @"$SECRETS" >/dev/null 2>&1 || true
    sleep 1

    # Run iperf3 client from source
    local result
    result=$(ansible "$src_name" -i "$INVENTORY" -m shell \
        -a "iperf3 -c $dst_ip -t $DURATION -p $IPERF_PORT 2>&1 | grep sender | awk '{print \$7}'" \
        --extra-vars @"$SECRETS" 2>/dev/null | grep -E '^[0-9.]+$' || echo "0")

    # Parse result (in Mbits/sec or Gbits/sec)
    local bw_mbps=0
    if [[ -n "$result" && "$result" != "0" ]]; then
        # Check for Gbits/sec in the output
        local unit
        unit=$(ansible "$src_name" -i "$INVENTORY" -m shell \
            -a "iperf3 -c $dst_ip -t 1 -p $IPERF_PORT 2>&1 | grep sender | awk '{print \$8}'" \
            --extra-vars @"$SECRETS" 2>/dev/null | grep -E 'bits' || echo "Mbits/sec")
        
        if [[ "$unit" == *"Gbits"* ]]; then
            bw_mbps=$(echo "$result * 1000" | bc 2>/dev/null || echo "0")
        else
            bw_mbps="$result"
        fi
    fi

    # Kill iperf3 server
    ansible "$dst_name" -i "$INVENTORY" -m shell \
        -a "pkill iperf3 2>/dev/null" \
        --extra-vars @"$SECRETS" >/dev/null 2>&1 || true

    echo "$bw_mbps"
}

echo "Measuring bandwidth between all node pairs..."
echo "(This may take a few minutes)"
echo ""

OUTPUT_FILE="$BASELINES_DIR/configs/link_bandwidths.yaml"

echo "# Auto-generated link bandwidth measurements" > "$OUTPUT_FILE"
echo "# Generated: $(date -Iseconds)" >> "$OUTPUT_FILE"
echo "# Duration: ${DURATION}s per test" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "link_bandwidths:" >> "$OUTPUT_FILE"

for i in "${!NODES[@]}"; do
    src_ip="${NODES[$i]}"
    src_name="${NODE_NAMES[$i]}"
    
    for j in "${!NODES[@]}"; do
        if [[ $i -eq $j ]]; then
            continue
        fi
        
        dst_ip="${NODES[$j]}"
        dst_name="${NODE_NAMES[$j]}"
        
        printf "  %s → %s: " "$src_name" "$dst_name"
        
        bw=$(measure_bandwidth "$src_name" "$src_ip" "$dst_ip" "$dst_name")
        
        if [[ -z "$bw" || "$bw" == "0" ]]; then
            bw="100.0"  # fallback default
            echo "FAILED (using default 100 Mbps)"
        else
            printf "%.1f Mbps\n" "$bw"
        fi
        
        # Store in format: "src_rank,dst_rank": bandwidth_mbps
        echo "  \"$i,$j\": $bw" >> "$OUTPUT_FILE"
        BW_MATRIX["$i,$j"]="$bw"
    done
done

echo ""
echo "=========================================="
echo " Bandwidth Matrix (Mbps)"
echo "=========================================="

# Print header
printf "%12s" ""
for i in "${!NODES[@]}"; do
    printf "%12s" "[$i]"
done
echo ""

# Print matrix
for i in "${!NODES[@]}"; do
    printf "%12s" "[${NODE_NAMES[$i]:0:10}]"
    for j in "${!NODES[@]}"; do
        if [[ $i -eq $j ]]; then
            printf "%12s" "-"
        else
            bw="${BW_MATRIX[$i,$j]:-0}"
            printf "%12.0f" "$bw"
        fi
    done
    echo ""
done

echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Calculate min/avg/max for YAML comment
all_bw=()
for key in "${!BW_MATRIX[@]}"; do
    all_bw+=("${BW_MATRIX[$key]}")
done

if [[ ${#all_bw[@]} -gt 0 ]]; then
    min_bw=$(printf '%s\n' "${all_bw[@]}" | sort -n | head -1)
    max_bw=$(printf '%s\n' "${all_bw[@]}" | sort -n | tail -1)
    avg_bw=$(printf '%s\n' "${all_bw[@]}" | awk '{sum+=$1} END {print sum/NR}')
    
    echo "Summary:"
    echo "  Min: ${min_bw} Mbps"
    echo "  Max: ${max_bw} Mbps"
    echo "  Avg: ${avg_bw} Mbps"
    echo ""
    echo "Suggested d2d_bandwidth_mbps value (conservative): ${min_bw}"
fi

echo ""
echo "To use these values, add to asteroid_default.yaml:"
echo "  cluster:"
echo "    link_bandwidths: !include link_bandwidths.yaml"
echo ""
echo "Or manually set d2d_bandwidth_mbps to the minimum measured value."
