#!/bin/bash

# Batch device launch script
# Usage: ./start_multiple_devices.sh <num_devices> [interval_seconds] [device_id_start]
#
# Examples:
#   ./start_multiple_devices.sh 4           # Start 4 devices with default 5s interval, starting from device 0
#   ./start_multiple_devices.sh 4 3         # Start 4 devices with 3s interval
#   ./start_multiple_devices.sh 4 2 10      # Start 4 devices with 2s interval, starting from device 10

NUM_DEVICES=${1:-4}           # Default: 4 devices
INTERVAL=${2:-5}              # Default: 5 seconds between launches
START_ID=${3:-0}              # Default: start from device 0

# Validate input
if ! [[ "$NUM_DEVICES" =~ ^[0-9]+$ ]] || [ "$NUM_DEVICES" -lt 1 ]; then
    echo "Error: NUM_DEVICES must be a positive integer"
    exit 1
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "Error: INTERVAL must be a non-negative integer"
    exit 1
fi

if ! [[ "$START_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: START_ID must be a non-negative integer"
    exit 1
fi

echo "Starting $NUM_DEVICES device(s)..."
echo "Interval: ${INTERVAL}s"
echo "Starting device ID: $START_ID"
echo ""

# Array to store PIDs of launched devices
pids=()

# Launch devices in a loop
for i in $(seq 0 $((NUM_DEVICES - 1))); do
    DEVICE_ID=$((START_ID + i))
    
    echo "[$(date '+%H:%M:%S')] Launching device $DEVICE_ID..."
    
    morphling_device \
      --id "$DEVICE_ID" \
      --flops 1T \
      --memory 8G \
      --ul_bw 100M \
      --dl_bw 100M \
      --ul_lat 10 \
      --dl_lat 10 \
      --backend proxy \
      --cfg config/proxy/cli.ini \
      --redis_host 128.110.220.77:6379 \
      --proxy_host 128.110.220.77:39000 &
    
    # Save the PID
    pids+=($!)
    
    # Wait before launching the next device (except after the last one)
    if [ $i -lt $((NUM_DEVICES - 1)) ]; then
        echo "Waiting ${INTERVAL}s before launching next device..."
        sleep "$INTERVAL"
    fi
done

echo ""
echo "All $NUM_DEVICES device(s) launched!"
echo "Device PIDs: ${pids[@]}"
echo ""
echo "To stop all devices, run:"
echo "  kill ${pids[@]}"
echo ""
echo "To monitor devices:"
echo "  ps aux | grep morphling_device"

# Optional: wait for all background processes
# Uncomment the line below if you want the script to wait until all devices exit
# wait "${pids[@]}"
