#!/usr/bin/env bash
# Watchdog for VTIME sweep: detects hung emulations and kills them.
#
# Two hang signals checked every CHECK_INTERVAL:
#   1. Sweep log stops growing for HANG_TIMEOUT seconds
#   2. Sweep log growing but only "WaitMatMul" chatter for CHATTER_TIMEOUT
#      (server stuck in response-wait loop, will never finish)
#
# Kill method: SIGKILL the host-side "timeout ... docker run" process tree,
# since "docker kill" fails on D-state GPU containers.
#
# Usage:
#   bash scripts/vtime_watchdog.sh
#
# Environment overrides:
#   HANG_TIMEOUT=600        seconds of no log activity (default: 10min)
#   CHATTER_TIMEOUT=300     seconds of WaitMatMul-only output (default: 5min)
#   CHECK_INTERVAL=30       seconds between checks (default: 30s)
#   SWEEP_LOG=results/vtime_sweep_run.log

set -euo pipefail

HANG_TIMEOUT="${HANG_TIMEOUT:-600}"
CHATTER_TIMEOUT="${CHATTER_TIMEOUT:-300}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
SWEEP_LOG="${SWEEP_LOG:-results/vtime_sweep_run.log}"

last_size=0
stale_since=0
chatter_since=0
last_run_marker=""

echo "[watchdog] started — hang=${HANG_TIMEOUT}s, chatter=${CHATTER_TIMEOUT}s, interval=${CHECK_INTERVAL}s"
echo "[watchdog] monitoring: $SWEEP_LOG"

kill_current_run() {
    local reason="$1"
    echo "[watchdog] $(date '+%H:%M:%S') KILLING — $reason"

    local pid
    pid=$(ps aux | grep 'timeout.*docker run.*device-emulator' | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$pid" ]; then
        echo "[watchdog] killing host process tree pid=$pid"
        kill -9 "$pid" 2>/dev/null || true
        sleep 3
        kill -9 "$pid" 2>/dev/null || true
    else
        echo "[watchdog] no timeout+docker process found, trying docker kill"
        local container
        container=$(docker ps --filter ancestor=device-emulator:latest --format '{{.Names}}' | head -1)
        if [ -n "$container" ]; then
            docker kill "$container" 2>/dev/null || true
        fi
    fi

    stale_since=0
    chatter_since=0
    last_size=$(stat -c %s "$SWEEP_LOG" 2>/dev/null || echo 0)
    echo "[watchdog] reset — waiting for next run"
    sleep 15
}

while true; do
    if ! tmux has-session -t vtime-sweep 2>/dev/null; then
        echo "[watchdog] tmux session gone — exiting."
        break
    fi

    if [ ! -f "$SWEEP_LOG" ]; then
        sleep "$CHECK_INTERVAL"
        continue
    fi

    cur_size=$(stat -c %s "$SWEEP_LOG" 2>/dev/null || echo 0)
    now=$(date +%s)

    cur_marker=$(grep -c '^\[RUN\]' "$SWEEP_LOG" 2>/dev/null || echo 0)
    if [ "$cur_marker" != "$last_run_marker" ]; then
        last_run_marker="$cur_marker"
        chatter_since=0
        stale_since=0
    fi

    if [ "$cur_size" -gt "$last_size" ]; then
        last_size=$cur_size
        stale_since=0

        recent=$(tail -5 "$SWEEP_LOG" 2>/dev/null || echo "")
        if echo "$recent" | grep -qE 'Dispatching level|Completed level|\[OK\]|\[RUN\]|\[SKIP\]'; then
            chatter_since=0
            echo "[watchdog] $(date '+%H:%M:%S') progress OK (${cur_size}b)"
        else
            if [ "$chatter_since" -eq 0 ]; then
                chatter_since=$now
            fi
            chatter_dur=$(( now - chatter_since ))
            if [ "$chatter_dur" -ge "$CHATTER_TIMEOUT" ]; then
                kill_current_run "WaitMatMul chatter for ${chatter_dur}s"
                continue
            fi
            echo "[watchdog] $(date '+%H:%M:%S') chatter ${chatter_dur}s/${CHATTER_TIMEOUT}s"
        fi
    else
        if [ "$stale_since" -eq 0 ]; then
            stale_since=$now
        fi
        stale_dur=$(( now - stale_since ))
        if [ "$stale_dur" -ge "$HANG_TIMEOUT" ]; then
            kill_current_run "no output for ${stale_dur}s"
            continue
        fi
        echo "[watchdog] $(date '+%H:%M:%S') stale ${stale_dur}s/${HANG_TIMEOUT}s"
    fi

    sleep "$CHECK_INTERVAL"
done

echo "[watchdog] done."
