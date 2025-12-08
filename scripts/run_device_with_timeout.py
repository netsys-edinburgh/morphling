#!/usr/bin/env python3
"""
Device launcher with graceful timeout disconnect using SIGTERM (kill -15).

Usage:
    python3 run_device_with_timeout.py --id <device_id> --timeout <seconds>
    python3 run_device_with_timeout.py --id 0 --timeout 10
"""

import argparse
import subprocess
import signal
import time
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch device and disconnect after timeout using SIGTERM"
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="Device ID (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds before graceful disconnect (default: 10)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="proxy",
        help="Backend type (default: proxy)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="config/proxy/cli.ini",
        help="Config file path (default: config/proxy/cli.ini)",
    )
    parser.add_argument(
        "--proxy_host",
        type=str,
        default="127.0.0.1:39000",
        help="Proxy server address (default: 127.0.0.1:39000)",
    )

    args = parser.parse_args()

    # Build the command to launch device
    cmd = [
        "python3",
        "morphling/entrypoint/run_device.py",
        "--id",
        str(args.id),
        "--flops",
        "1T",
        "--memory",
        "8G",
        "--ul_bw",
        "100M",
        "--dl_bw",
        "100M",
        "--ul_lat",
        "10",
        "--dl_lat",
        "10",
        "--backend",
        args.backend,
        "--cfg",
        args.cfg,
        "--proxy_host",
        args.proxy_host,
    ]

    print(f"[Device {args.id}] Starting device with timeout={args.timeout}s")
    print(f"[Device {args.id}] Command: {' '.join(cmd)}")

    # Start device process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    device_pid = process.pid

    print(f"[Device {args.id}] Process started with PID={device_pid}")

    try:
        # Wait for the specified timeout
        time.sleep(args.timeout)

        print(f"[Device {args.id}] Timeout reached ({args.timeout}s). Sending SIGTERM...")
        # Send SIGTERM (kill -15) for graceful shutdown
        process.terminate()

        # Wait for process to terminate gracefully (give it 5 seconds)
        try:
            process.wait(timeout=5)
            print(f"[Device {args.id}] Process terminated gracefully")
        except subprocess.TimeoutExpired:
            print(f"[Device {args.id}] Process did not terminate gracefully. Sending SIGKILL...")
            process.kill()
            process.wait()
            print(f"[Device {args.id}] Process killed with SIGKILL")

    except KeyboardInterrupt:
        print(f"\n[Device {args.id}] Keyboard interrupt detected. Terminating process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print(f"[Device {args.id}] Process terminated")
        sys.exit(0)


if __name__ == "__main__":
    main()
