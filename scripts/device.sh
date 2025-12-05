#!/bin/bash

# Device launch script with device ID argument
# Usage: ./device.sh <device_id>

DEVICE_ID=${1:-0}  # Default to device 0 if not provided

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
  --proxy_host 127.0.0.1:39000
