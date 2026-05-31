# Deployment

This document covers running Morphling in two modes:

- **Virtual devices** — multiple emulated devices co-located on one host.
- **Physical devices** — emulator instances on real edge hardware fronted by
  an Nginx stream proxy.

## Virtual device deployment

```bash
# Start Redis (stop existing container if needed)
if [ "$(docker ps -q -f name=redis)" ]; then
    docker stop redis
fi

docker run -dit --rm --name redis -p 6379:6379 redis
sleep 5

cd scripts
SPDLOG_LEVEL=debug python run_devices.py \
    --num_devices 4 \
    --model_name facebook/opt-125m \
    --backend proxy \
    --seq_length 128 \
    --batch_size 1 \
    --cfg ../config/proxy/svr.ini
```

## Physical device deployment

```bash
#!/usr/bin/env bash
set -e

# 1) Remove any existing redis or morphling containers
REDIS_CONTAINERS=$(docker ps -aq -f name=redis)
MORPHLING_CONTAINERS=$(docker ps -aq -f name=morphling)

if [ -n "$REDIS_CONTAINERS" ]; then
    echo "Stopping and removing existing redis containers..."
    docker rm -f $REDIS_CONTAINERS
fi

if [ -n "$MORPHLING_CONTAINERS" ]; then
    echo "Stopping and removing existing morphling containers..."
    docker rm -f $MORPHLING_CONTAINERS
fi

# 2) (Optional) Kill any leftover run_devices.py processes
if pgrep -f "run_devices.py" >/dev/null; then
    echo "Killing leftover run_devices.py processes..."
    pkill -f "run_devices.py"
fi

# 3) Start a new Redis container
echo "Starting a new Redis container..."
docker run -dit --rm --name redis -p 6379:6379 redis
sleep 5

# 4) Generate device config
cd morphling/entrypoint
SPDLOG_LEVEL=debug python generate_device_config.py --num_devices 1 --device_type physical
cp device_config.json ../../scripts/

# 5) Run Morphling devices in the background
cd ../../scripts
SPDLOG_LEVEL=debug python run_devices.py \
    --num_devices 1 \
    --model_name facebook/opt-125m \
    --backend proxy \
    --seq_length 128 \
    --batch_size 1 \
    --cfg ../config/proxy/svr.ini \
    &

# 6) Start Nginx container (morphling-proxy) with the correct mounts for stream
cd ..
docker run -d \
    --name morphling-proxy \
    -p 443:443 \
    -v "$(pwd)/docker-nginx/nginx.conf:/etc/nginx/nginx.conf:ro" \
    -v "$(pwd)/docker-nginx/morphling_stream.conf:/etc/nginx/stream_conf.d/morphling_stream.conf:ro" \
    nginx:latest

echo "All done. Now test from local with: nc -vz <server_ip> 443"

# Keep script alive so the background job isn't killed
wait
```

## Server-measured device profile (issue #55)

By default the proxy server trusts the seven legacy fields the device
reports in its `DeviceProfileData` (uuid, flops, memory, ul/dl bandwidth,
ul/dl latency). When operators want to validate those numbers, the
server can drive a three-stage probe sequence (M1 latency, M2 bandwidth,
M3 FLOPS) AFTER device registration and BEFORE the device becomes
scheduler-eligible. All three probes default off — when none are
enabled, the pre-#55 control flow is preserved (register →
`SendIdlePartitions`).

### Configuration

Edit the `[device_measurement]` section of
[`config/proxy/svr.ini`](../config/proxy/svr.ini), or override per-process
with the corresponding `MORPHLING_MEASURE_*` env var. Precedence per knob
is **INI value > env var > built-in default**.

| INI key                    | Env var                          | Default     | Purpose                                                                          |
|----------------------------|----------------------------------|-------------|----------------------------------------------------------------------------------|
| `enable_latency`           | `MORPHLING_MEASURE_LAT`          | `0`         | M1: echo RTT, reports `measured_lat_ns` as RTT/2.                                |
| `enable_bandwidth`         | `MORPHLING_MEASURE_BW`           | `0`         | M2: echo a payload, reports `measured_{ul,dl}_bw_bps`.                           |
| `enable_flops`             | `MORPHLING_MEASURE_FLOPS`        | `0`         | M3: seeded float32 GEMM challenge with element-wise verification.                |
| `latency_payload_bytes`    | `MORPHLING_MEASURE_LAT_BYTES`    | `64`        | M1 payload size on the wire.                                                     |
| `bandwidth_payload_bytes`  | `MORPHLING_MEASURE_BW_BYTES`     | `4194304`   | M2 payload size (4 MiB).                                                         |
| `flops_matrix_dim`         | `MORPHLING_MEASURE_FLOPS_DIM`    | `256`       | M3 square matrix dim (m=n=k); 256³ f32 runs inline on the libevent loop thread. |
| `probe_timeout_sec`        | `MORPHLING_MEASURE_TIMEOUT_S`    | `5.0`       | Per-probe timeout. Timeout marks the session FAILED and reverts to the trust model. |
| `flops_tolerance`          | `MORPHLING_MEASURE_FLOPS_TOL`    | `1e-3`      | `atol + rtol` used to verify the device-computed C matrix.                       |

### What the probes report

The server merges measured values into the device's stored
`DeviceProfileData` via `UpdateMeasuredProfile`; the seven device-reported
fields are NEVER overwritten. The five measured fields are:

- `measured_lat_ns` — one-way latency estimate (RTT/2).
- `measured_ul_bw_bps`, `measured_dl_bw_bps` — both filled with the echo
  throughput, which is `min(ul, dl)` rather than per-direction; splitting
  the two requires device cooperation and is deferred.
- `measured_flops` — `2*m*n*k / device_compute_time_s`, derived from RTT
  minus the M1 latency.
- `measured_flops_verified` — `true` iff every element of the
  device-returned C matrix matches the server's OpenBLAS reference within
  `flops_tolerance`.

A completed probe sequence emits a single INFO line like:

```
Device <id> measurement complete: ok=1 state=DONE
  measured_lat_ns=42100 measured_ul_bw_bps=950000000 measured_dl_bw_bps=950000000
  measured_flops=1430000000000 verified=1
```

### Reconciliation policy

Out of scope for #55. The current build stores both reported and
measured values side-by-side; downstream consumers (schedulers,
analytics) can pick either. A follow-up issue will decide whether
`flops` exposed to schedulers should be `min(reported, measured)`,
`measured`, or a weighted blend, once enough measurement data exists.

## Notes

- Both modes require Redis for inter-device coordination. The container
  exposes port `6379`.
- The physical deployment expects `docker-nginx/nginx.conf` and
  `docker-nginx/morphling_stream.conf` to exist in the repo root.
- `SPDLOG_LEVEL` controls C++ backend verbosity (`debug`, `info`, `warn`,
  `error`).
- For multi-GPU or multi-host runs, see [`docs/DOCKER.md`](DOCKER.md) and
  [`docs/DEV_README.md`](DEV_README.md).
