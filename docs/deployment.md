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

## Notes

- Both modes require Redis for inter-device coordination. The container
  exposes port `6379`.
- The physical deployment expects `docker-nginx/nginx.conf` and
  `docker-nginx/morphling_stream.conf` to exist in the repo root.
- `SPDLOG_LEVEL` controls C++ backend verbosity (`debug`, `info`, `warn`,
  `error`).
- For multi-GPU or multi-host runs, see [`docs/DOCKER.md`](DOCKER.md) and
  [`docs/DEV_README.md`](DEV_README.md).
