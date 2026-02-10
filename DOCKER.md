# DeviceEmulator Docker Deployment

This document describes how to deploy DeviceEmulator using Docker.

## Requirements

### Base requirements

- Docker 20.10+
- Docker Compose 2.0+
- At least 8 GB RAM
- At least 10 GB disk space

### GPU support (optional)

If you need GPU support:

- NVIDIA GPU (compute capability 7.0+)
- NVIDIA Docker support
- `nvidia-docker2` or Docker with `nvidia-container-toolkit`

## Quick Start

### 1. One-command launch (recommended)

```bash
chmod +x quick-start.sh
./quick-start.sh
```

### 2. Manual launch

Build the image:

```bash
DOCKER_BUILDKIT=1 docker build -t device-emulator:latest .
```

Start services (GPU mode):

```bash
docker-compose up -d
```

Start services (CPU mode):

```bash
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Usage

### Enter the container

```bash
docker-compose exec device-emulator bash
```

### Run the emulator

```bash
# Inside the container
morphling_emulator \
  --num_devices 4 \
  --model_name facebook/opt-125m \
  --backend proxy \
  --seq_length 128 \
  --batch_size 1

# From the host
docker-compose exec device-emulator morphling_emulator --num_devices 4 --model_name facebook/opt-125m
```

### Generate a device config

```bash
docker-compose exec device-emulator morphling_device_config --num_devices 4 --device_type virtual
```

### View logs

```bash
# All services
docker-compose logs

# Specific services
docker-compose logs device-emulator
docker-compose logs redis

# Follow logs
docker-compose logs -f device-emulator
```

## Services

### Service list

- `redis`: Redis (port 6379)
- `device-emulator`: main DeviceEmulator service
- `nginx-proxy`: optional Nginx proxy (use the profile to start)

### Port mappings

- `6379`: Redis
- `8080`: DeviceEmulator HTTP
- `443`: DeviceEmulator HTTPS

### Volumes

- `./logs`: log output
- `./data`: data files
- `./config`: configuration
- `redis_data`: Redis persistence

## Advanced Configuration

### Environment variables

Edit `docker-compose.yml` and set:

```yaml
environment:
  - SPDLOG_LEVEL=debug
  - CUDA_VISIBLE_DEVICES=0
  - MORPHLING_HOME=/app
```

### GPU selection

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
```

### Start the Nginx proxy

```bash
docker-compose --profile proxy up -d
```

## Troubleshooting

### CUDA errors

```bash
# Check NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# If GPU is not available, use CPU mode
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### Out of memory

```bash
# Reduce device count
docker-compose exec device-emulator morphling_emulator --num_devices 2
```

### Port conflicts

```bash
netstat -tulpn | grep -E ':(6379|8080|443)'
# Or update port mappings in docker-compose.yml
```

### Permission issues

```bash
chmod +x quick-start.sh docker-entrypoint.sh
sudo usermod -aG docker $USER
# Re-login for the change to take effect
```

### Debugging commands

```bash
# Container status
docker-compose ps

# Resource usage
docker stats

# Open a shell
docker-compose exec device-emulator bash

# Restart a service
docker-compose restart device-emulator

# Detailed logs
docker-compose logs --details device-emulator
```

## Cleanup

```bash
# Stop services
docker-compose down

# Remove data volumes
docker-compose down -v

docker rmi device-emulator:latest
```

## ccache Build Caching

The Docker build uses ccache to cache C++/CUDA compilation results across
rebuilds. This requires BuildKit (enabled by default in Docker 23+, or set
`DOCKER_BUILDKIT=1`).

### How it works

- `setup.py` detects `ccache` on `$PATH` and passes
  `-DCMAKE_<LANG>_COMPILER_LAUNCHER=ccache` to CMake.
- The Dockerfile uses `--mount=type=cache,target=/ccache` so the cache persists
  across layer rebuilds.
- The cache is capped at 5 GB (`ccache -M 5G`).

### Build behavior

- First build (cold cache): normal build time, cache is populated.
- Subsequent builds: ccache hits on unchanged translation units, even when
  `COPY . /app/` invalidates the layer.

### Check cache stats

```bash
docker run --rm device-emulator:latest ccache -s
```

### Run a container with GPU

```bash
docker run --rm -it --gpus=all --ulimit memlock=-1:-1 --cap-add IPC_LOCK device-emulator bash
```
