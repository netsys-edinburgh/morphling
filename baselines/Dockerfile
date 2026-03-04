# Baselines distributed training Docker image.
# Matches conda env emulator: torch 2.3.0 + CUDA 12.1.
#
# Build (from baselines/ as context):
#   docker build -t baselines:latest .
#
# Run (single-node, 4 GPUs):
#   docker run --rm --gpus all --ipc=host baselines:latest \
#       python -m baselines.train --spawn --num-gpus 4 --strategy dtfm
#
# Interactive:
#   docker run --rm -it --gpus all --ipc=host baselines:latest bash

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

LABEL maintainer="baselines-team"
LABEL description="Baselines distributed training framework"

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

# Install Python dependencies first (layer caching)
# requirements.txt is auto-generated from pyproject.toml by deploy_asteroid.sh
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy baselines package (build context is baselines/)
COPY . /workspace/baselines/

# Default: show help
CMD ["python", "-m", "baselines.train", "--help"]
