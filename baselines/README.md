# Baselines — Distributed Training Framework

A modular framework for pipeline-parallel + data-parallel distributed training,
refactored from three research baseline implementations (DT-FM, Asteroid,
Confident) into a professional, reusable Python package.

## Architecture

The framework follows a layered architecture (L1–L8) derived from the DT-FM
system:

```
L8  Orchestration     baselines/train.py          Training loop + launch
L7  Fault Tolerance   baselines/fault_tolerance/  Checkpoint, heartbeat, replication
L6  Communication     baselines/communication/    NCCL P2P, AllReduce, Gloo
L5  Scheduling        baselines/schedulers/       GCMA, DP partitioner, HPP
L4  Optimizer         baselines/utils/flatten.py  Flattened params + AdamW
L3  Dataset           (synthetic / HuggingFace)   Data loading
L2  Compute           baselines/models/           GPT-2, Llama, BERT, pipeline stages
L1  Config & State    baselines/core/             Dataclass configs, ABCs
```

### Strategy Pattern

Each baseline inherits from `ParallelismStrategy` and provides its own
planning algorithm:

| Strategy | Planner | Schedule |
|----------|---------|----------|
| `DTFMStrategy` | GCMA topology optimizer + DP layer partitioner | GPipe |
| `AsteroidStrategy` | HPP heterogeneity-aware planner | 1F1B |
| `ConfidentStrategy` | DP bottleneck minimizer + re-planning | GPipe |

## Package Structure

```
baselines/
├── __init__.py
├── train.py                    # Distributed training entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
├── README.md
│
├── core/                       # Abstract base classes + config
│   ├── config.py               # BaseConfig, ModelConfig, TrainingConfig, etc.
│   ├── state.py, compute.py, optimizer.py
│   ├── profiler.py, scheduler.py
│   ├── communication.py, fault_tolerance.py
│   └── orchestrator.py
│
├── models/                     # Model implementations
│   ├── registry.py             # MODEL_REGISTRY + create_block/create_head
│   ├── stage.py                # PipelineStage (nn.Module)
│   ├── gpt2.py                 # GPT-2 blocks
│   ├── llama.py                # Llama (RoPE + RMSNorm + SwiGLU + GQA)
│   ├── bert.py                 # BERT encoder blocks
│   └── hf_adapter.py           # HFModelAdapter (HuggingFace → pipeline stages)
│
├── strategies/                 # Parallelism strategy implementations
│   ├── base.py                 # ParallelismStrategy ABC
│   ├── dtfm_strategy.py        # GCMA + DP partition
│   ├── asteroid_strategy.py    # HPP planner
│   └── confident_strategy.py   # DP bottleneck minimizer
│
├── communication/              # Communication backends
│   ├── nccl.py                 # CuPy NCCL (GPU-direct P2P + AllReduce)
│   ├── torch_dist.py           # torch.distributed wrappers
│   └── gloo.py                 # CPU communication
│
├── fault_tolerance/            # Fault tolerance mechanisms
│   ├── basic_checkpoint.py     # Synchronous torch.save
│   ├── async_checkpoint.py     # Background CPU offload
│   ├── heartbeat.py            # Distributed store heartbeat
│   ├── passive_timeout.py      # Backward timeout detection
│   └── replication.py          # Weight replication strategies
│
├── schedulers/                 # Scheduling algorithms
│   ├── dp_partitioner.py       # DP-based layer partitioner
│   ├── gcma.py                 # GCMA evolutionary topology solver
│   ├── asteroid_planner.py     # HPP heterogeneity-aware planner
│   └── confident_scheduler.py  # Confident DP + re-plan
│
├── utils/                      # Shared utilities
│   ├── flatten.py              # Parameter flattening for AllReduce
│   ├── logging.py              # Event logger
│   ├── lr_schedule.py          # Cosine LR with warmup
│   ├── seed.py                 # Deterministic seeding
│   └── config_loader.py        # YAML → BaseConfig loader
│
├── configs/                    # Default YAML configurations
│   ├── dtfm_default.yaml
│   ├── asteroid_default.yaml
│   └── confident_default.yaml
│
├── examples/                   # Dry-run planning demos
│   ├── train_gpt2_dtfm.py
│   └── train_llama_asteroid.py
│
└── scripts/                    # Launch scripts
    ├── launch_single_node.sh   # mp.spawn single-node
    ├── launch_torchrun.sh      # torchrun elastic launcher
    ├── launch_docker.sh        # Docker GPU launcher
    └── launch_ssh.sh           # SSH multi-node (DT-FM style)
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA
- CuPy with CUDA support (`cupy-cuda12x`)
- PyYAML

Install dependencies:
```bash
pip install -r baselines/requirements.txt
```

Or use conda env `emulator` (if available):
```bash
conda activate emulator
```

### Dry-Run (Planning Only)

Test that the framework loads and planning works without GPUs:
```bash
python -m baselines.train \
    --strategy dtfm \
    --config baselines/configs/dtfm_default.yaml \
    --dry-run
```

### Single-Node Multi-GPU Training

#### Option 1: mp.spawn (simplest)

```bash
python -m baselines.train \
    --spawn \
    --num-gpus 4 \
    --strategy dtfm \
    --config baselines/configs/dtfm_default.yaml

# Or use the launch script:
bash baselines/scripts/launch_single_node.sh 4 dtfm
```

#### Option 2: torchrun (elastic)

```bash
torchrun --nproc_per_node=4 \
    -m baselines.train \
    --strategy dtfm \
    --config baselines/configs/dtfm_default.yaml \
    --world-size 4 \
    --dist-url tcp://127.0.0.1:29500

# Or use the launch script:
bash baselines/scripts/launch_torchrun.sh 4 dtfm
```

### Multi-Node Training

#### Via torchrun

On node 0:
```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
    bash baselines/scripts/launch_torchrun.sh 4 dtfm
```

On node 1:
```bash
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
    bash baselines/scripts/launch_torchrun.sh 4 dtfm
```

#### Via SSH (DT-FM style)

Edit `baselines/scripts/launch_ssh.sh` with your node IPs, then:
```bash
bash baselines/scripts/launch_ssh.sh dtfm
```

### Docker

Build:
```bash
docker build -t baselines:latest -f baselines/Dockerfile .
```

Run:
```bash
docker run --rm --gpus all --ipc=host baselines:latest \
    python -m baselines.train --spawn --num-gpus 4 --strategy dtfm

# Or via docker-compose:
docker compose -f baselines/docker-compose.yaml up --build
```

## Configuration

Configs are YAML files with these sections:

```yaml
model:
  model_name: "gpt2"       # Model architecture
  model_type: "gpt2"       # Registry key
  num_layers: 12            # Transformer layers
  embedding_dim: 768        # Hidden dimension
  num_heads: 12             # Attention heads
  seq_length: 1024          # Sequence length
  vocab_size: 50257         # Vocabulary size

training:
  batch_size: 64
  micro_batch_size: 4       # Per micro-batch
  lr: 3.0e-4
  max_iters: 500
  warmup_iters: 50
  grad_clip: 1.0

distributed:
  dist_backend: "nccl"
  dist_url: "tcp://127.0.0.1:29500"
  world_size: 4

parallelism:
  pp_size: 2                # Pipeline parallel stages
  dp_size: 2                # Data parallel replicas
  schedule_type: "gpipe"    # gpipe or 1f1b

fault_tolerance:
  checkpoint_dir: "./checkpoints/dtfm"
  checkpoint_interval: 100
```

See `baselines/configs/` for full examples.

## Communication Stack

The framework uses a dual-backend communication design (matching DT-FM):

1. **Control plane**: `torch.distributed` with Gloo backend
   - Process group creation (`new_group`)
   - Barriers, broadcasts
   - Rendezvous (TCP store)

2. **Data plane**: CuPy NCCL (GPU-direct)
   - P2P send/recv for pipeline activations/gradients
   - AllReduce for DP gradient synchronization
   - 3-stream async: compute, send, recv

## Strategies

### DT-FM (`--strategy dtfm`)
- **Planning**: GCMA evolutionary topology search → DP layer partitioner
- **Schedule**: GPipe (all-forward then all-backward)
- **Fault tolerance**: Basic sync checkpointing

### Asteroid (`--strategy asteroid`)
- **Planning**: HPP heterogeneity-aware DP planner
- **Schedule**: 1F1B (interleaved forward-backward)
- **Fault tolerance**: Heartbeat + passive timeout + weight replication

### Confident (`--strategy confident`)
- **Planning**: DP bottleneck minimizer with re-planning
- **Schedule**: GPipe
- **Fault tolerance**: Passive timeout + weight replication

## Adding New Models

Register a new model in `baselines/models/registry.py`:

```python
from baselines.models.registry import register_model

@register_model("my_model")
def my_block_factory(config):
    return MyTransformerBlock(config)
```

Or use the HuggingFace adapter:

```python
from baselines.models.hf_adapter import HFModelAdapter

adapter = HFModelAdapter("meta-llama/Llama-2-7b-hf")
stages = adapter.split_into_stages(num_stages=4)
```

## Reference Implementations

The original monolithic baselines are preserved as reference:
- `baselines/dtfm_gpt2_train copy (1).py` — DT-FM (3177 lines)
- `baselines/asteroid (1).py` — Asteroid (2144 lines)
- `baselines/Confident-production-FIXED (1) copy 2 (1).ipynb` — Confident

## Cluster Deployment (Asteroid)

Deployment uses Ansible playbooks under `deploy_asteroid/` and the
orchestration script `scripts/deploy_asteroid.sh`.

### Prerequisites

- K3s cluster with GPU nodes
- NVIDIA drivers installed on each node
- Ansible installed on the control node

### Quick start

```bash
# Full pipeline (bootstrap → K3s → GPU → profile → plan → build → deploy)
bash baselines/scripts/deploy_asteroid.sh all

# Or run individual phases
bash baselines/scripts/deploy_asteroid.sh k3s
bash baselines/scripts/deploy_asteroid.sh gpu
bash baselines/scripts/deploy_asteroid.sh profile
bash baselines/scripts/deploy_asteroid.sh plan
bash baselines/scripts/deploy_asteroid.sh build
bash baselines/scripts/deploy_asteroid.sh manifests
bash baselines/scripts/deploy_asteroid.sh deploy
```

### Key files

| File | Purpose |
|------|---------|
| `deploy_asteroid/inventory.ini` | Ansible inventory (auto-generated from YAML config) |
| `deploy_asteroid/setup_k3s.yaml` | Install / configure K3s cluster |
| `deploy_asteroid/setup_gpu.yaml` | nvidia-container-toolkit + NVIDIA device plugin |
| `deploy_asteroid/bootstrap_nodes.yaml` | Create venvs, install PyTorch on remote nodes |
| `deploy_asteroid/profile_and_gather.yaml` | Sync code, profile hardware, measure network |
| `deploy_asteroid/setup_cluster.yml` | Full end-to-end orchestration playbook |
| `deploy_asteroid/generate_manifests.py` | Generate K8s Job YAML from HPP plan |
| `deploy_asteroid/templates/stage_job.yaml.j2` | Jinja2 template for per-rank K8s Jobs |
| `requirements-deploy.txt` | Runtime deps installed on remote nodes |
| `configs/asteroid_default.yaml` | Cluster topology, model, training config |

