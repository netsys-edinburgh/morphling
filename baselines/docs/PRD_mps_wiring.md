# PRD: Wire MPS Config End-to-End

## Problem

The `asteroid_default.yaml` defines per-node `mps.active_thread_percentage` under
`cluster.nodes[]` and a global `mps.enabled` switch, but **neither value reaches
the planner or the deployed training pods**:

| Gap | Location | Current behaviour | Required behaviour |
|-----|----------|-------------------|--------------------|
| G1 | `config_loader.py` | Sets `cluster=ClusterConfig()` (empty) | Parse `cluster.nodes[]` including per-node MPS |
| G2 | `train.py:_build_topology()` | Hardcodes `compute_capacity=1.0` for every device | Read per-node MPS → `compute_capacity = pct/100` when `mps.enabled`, else `1.0` |
| G3 | `generate_manifests.py` + `stage_job.yaml.j2` | No `--enable-mps` or `--mps-thread-pct` args | Pass per-rank MPS CLI args to training command |
| G4 | `deploy_asteroid.sh:phase_mps()` | Applies single global `thread_pct` to ALL nodes | Apply each node's individual `active_thread_percentage` |
| G5 | MPS enabled switch | Only checked in `phase_mps()` gate | Must gate everything: topology capacity, deploy args, MPS daemon setup |

## Design

### 1. `ClusterConfig` + `NodeMpsConfig` (config.py)

Add `NodeMpsConfig` dataclass with `active_thread_percentage: int = 100` and
`memory_limit_mb: int = 0`.

Extend `ClusterConfig` to hold:
- `mps_enabled: bool = False`
- `nodes: list[ClusterNodeConfig]` where `ClusterNodeConfig` has `ip`, `hostname`,
  `rank`, `mps: NodeMpsConfig`, etc.

### 2. `config_loader.py` — parse cluster nodes + global MPS

In `_parse_config()`:
1. Read `raw["mps"]["enabled"]` → `ClusterConfig.mps_enabled`
2. Read `raw["cluster"]["nodes"]` → list of `ClusterNodeConfig`
3. Per node: merge `node.mps.active_thread_percentage` with global
   `mps.active_thread_percentage` as fallback.

### 3. `train.py:_build_topology()` — capacity from MPS

Change signature: `_build_topology(world_size, cfg) -> DeviceTopology`

Logic:
```python
mps_enabled = cfg.cluster.mps_enabled
for rank in range(world_size):
    if mps_enabled and rank in node_pct_map:
        capacity = node_pct_map[rank] / 100.0
    else:
        capacity = 1.0
```

### 4. `generate_manifests.py` + `stage_job.yaml.j2`

In `generate_job_manifest()`:
- Read per-rank `active_thread_percentage` from `cluster_nodes` in `extra_vars`
- Read `mps_enabled` from `extra_vars`
- Pass `mps_enabled` and `mps_thread_pct` to template variables

In template, conditionally add args:
```yaml
{% if mps_enabled %}
- "--enable-mps"
- "--mps-thread-pct"
- "{{ mps_thread_pct }}"
{% endif %}
```

In `main()`:
- Read `mps.enabled` from the asteroid config
- Add to `extra_vars`

### 5. `deploy_asteroid.sh:phase_mps()` — per-node

Replace single `run_ansible_adhoc "cluster"` with per-node loop:
```bash
for each node in cluster.nodes:
    ssh to node.ip
    set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = node.mps.active_thread_percentage
```

Gate entire function on `mps.enabled`.

### 6. MPS enabled switch semantics

When `mps.enabled = false`:
- `_build_topology()` → all `compute_capacity = 1.0` (homogeneous)
- `generate_manifests` → no `--enable-mps` / `--mps-thread-pct` args
- `deploy_asteroid.sh phase_mps()` → skip (already gated)
- Training worker → does NOT start MPS client env injection

When `mps.enabled = true`:
- Everything wired per-node

## Files Changed

1. `baselines/core/config.py` — add `NodeMpsConfig`, `ClusterNodeConfig`, extend `ClusterConfig`
2. `baselines/utils/config_loader.py` — parse `cluster.nodes[]` + `mps.*`
3. `baselines/train.py` — update `_build_topology()` signature + capacity logic
4. `baselines/deploy_asteroid/generate_manifests.py` — pass MPS vars per rank
5. `baselines/deploy_asteroid/templates/stage_job.yaml.j2` — conditional MPS args
6. `baselines/scripts/deploy_asteroid.sh` — per-node MPS in `phase_mps()`
