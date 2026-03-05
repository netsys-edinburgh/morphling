#!/usr/bin/env python3
"""Baselines HPP Planner.

Aggregate node profiles and generate hpp_plan.json.

Usage:
    python run_asteroid_planner.py \
        --profiles-dir ./profiles \
        --output ./hpp_plan.json
    python run_asteroid_planner.py \
        --profiles-dir ./profiles \
        --num-stages 3 --num-layers 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# parents[2] = DeviceEmulator/ so "from baselines.core..." resolves
# (parents[1] = baselines/ which would look for baselines/baselines/)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)

from baselines.core.config import (
    DeviceConfig,
    DeviceTopology,
    ModelConfig,
    NodeInfo,
    ParallelismPlan,
)
from baselines.core.profiler import ProfilerBackend

# Try to import the full planner; fall back to
# synthetic plan if unavailable.
_HAS_PLANNER = False
try:
    from baselines.schedulers.asteroid_planner import (
        AsteroidHPPPlanner,
    )

    _HAS_PLANNER = True
except ImportError:
    pass

# Try to import strategy classes for dispatch
_HAS_STRATEGIES = False
try:
    from baselines.strategies.dtfm_strategy import (
        DTFMStrategy,
    )
    from baselines.strategies.asteroid_strategy import (
        AsteroidStrategy,
    )
    from baselines.strategies.confident_strategy import (
        ConfidentStrategy,
    )

    _HAS_STRATEGIES = True
except ImportError:
    pass


class _ProfileDataAdapter(ProfilerBackend):
    """Adapts JSON profile data to the ProfilerBackend
    interface used by strategy create_plan() methods.
    """

    def __init__(
        self,
        profiles: Dict[int, Dict[str, Any]],
        cluster_nodes: Dict[str, Dict[str, Any]],
        num_layers: int,
        embed_dim: int = 4096,
        d_ff: int = 11008,
        seq_len: int = 512,
    ) -> None:
        self._profiles = profiles
        self._num_layers = num_layers
        self._embed_dim = embed_dim
        self._d_ff = d_ff
        self._seq_len = seq_len
        # Pre-compute per-rank per-layer (fwd, bwd)
        self._exec: Dict[
            int, Dict[int, Tuple[float, float]]
        ] = {}
        for rank, pdata in profiles.items():
            lp = pdata.get("layer_profile", {})
            if not lp:
                continue
            # Use smallest batch-size entry as
            # the per-layer representative time
            best_bs = min(
                (int(k) for k in lp), default=1
            )
            metrics = lp.get(str(best_bs), {})
            combined = metrics.get(
                "latency_ms_mean", 3.0
            )
            fwd = combined / 3.0
            bwd = combined * 2.0 / 3.0
            rank_map: Dict[
                int, Tuple[float, float]
            ] = {}
            for li in range(num_layers):
                rank_map[li] = (fwd, bwd)
            self._exec[int(rank)] = rank_map

        # Pre-compute bandwidth map (rank→rank)
        ip_to_rank: Dict[str, int] = {}
        for ip, info in cluster_nodes.items():
            ip_to_rank[ip] = info["rank"]
        self._bw: Dict[int, float] = {}
        self._bw_pairs: Dict[
            Tuple[int, int], float
        ] = {}
        for rank, pdata in profiles.items():
            net = pdata.get("network", {})
            rank_bws: List[float] = []
            for peer, result in net.items():
                if not result.get("ok"):
                    continue
                bw_mbps = result.get(
                    "bandwidth_mbps", 0
                )
                peer_ip = peer.split(":")[0]
                if peer_ip in ip_to_rank:
                    dst = ip_to_rank[peer_ip]
                    self._bw_pairs[
                        (int(rank), dst)
                    ] = bw_mbps
                    rank_bws.append(bw_mbps)
            if rank_bws:
                self._bw[int(rank)] = min(
                    rank_bws
                )

    def profile_layer(
        self, model: object,
        input_data: object,
        num_iterations: int,
    ) -> Tuple[float, float, float]:
        return (1.0, 2.0, 0.0)

    def profile_bandwidth(
        self, src_device: object,
        dst_device: object,
        data_size_mb: float,
    ) -> float:
        # This method is not used in the
        # planning path — get_bandwidth() is
        # called instead.  Raise so we catch
        # any unexpected caller.
        raise NotImplementedError(
            "profile_bandwidth() should not be "
            "called; use get_bandwidth() with "
            "real iperf3 data."
        )

    def get_memory_info(
        self, device: object,
    ) -> Tuple[float, float]:
        return (40960.0, 40960.0)

    def get_time_interval(
        self, device_id: int,
        start: int, end: int,
        phase: int,
    ) -> float:
        rank_map = self._exec.get(device_id, {})
        total = 0.0
        for li in range(start, end + 1):
            pair = rank_map.get(li)
            if pair is None:
                return -1.0
            total += pair[phase]
        return total

    def get_output_size(
        self, layer_idx: int,
    ) -> float:
        # Activation size in MB (consistent
        # with _default_output_mb in the
        # strategy fallback path).
        total_bytes = float(
            self._seq_len
            * self._embed_dim
            * 4
        )
        return total_bytes / (1024.0 * 1024.0)

    def get_bandwidth(
        self, device_id: int,
    ) -> float:
        # Return MB/ms so that
        #   output_size_MB / bw_MB_per_ms = ms
        # Profile stores Mbps; convert:
        #   Mbps → MB/s (÷8) → MB/ms (÷1000)
        if device_id not in self._bw:
            raise RuntimeError(
                f"No iperf3 bandwidth data for "
                f"rank {device_id}. Re-run "
                f"profiling with --strict-iperf."
            )
        mbps = self._bw[device_id]
        return mbps / 8000.0

    def get_computing_capacity(
        self, device_id: int,
    ) -> float:
        return 1.0

    def get_available_memory(
        self, device_id: int,
    ) -> float:
        return 40960.0


def _build_strategy_plan(
    strategy_name: str,
    profiles: Dict[int, Dict[str, Any]],
    cluster_nodes: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    node_mapping: Dict[int, NodeInfo],
) -> Optional[ParallelismPlan]:
    """Dispatch to the correct strategy's create_plan
    based on ``strategy_name``.

    Returns None if the strategy is not available or
    if it should fall through to AsteroidHPPPlanner.
    """
    if not _HAS_STRATEGIES:
        return None
    if strategy_name not in (
        "dtfm", "confident",
    ):
        # 'asteroid' and unknown → use
        # AsteroidHPPPlanner (existing path)
        return None

    world_size = len(cluster_nodes)
    num_micro = max(
        1,
        args.global_batch_size
        // args.micro_batch_size,
    )

    # Build ModelConfig
    model_config = ModelConfig(
        num_layers=args.num_layers,
        embedding_dim=args.embed_dim,
        d_ff=args.d_ff,
        seq_length=args.seq_len,
        max_seq_len=args.seq_len,
    )

    # Build DeviceTopology from cluster data
    device_specs: List[DeviceConfig] = []
    for ip, info in cluster_nodes.items():
        r = info["rank"]
        mem = info.get(
            "memory_limit_mb",
            info.get("memory_mb", 46000),
        )
        pct = info.get(
            "active_thread_pct", 100
        )
        device_specs.append(
            DeviceConfig(
                device_id=r,
                memory_budget_mb=float(mem),
                compute_capacity=pct / 100.0,
            )
        )
    device_specs.sort(
        key=lambda d: d.device_id
    )
    # Build bandwidth/latency dicts
    bandwidths: Dict[
        Tuple[int, int], float
    ] = {}
    latencies: Dict[
        Tuple[int, int], float
    ] = {}
    ip_to_rank: Dict[str, int] = {}
    for ip, info in cluster_nodes.items():
        ip_to_rank[ip] = info["rank"]
    for rank, pdata in profiles.items():
        net = pdata.get("network", {})
        for peer, result in net.items():
            if not result.get("ok"):
                continue
            bw_mbps = result.get(
                "bandwidth_mbps", 0
            )
            peer_ip = peer.split(":")[0]
            if peer_ip in ip_to_rank:
                dst = ip_to_rank[peer_ip]
                bandwidths[
                    (int(rank), dst)
                ] = bw_mbps
                latencies[
                    (int(rank), dst)
                ] = 0.1
    # Validate all inter-rank links have measured
    # bandwidth.  Reject fallback / default values
    # so the planner uses only real iperf3 data.
    missing_links: List[Tuple[int, int]] = []
    for i in range(world_size):
        for j in range(world_size):
            if i != j and (i, j) not in bandwidths:
                missing_links.append((i, j))
    if missing_links:
        print(
            f"\n  ERROR: {len(missing_links)} "
            f"inter-rank bandwidth link(s) have "
            f"no iperf3 data:",
            file=sys.stderr,
        )
        for src, dst in sorted(missing_links):
            print(
                f"    rank {src} -> rank {dst}: "
                f"NO DATA",
                file=sys.stderr,
            )
        print(
            "\n  Re-run profiling phase to fix."
            "  Ensure iperf3 server is running "
            "on all nodes before profiling.",
            file=sys.stderr,
        )
        sys.exit(1)

    # All links present — fill latencies for any
    # that only have bandwidth.
    for i in range(world_size):
        for j in range(world_size):
            if i != j:
                latencies.setdefault(
                    (i, j), 0.1
                )

    topology = DeviceTopology(
        device_specs=device_specs,
        bandwidths=bandwidths,
        latencies=latencies,
    )

    # Build profiler adapter
    profiler = _ProfileDataAdapter(
        profiles=profiles,
        cluster_nodes=cluster_nodes,
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        d_ff=args.d_ff,
        seq_len=args.seq_len,
    )

    # Instantiate strategy
    pp = min(world_size, args.num_stages)
    dp = max(1, world_size // pp)

    if strategy_name == "dtfm":
        strategy = DTFMStrategy(
            pp_size=pp,
            dp_size=dp,
            global_batch_size=(
                args.global_batch_size
            ),
            micro_batch_size=(
                args.micro_batch_size
            ),
        )
    elif strategy_name == "confident":
        # Confident uses ALL devices as PP stages
        # (matches notebook ConfidantScheduler)
        strategy = ConfidentStrategy(
            pp_size=world_size,
            dp_size=1,
        )
    else:
        return None

    print(
        f"\n  Using {strategy_name.upper()} "
        f"strategy create_plan()"
    )
    plan = strategy.create_plan(
        model_config, topology, profiler,
    )
    # Attach node mapping
    plan.node_mapping = dict(node_mapping)
    return plan


def parse_cluster_conf(
    path: str,
) -> Dict[str, Dict[str, Any]]:
    """Parse cluster.conf for node info.

    Format: IP  NIC  RANK  GPU_ID
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        print(
            f"Warning: cluster.conf not found "
            f"at {path}"
        )
        return nodes

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                ip = parts[0]
                nic = parts[1]
                try:
                    rank = int(parts[2])
                    gpu_id = int(parts[3])
                except ValueError:
                    print(
                        f"Warning: bad rank/gpu_id "
                        f"on line {line_num}"
                    )
                    continue
                nodes[ip] = {
                    "ip": ip,
                    "nic": nic,
                    "rank": rank,
                    "gpu_id": gpu_id,
                }
    return nodes


def load_profiles(
    profiles_dir: str,
) -> Dict[int, Dict[str, Any]]:
    """Load profile_*.json files."""
    profiles: Dict[int, Dict[str, Any]] = {}
    profiles_path = Path(profiles_dir)
    if not profiles_path.exists():
        print(
            f"Error: Profiles directory not "
            f"found: {profiles_dir}"
        )
        return profiles

    for pf in profiles_path.glob("profile_*.json"):
        try:
            with open(pf, "r") as f:
                data = json.load(f)
            rank = data.get("rank")
            if rank is None:
                fname = pf.stem
                if "_rank" in fname:
                    try:
                        rank = int(
                            fname.split("_rank")[-1]
                        )
                    except ValueError:
                        rank = None
            if rank is None:
                print(
                    f"  Warning: No rank in "
                    f"{pf.name}, skipping"
                )
                continue
            rank = int(rank)
            hostname = data.get(
                "hostname", "unknown"
            )
            profiles[rank] = data
            print(
                f"  Loaded: {pf.name} "
                f"(hostname: {hostname}, "
                f"rank: {rank})"
            )
        except json.JSONDecodeError as e:
            print(f"  Error loading {pf}: {e}")
    return profiles


def match_profile_to_cluster(
    profiles: Dict[int, Dict[str, Any]],
    cluster_nodes: Dict[str, Dict[str, Any]],
) -> Dict[
    int, Tuple[str, Dict[str, Any], Dict[str, Any]]
]:
    """Match profile ranks to cluster.conf."""
    matched: Dict[
        int,
        Tuple[str, Dict[str, Any], Dict[str, Any]],
    ] = {}
    for ip, ci in cluster_nodes.items():
        rank = ci["rank"]
        if rank in profiles:
            hn = profiles[rank].get(
                "hostname", f"node-{rank}"
            )
            matched[rank] = (
                hn,
                profiles[rank],
                ci,
            )
            print(f"  Rank {rank}: {hn} -> {ip}")
        else:
            print(
                f"  Warning: No profile for "
                f"rank {rank} ({ip})"
            )
    return matched


def densify_ranks(
    matched: Dict[
        int,
        Tuple[str, Dict[str, Any], Dict[str, Any]],
    ],
    profiles: Dict[int, Dict[str, Any]],
    cluster_nodes: Dict[str, Dict[str, Any]],
) -> Tuple[
    Dict[
        int,
        Tuple[str, Dict[str, Any], Dict[str, Any]],
    ],
    Dict[int, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[int, int],
]:
    """Remap matched ranks to dense 0-based sequence.

    When only a subset of cluster nodes were profiled,
    the matched dict may have non-contiguous keys (e.g.,
    {1: ..., 3: ...}). This remaps everything to
    {0: ..., 1: ...} so the planner and manifest
    generator work with consistent 0-based device IDs.

    Returns:
        (dense_matched, dense_profiles,
         dense_cluster_nodes, old_to_new_rank_map)
    """
    sorted_old_ranks = sorted(matched.keys())
    # Check if already dense 0-based
    if sorted_old_ranks == list(
        range(len(sorted_old_ranks))
    ):
        return matched, profiles, cluster_nodes, {
            r: r for r in sorted_old_ranks
        }

    old_to_new = {
        old: new
        for new, old in enumerate(sorted_old_ranks)
    }
    print(
        f"  Remapping ranks: "
        f"{dict(zip(sorted_old_ranks, range(len(sorted_old_ranks))))}"
    )

    dense_matched: Dict[
        int,
        Tuple[str, Dict[str, Any], Dict[str, Any]],
    ] = {}
    dense_profiles: Dict[
        int, Dict[str, Any]
    ] = {}
    dense_cluster_nodes: Dict[
        str, Dict[str, Any]
    ] = {}
    for old_rank in sorted_old_ranks:
        new_rank = old_to_new[old_rank]
        hn, pd, ci = matched[old_rank]
        # Update cluster info rank
        ci_copy = dict(ci)
        ci_copy["rank"] = new_rank
        dense_matched[new_rank] = (
            hn, pd, ci_copy,
        )
        dense_profiles[new_rank] = pd
        # Find the IP for this node
        for ip, c in cluster_nodes.items():
            if c["rank"] == old_rank:
                dense_cluster_nodes[ip] = ci_copy
                break

    return (
        dense_matched,
        dense_profiles,
        dense_cluster_nodes,
        old_to_new,
    )


def build_profiler_data(
    profiles: Dict[int, Dict[str, Any]],
    cluster_nodes: Dict[str, Dict[str, Any]],
    num_layers: int,
    embed_dim: int = 4096,
    d_ff: int = 11008,
    seq_len: int = 512,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """Convert baselines profile format to
    AsteroidHPPPlanner's profiler_data + device_specs.

    Baselines profile:
        layer_profile[bs] = {latency_ms_mean, ...}
        network[peer:port] = {ok, bandwidth_mbps}
    Planner expects:
        exec_times[device][layer][bs] = (fwd, bwd)
        bandwidths[(src, dst)] = MB/s
        activation_sizes[layer] = bytes
        weight_sizes[layer] = bytes
    """
    exec_times: Dict[
        int, Dict[int, Dict[int, Tuple[float, float]]]
    ] = {}
    bandwidths: Dict[Tuple[int, int], float] = {}

    # Map IPs to ranks for bandwidth
    ip_to_rank: Dict[str, int] = {}
    for ip, info in cluster_nodes.items():
        ip_to_rank[ip] = info["rank"]

    for rank, pdata in profiles.items():
        rank_int = int(rank)
        layer_profile = pdata.get(
            "layer_profile", {}
        )

        # Build per-layer exec times (replicate
        # the single-block profile to all layers)
        layer_map: Dict[
            int, Dict[int, Tuple[float, float]]
        ] = {}
        for layer_idx in range(num_layers):
            bs_map: Dict[
                int, Tuple[float, float]
            ] = {}
            for bs_str, metrics in (
                layer_profile.items()
            ):
                bs = int(bs_str)
                combined_ms = metrics.get(
                    "latency_ms_mean", 3.0
                )
                # Split: ~1/3 forward, ~2/3 backward
                fwd_ms = combined_ms / 3.0
                bwd_ms = combined_ms * 2.0 / 3.0
                bs_map[bs] = (fwd_ms, bwd_ms)
            if bs_map:
                layer_map[layer_idx] = bs_map
        if layer_map:
            exec_times[rank_int] = layer_map

        # Network bandwidth
        network = pdata.get("network", {})
        for peer, result in network.items():
            if not result.get("ok"):
                continue
            bw_mbps = result.get(
                "bandwidth_mbps", 0
            )
            # Convert Mbps → MB/s
            bw_MBps = bw_mbps / 8.0
            peer_ip = peer.split(":")[0]
            if peer_ip in ip_to_rank:
                dst = ip_to_rank[peer_ip]
                bandwidths[
                    (rank_int, dst)
                ] = bw_MBps

    # Compute activation and weight sizes
    # (bytes per layer, model-accurate)
    activation_sizes: List[float] = []
    weight_sizes: List[float] = []
    # Assume 32 attention heads for LLaMA-7B
    num_heads = max(1, embed_dim // 128)
    for _ in range(num_layers):
        # Activation per sample per layer for backward
        # pass (stored intermediates):
        #  - 2 LayerNorm inputs:  2 * S * D * 4
        #  - Q,K,V projections:   3 * S * D * 4
        #  - Attention output:    S * D * 4
        #  - Softmax attn matrix: H * S * S * 4
        #  - FFN input:           S * D * 4
        #  - FFN intermediate:    2 * S * d_ff * 4
        #    (SwiGLU: up_proj + gate_proj)
        #  - Residual buffers:    2 * S * D * 4
        act_bytes = (
            seq_len * (
                9 * embed_dim
                + num_heads * seq_len
                + 2 * d_ff
            ) * 4
        )
        activation_sizes.append(float(act_bytes))
        # Weights: attention (4 projections) + FF
        # + layer norms (approx)
        # LLaMA SwiGLU uses 3 FF matrices
        # (up_proj, gate_proj, down_proj)
        attn_bytes = 4 * embed_dim * embed_dim * 4
        ff_bytes = 3 * embed_dim * d_ff * 4
        ln_bytes = 4 * embed_dim * 4
        weight_sizes.append(
            float(attn_bytes + ff_bytes + ln_bytes)
        )

    profiler_data: Dict[str, Any] = {
        "exec_times": exec_times,
        "bandwidths": bandwidths,
        "activation_sizes": activation_sizes,
        "weight_sizes": weight_sizes,
    }

    # Build device specs from cluster MPS info
    device_specs: List[Dict[str, Any]] = []
    for ip, info in cluster_nodes.items():
        r = info["rank"]
        mem_mb = info.get(
            "memory_limit_mb",
            info.get("memory_mb", 46000),
        )
        pct = info.get(
            "active_thread_pct", 100
        )
        cap = pct / 100.0
        device_specs.append(
            {
                "device_id": r,
                "memory_budget_mb": float(mem_mb),
                "compute_capacity": cap,
            }
        )
    device_specs.sort(key=lambda d: d["device_id"])

    return profiler_data, device_specs


def create_synthetic_plan(
    num_devices: int,
    num_stages: int,
    num_layers: int,
    cluster_nodes: Dict[str, Dict[str, Any]],
) -> ParallelismPlan:
    """Create a simple synthetic plan."""
    print("\nCreating synthetic plan...")

    devs_per = num_devices // num_stages
    remainder = num_devices % num_stages
    device_groups: Dict[int, list[int]] = {}
    cur = 0
    for s in range(num_stages):
        cnt = devs_per + (
            1 if s < remainder else 0
        )
        device_groups[s] = list(
            range(cur, cur + cnt)
        )
        cur += cnt

    layers_per = num_layers // num_stages
    partition_points = [
        i * layers_per
        for i in range(1, num_stages)
    ]

    micro_batch_alloc: Dict[
        int, Dict[int, int]
    ] = {}
    for s, devs in device_groups.items():
        micro_batch_alloc[s] = {
            d: 4 for d in devs
        }

    node_mapping: Dict[int, NodeInfo] = {}
    for ip, info in cluster_nodes.items():
        rank = info["rank"]
        if rank < num_devices:
            node_mapping[rank] = NodeInfo(
                hostname=f"node-{rank}",
                ip=ip,
                nic=info["nic"],
                gpu_id=info["gpu_id"],
                memory_mb=4096,
            )

    return ParallelismPlan(
        partition_points=partition_points,
        device_groups=device_groups,
        micro_batch_alloc=micro_batch_alloc,
        schedule_type="1f1b",
        estimated_latency_ms=0.0,
        node_mapping=node_mapping,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Baselines HPP Planner - "
            "Generate hpp_plan.json"
        ),
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="./profiles",
        help="Profile JSON directory",
    )
    parser.add_argument(
        "--cluster-conf",
        type=str,
        default="./cluster.conf",
        help="Path to cluster.conf",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./hpp_plan.json",
        help="Output path for hpp_plan.json",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=2,
        help="Number of pipeline stages",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="Number of model layers",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=4096,
        help="Model embedding dimension",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=11008,
        help="Feed-forward hidden dimension",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for sizing",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=4,
        help="Micro-batch size",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=256,
        help="Global batch size",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic plan only",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to asteroid YAML config "
        "(overrides cluster.conf)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Parallelism strategy "
        "(asteroid, uniform, etc.)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BASELINES HPP PLANNER")
    print("=" * 60)

    # Load cluster configuration
    # --config (YAML) takes precedence over --cluster-conf
    if args.config and os.path.exists(args.config):
        print(f"\nLoading config from {args.config}")
        import yaml

        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        cluster_section = yaml_cfg.get("cluster", {})
        nodes_list = cluster_section.get("nodes", [])
        mps_global = yaml_cfg.get("mps", {})
        cluster_nodes: Dict[str, Dict[str, Any]] = {}
        for i, node in enumerate(nodes_list):
            if isinstance(node, dict):
                ip = node.get("ip", f"127.0.0.{i}")
                node_mps = node.get("mps", {})
                cluster_nodes[ip] = {
                    "ip": ip,
                    "nic": node.get("nic", "eth0"),
                    "rank": node.get("rank", i),
                    "gpu_id": node.get("gpu_id", 0),
                    "hostname": node.get(
                        "hostname", f"node-{i}"
                    ),
                    "memory_mb": node.get(
                        "memory_mb", 4096
                    ),
                    "active_thread_pct": node_mps.get(
                        "active_thread_percentage",
                        mps_global.get(
                            "active_thread_percentage",
                            100,
                        ),
                    ),
                    "memory_limit_mb": node_mps.get(
                        "memory_limit_mb",
                        node.get("memory_mb", 4096),
                    ),
                }
            else:
                ip = str(node)
                cluster_nodes[ip] = {
                    "ip": ip,
                    "nic": "eth0",
                    "rank": i,
                    "gpu_id": 0,
                    "hostname": f"node-{i}",
                    "memory_mb": 4096,
                    "active_thread_pct": 100,
                    "memory_limit_mb": 4096,
                }
        print(f"  Found {len(cluster_nodes)} nodes")
        # Override args from YAML config sections
        model_cfg = yaml_cfg.get("model", {})
        par_cfg = yaml_cfg.get("parallelism", {})
        train_cfg = yaml_cfg.get("training", {})
        if model_cfg.get("num_layers"):
            args.num_layers = model_cfg["num_layers"]
        if model_cfg.get("embedding_dim"):
            args.embed_dim = model_cfg[
                "embedding_dim"
            ]
        if model_cfg.get("d_ff"):
            args.d_ff = model_cfg["d_ff"]
        if model_cfg.get("max_seq_len"):
            args.seq_len = model_cfg["max_seq_len"]
        if par_cfg.get("num_stages"):
            args.num_stages = par_cfg["num_stages"]
        if (
            par_cfg.get("strategy")
            and args.strategy is None
        ):
            args.strategy = par_cfg["strategy"]
        if train_cfg.get("micro_batch_size"):
            args.micro_batch_size = (
                train_cfg["micro_batch_size"]
            )
        if train_cfg.get("global_batch_size"):
            args.global_batch_size = (
                train_cfg["global_batch_size"]
            )
    else:
        print(
            f"\nLoading cluster config from "
            f"{args.cluster_conf}"
        )
        cluster_nodes = parse_cluster_conf(
            args.cluster_conf
        )
        print(
            f"  Found {len(cluster_nodes)} node(s)"
        )

    print(
        f"\nLoading profiles from "
        f"{args.profiles_dir}"
    )
    profiles = load_profiles(args.profiles_dir)
    print(f"  Loaded {len(profiles)} profile(s)")

    world_size = max(
        len(cluster_nodes), len(profiles), 1
    )

    plan: ParallelismPlan
    if not profiles:
        if args.synthetic or not cluster_nodes:
            plan = create_synthetic_plan(
                num_devices=world_size,
                num_stages=args.num_stages,
                num_layers=args.num_layers,
                cluster_nodes=cluster_nodes,
            )
        else:
            print(
                "Error: No profiles found. "
                "Run profile_and_gather.yaml "
                "first, or use --synthetic."
            )
            return 1
    elif not _HAS_PLANNER:
        # AsteroidHPPPlanner unavailable — try
        # strategy dispatch first, then synthetic.
        matched = match_profile_to_cluster(
            profiles, cluster_nodes,
        )
        matched, profiles, cluster_nodes, _ = (
            densify_ranks(
                matched, profiles,
                cluster_nodes,
            )
        )
        world_size = len(matched)
        node_mapping: Dict[int, NodeInfo] = {}
        for rank, (hn, pd, ci) in (
            matched.items()
        ):
            hw = pd.get("hardware", {})
            node_mapping[rank] = NodeInfo(
                hostname=hn,
                ip=ci["ip"],
                nic=ci["nic"],
                gpu_id=ci["gpu_id"],
                memory_mb=hw.get(
                    "gpu_memory_mb", 4096
                ),
                architecture=hw.get(
                    "architecture", "x86_64"
                ),
            )
        strategy_name = (
            args.strategy or "asteroid"
        ).lower()
        strategy_plan = _build_strategy_plan(
            strategy_name=strategy_name,
            profiles=profiles,
            cluster_nodes=cluster_nodes,
            args=args,
            node_mapping=node_mapping,
        )
        if strategy_plan is not None:
            plan = strategy_plan
        else:
            print(
                "Warning: AsteroidPlanner not "
                "available, using synthetic plan."
            )
            plan = create_synthetic_plan(
                num_devices=world_size,
                num_stages=args.num_stages,
                num_layers=args.num_layers,
                cluster_nodes=cluster_nodes,
            )
            plan.node_mapping.update(
                node_mapping
            )
    else:
        matched = match_profile_to_cluster(
            profiles, cluster_nodes,
        )
        if not matched:
            print(
                "Error: Could not match "
                "profiles to cluster."
            )
            return 1
        matched, profiles, cluster_nodes, _ = (
            densify_ranks(
                matched, profiles,
                cluster_nodes,
            )
        )
        world_size = len(matched)
        # Build node mapping from matched data
        node_mapping_full: Dict[
            int, NodeInfo
        ] = {}
        for rank, (hn, pd, ci) in (
            matched.items()
        ):
            hw = pd.get("hardware", {})
            node_mapping_full[rank] = NodeInfo(
                hostname=hn,
                ip=ci["ip"],
                nic=ci["nic"],
                gpu_id=ci["gpu_id"],
                memory_mb=hw.get(
                    "gpu_memory_mb", 4096
                ),
                architecture=hw.get(
                    "architecture", "x86_64"
                ),
            )
        try:
            num_micro = max(
                1,
                args.global_batch_size
                // args.micro_batch_size,
            )
            profiler_data, device_specs = (
                build_profiler_data(
                    profiles=profiles,
                    cluster_nodes=cluster_nodes,
                    num_layers=args.num_layers,
                    embed_dim=args.embed_dim,
                    d_ff=args.d_ff,
                    seq_len=args.seq_len,
                )
            )
            print(
                f"\n  Planner input: "
                f"L={args.num_layers} "
                f"N={world_size} "
                f"M={num_micro} "
                f"micro_bs={args.micro_batch_size}"
            )
            print(
                f"  Device specs: "
                f"{len(device_specs)} devices"
            )
            print(
                f"  Exec times: "
                f"{len(profiler_data.get('exec_times', {}))} "
                f"ranks profiled"
            )
            bw = profiler_data.get(
                "bandwidths", {}
            )
            print(
                f"  Bandwidths: "
                f"{len(bw)} links"
            )

            strategy_name = (
                args.strategy or "asteroid"
            ).lower()
            strategy_plan = (
                _build_strategy_plan(
                    strategy_name=strategy_name,
                    profiles=profiles,
                    cluster_nodes=cluster_nodes,
                    args=args,
                    node_mapping=(
                        node_mapping_full
                    ),
                )
            )
            if strategy_plan is not None:
                plan = strategy_plan
            else:
                # Default: AsteroidHPPPlanner
                print(
                    "\n  Using ASTEROID "
                    "HPP planner"
                )
                planner = AsteroidHPPPlanner(
                    num_layers=args.num_layers,
                    num_devices=world_size,
                    num_microbatches=num_micro,
                    micro_batch_size=(
                        args.micro_batch_size
                    ),
                    profiler_data=profiler_data,
                    device_specs=device_specs,
                )
                raw_plan = planner.plan()

                # Convert raw dict → ParallelismPlan
                plan = ParallelismPlan(
                    partition_points=list(
                        raw_plan.get(
                            "partition_points",
                            [],
                        )
                    ),
                    device_groups={
                        int(k): [
                            int(d) for d in v
                        ]
                        for k, v in raw_plan.get(
                            "device_groups", {}
                        ).items()
                    },
                    micro_batch_alloc={
                        int(k): {
                            int(d): int(n)
                            for d, n in (
                                v.items()
                            )
                        }
                        for k, v in raw_plan.get(
                            "micro_batch_alloc",
                            {},
                        ).items()
                    },
                    schedule_type="1f1b",
                    estimated_latency_ms=float(
                        raw_plan.get(
                            "latency", 0.0
                        )
                    ),
                    node_mapping=(
                        node_mapping_full
                    ),
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(
                f"Warning: Planner failed "
                f"({e}), using fallback"
            )
            plan = create_synthetic_plan(
                num_devices=world_size,
                num_stages=args.num_stages,
                num_layers=args.num_layers,
                cluster_nodes=cluster_nodes,
            )
            plan.node_mapping.update(
                node_mapping_full
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True, exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(plan.to_json(), f, indent=2)

    print("\n" + "=" * 60)
    print("HPP PLAN GENERATED")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(
        f"Stages: "
        f"{len(plan.partition_points) + 1}"
    )
    print(
        f"Partition points: "
        f"{plan.partition_points}"
    )
    print("Device groups:")
    for stage, devices in (
        plan.device_groups.items()
    ):
        print(f"  Stage {stage}: devices {devices}")
    print(
        f"Estimated latency: "
        f"{plan.estimated_latency_ms:.2f} ms"
    )
    ws = sum(
        len(d)
        for d in plan.device_groups.values()
    )
    print(f"World size: {ws}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
