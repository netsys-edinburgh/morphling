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

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from baselines.core.config import (
    NodeInfo,
    ParallelismPlan,
)

# Try to import the full planner; fall back to
# synthetic plan if unavailable.
_HAS_PLANNER = False
try:
    from baselines.schedulers.asteroid_planner import (
        AsteroidPlanner,
    )

    _HAS_PLANNER = True
except ImportError:
    pass


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
    args = parser.parse_args()

    print("=" * 60)
    print("BASELINES HPP PLANNER")
    print("=" * 60)

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
        print(
            "Warning: AsteroidPlanner not "
            "available, using synthetic plan."
        )
        matched = match_profile_to_cluster(
            profiles, cluster_nodes,
        )
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
        plan = create_synthetic_plan(
            num_devices=world_size,
            num_stages=args.num_stages,
            num_layers=args.num_layers,
            cluster_nodes=cluster_nodes,
        )
        plan.node_mapping.update(node_mapping)
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
            planner = AsteroidPlanner(
                profiles=profiles,
                cluster_nodes=cluster_nodes,
                num_stages=args.num_stages,
                num_layers=args.num_layers,
            )
            plan = planner.plan()
            plan.node_mapping = (
                node_mapping_full
            )
        except Exception as e:
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
