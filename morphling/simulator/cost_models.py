from __future__ import annotations

import math
from typing import List

from .config import DeviceSpec, TopologyConfig


def compute_time_s(flops_count: float, device_flops: float) -> float:
    return flops_count / device_flops


def ring_allreduce_time_s(
    msg_bytes: float,
    devices: List[DeviceSpec],
    ring_order: List[int] | None = None,
) -> dict[int, float]:
    n = len(devices)
    if n <= 1:
        return {d.rank: 0.0 for d in devices}

    order = ring_order or [d.rank for d in devices]
    dev_by_rank = {d.rank: d for d in devices}

    # Ring: 2*(N-1) steps. Each step transfers msg_bytes/N through one link.
    # Bottleneck link determines per-step transfer time.
    chunk_bytes = msg_bytes / n
    max_step_time = 0.0
    for i in range(len(order)):
        sender = dev_by_rank[order[i]]
        receiver = dev_by_rank[order[(i + 1) % n]]
        link_bw = min(sender.ul_bw_bytes_per_s, receiver.dl_bw_bytes_per_s)
        link_lat = sender.ul_lat_s + receiver.dl_lat_s
        step_time = link_lat + chunk_bytes / link_bw
        max_step_time = max(max_step_time, step_time)

    total_time = 2 * (n - 1) * max_step_time
    return {d.rank: total_time for d in devices}


def tree_allreduce_time_s(
    msg_bytes: float,
    devices: List[DeviceSpec],
) -> dict[int, float]:
    n = len(devices)
    if n <= 1:
        return {d.rank: 0.0 for d in devices}

    depth = math.ceil(math.log2(n))

    min_ul_bw = min(d.ul_bw_bytes_per_s for d in devices)
    min_dl_bw = min(d.dl_bw_bytes_per_s for d in devices)
    bottleneck_bw = min(min_ul_bw, min_dl_bw)
    max_lat = max(d.ul_lat_s + d.dl_lat_s for d in devices)

    # reduce-scatter: depth steps, then allgather: depth steps
    step_time = max_lat + msg_bytes / bottleneck_bw
    total_time = 2 * depth * step_time
    return {d.rank: total_time for d in devices}


def ps_push_time_s(
    msg_bytes: float,
    device: DeviceSpec,
    num_servers: int,
    server_dl_bw: float,
) -> float:
    bytes_per_server = msg_bytes / num_servers
    send_time = device.ul_lat_s + bytes_per_server / device.ul_bw_bytes_per_s
    recv_time = bytes_per_server / server_dl_bw
    return max(send_time, recv_time)


def ps_aggregate_time_s(
    msg_bytes: float,
    num_workers: int,
    num_servers: int,
    server_flops: float,
) -> float:
    # reduce: sum across workers, element-wise. 1 FLOP per element per worker.
    bytes_per_server = msg_bytes / num_servers
    elements_per_server = bytes_per_server / 4
    agg_flops = elements_per_server * num_workers
    return agg_flops / server_flops


def ps_pull_time_s(
    msg_bytes: float,
    device: DeviceSpec,
    num_servers: int,
    server_ul_bw: float,
) -> float:
    bytes_per_server = msg_bytes / num_servers
    send_time = bytes_per_server / server_ul_bw
    recv_time = device.dl_lat_s + bytes_per_server / device.dl_bw_bytes_per_s
    return max(send_time, recv_time)


def ps_total_time_s(
    msg_bytes: float,
    devices: List[DeviceSpec],
    topo: TopologyConfig,
) -> dict[int, float]:
    n = len(devices)
    if n <= 1:
        return {d.rank: 0.0 for d in devices}

    result = {}
    for dev in devices:
        push = ps_push_time_s(
            msg_bytes, dev, topo.num_ps_servers, topo.ps_server_dl_bw
        )
        agg = ps_aggregate_time_s(
            msg_bytes, n, topo.num_ps_servers, topo.ps_server_flops
        )
        pull = ps_pull_time_s(
            msg_bytes, dev, topo.num_ps_servers, topo.ps_server_ul_bw
        )
        result[dev.rank] = push + agg + pull
    return result


def gradient_sync_time_s(
    msg_bytes: float,
    devices: List[DeviceSpec],
    topo: TopologyConfig,
) -> dict[int, float]:
    if topo.mode == "allreduce":
        if topo.allreduce_algo == "ring":
            return ring_allreduce_time_s(msg_bytes, devices, topo.ring_order)
        return tree_allreduce_time_s(msg_bytes, devices)
    elif topo.mode == "ps":
        return ps_total_time_s(msg_bytes, devices, topo)
    elif topo.mode == "hybrid":
        return _hybrid_time_s(msg_bytes, devices, topo)
    raise ValueError(f"Unknown topology mode: {topo.mode}")


def _hybrid_time_s(
    msg_bytes: float,
    devices: List[DeviceSpec],
    topo: TopologyConfig,
) -> dict[int, float]:
    groups = topo.device_groups
    if not groups:
        return gradient_sync_time_s(
            msg_bytes,
            devices,
            TopologyConfig(
                mode=topo.intra_group_algo, allreduce_algo=topo.allreduce_algo
            ),
        )

    dev_by_rank = {d.rank: d for d in devices}
    result: dict[int, float] = {d.rank: 0.0 for d in devices}

    intra_topo = TopologyConfig(
        mode=topo.intra_group_algo, allreduce_algo=topo.allreduce_algo
    )
    for group_ranks in groups:
        group_devices = [dev_by_rank[r] for r in group_ranks]
        intra_times = gradient_sync_time_s(msg_bytes, group_devices, intra_topo)
        for rank, t in intra_times.items():
            result[rank] += t

    group_leaders = [dev_by_rank[g[0]] for g in groups]
    inter_topo = TopologyConfig(
        mode=topo.inter_group_algo,
        allreduce_algo=topo.allreduce_algo,
        num_ps_servers=topo.num_ps_servers,
        ps_server_flops=topo.ps_server_flops,
        ps_server_dl_bw=topo.ps_server_dl_bw,
        ps_server_ul_bw=topo.ps_server_ul_bw,
    )
    inter_times = gradient_sync_time_s(msg_bytes, group_leaders, inter_topo)

    inter_max = max(inter_times.values()) if inter_times else 0.0
    for rank in result:
        result[rank] += inter_max

    return result
