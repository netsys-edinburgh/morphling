#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from morphling._Msg import ProxySvr


@dataclass(frozen=True)
class DispatchEntry:
    gemm_id: int
    level: int
    device_id: int
    alpha: int
    beta: int
    n: int
    m_total: int
    q_total: int

    @classmethod
    def from_dict(cls, d: dict) -> "DispatchEntry":
        return cls(
            gemm_id=int(d.get("gemm_id", 0)),
            level=int(d.get("level", 0)),
            device_id=int(d.get("device_id", 0)),
            alpha=int(d.get("alpha", 1)),
            beta=int(d.get("beta", 1)),
            n=int(d.get("n", 1)),
            m_total=int(d.get("m_total", 0)),
            q_total=int(d.get("q_total", 0)),
        )


@dataclass
class DispatchManifest:
    baseline_name: str
    entries: list
    metadata: dict

    @classmethod
    def from_json(cls, data: dict) -> "DispatchManifest":
        raw_entries = data.get("entries", [])
        entries = [
            DispatchEntry.from_dict(e)
            for e in raw_entries
            if isinstance(e, dict)
        ]
        return cls(
            baseline_name=str(data.get("baseline_name", "")),
            entries=entries,
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class ReplayArgs:
    manifest: str
    cfg: str
    num_devices: int | None
    timeout: int
    output_log: str
    server_log: str
    max_inflight: int
    seed: int


def parse_args() -> ReplayArgs:
    parser = argparse.ArgumentParser(
        description="Replay a dispatch manifest via ProxySvr"
    )
    _ = parser.add_argument("--manifest", required=True, type=str)
    _ = parser.add_argument("--cfg", default="config/proxy/svr.ini", type=str)
    _ = parser.add_argument("--num-devices", default=None, type=int)
    _ = parser.add_argument("--timeout", default=300, type=int)
    _ = parser.add_argument("--output-log", required=True, type=str)
    _ = parser.add_argument(
        "--server-log", default="logs/perf_server.log", type=str
    )
    _ = parser.add_argument("--max-inflight", default=5, type=int)
    _ = parser.add_argument("--seed", default=42, type=int)
    ns = parser.parse_args()
    raw_num_devices = ns.num_devices
    num_devices = (
        int(raw_num_devices)
        if isinstance(raw_num_devices, int) and raw_num_devices > 0
        else None
    )
    return ReplayArgs(
        manifest=str(ns.manifest),
        cfg=str(ns.cfg),
        num_devices=num_devices,
        timeout=int(ns.timeout),
        output_log=str(ns.output_log),
        server_log=str(ns.server_log),
        max_inflight=int(ns.max_inflight),
        seed=int(ns.seed),
    )


def _load_manifest(path: Path) -> DispatchManifest:
    payload_obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    payload = {str(key): value for key, value in payload_obj.items()}
    return DispatchManifest.from_json(payload)


def _required_devices(manifest: DispatchManifest, override: int | None) -> int:
    if override is not None and override > 0:
        return override
    metadata_num = int(manifest.metadata.get("num_devices", 0) or 0)
    if metadata_num > 0:
        return metadata_num
    ids = {
        entry.device_id for entry in manifest.entries if entry.device_id >= 0
    }
    return max(1, len(ids))


@dataclass(frozen=True)
class GemmSpec:
    gemm_id: int
    level: int
    m: int
    n: int
    q: int


def _unique_gemms_by_level(
    manifest: DispatchManifest,
) -> dict[int, list[GemmSpec]]:
    seen: dict[tuple[int, int], GemmSpec] = {}
    for entry in manifest.entries:
        key = (entry.level, entry.gemm_id)
        if key not in seen:
            seen[key] = GemmSpec(
                gemm_id=entry.gemm_id,
                level=entry.level,
                m=int(getattr(entry, "m_total", 0) or entry.alpha),
                n=int(entry.n),
                q=int(getattr(entry, "q_total", 0) or entry.beta),
            )
    grouped: dict[int, list[GemmSpec]] = {}
    for spec in sorted(seen.values(), key=lambda s: (s.level, s.gemm_id)):
        grouped.setdefault(spec.level, []).append(spec)
    return grouped


def _dispatch_level(
    backend: Any,
    gemms: list[GemmSpec],
    gen: torch.Generator,
    global_oid: int,
) -> tuple[int, int]:
    oid = global_oid
    oids: list[int] = []
    # Keep tensor references alive until all wait_matmul calls
    # complete.  The C++ DispatchMatMulAsync stores raw void*
    # pointers into the tensor data; the scatter-gather send
    # happens asynchronously on the event-loop thread, so the
    # Python tensors must outlive the dispatch+wait cycle.
    tensor_refs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for g in gemms:
        m, n, q = g.m, g.n, g.q
        if m <= 0 or n <= 0 or q <= 0:
            continue
        mat_a = torch.rand((m, n), dtype=torch.float32, generator=gen)
        # C++ CalculateMatrixPartition assumes both matrices
        # share h_dim = a.shape[-1] as their last dimension
        # (the GEMM is C = A @ B^T).  Create B as (q, n) so
        # that b.shape[-1] == n == h_dim.
        mat_b = torch.rand((q, n), dtype=torch.float32, generator=gen)
        backend.async_dispatch_matmul(mat_a, mat_b)
        tensor_refs.append((mat_a, mat_b))
        oids.append(oid)
        oid += 1

    for wait_oid in oids:
        _ = backend.wait_matmul(wait_oid)
    return len(oids), oid


def _copy_vtime_log(source_log: Path, output_log: Path) -> None:
    output_log.parent.mkdir(parents=True, exist_ok=True)
    if source_log.exists():
        _ = shutil.copy2(source_log, output_log)
        return
    candidates = sorted(
        source_log.parent.glob(source_log.name + ".*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        _ = shutil.copy2(candidates[0], output_log)
        print(f"Copied rotated log: {candidates[0]}", flush=True)
        return
    raise FileNotFoundError(
        f"Server VTIME log not found: {source_log} (also checked rotated)"
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_log = Path(args.output_log)
    source_log = Path(args.server_log)
    cfg_path = args.cfg
    timeout_s = args.timeout
    max_inflight = args.max_inflight
    seed = args.seed
    num_devices_override = args.num_devices

    manifest = _load_manifest(manifest_path)
    required_devices = _required_devices(manifest, num_devices_override)
    grouped = _unique_gemms_by_level(manifest)

    print(
        f"Loaded manifest '{manifest_path}' baseline={manifest.baseline_name} "
        f"entries={len(manifest.entries)} levels={len(grouped)}"
    )
    print(f"Waiting for required devices: {required_devices}")

    if source_log.exists():
        source_log.unlink()

    import configparser
    import tempfile

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    if not cfg.has_section("worker"):
        cfg.add_section("worker")
    cfg.set("worker", "num_device", str(required_devices))
    cfg.set("worker", "block_size", "4096")
    if not cfg.has_section("internal"):
        cfg.add_section("internal")
    cfg.set("internal", "send_high_water_mark", "67108864")
    if not cfg.has_section("scalability"):
        cfg.add_section("scalability")
    cfg.set("scalability", "barrier_count", str(required_devices))
    cfg.set("scalability", "barrier_timeout", "600000")

    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False, prefix="svr_replay_"
    )
    cfg.write(tmp_cfg)
    tmp_cfg.flush()
    effective_cfg = tmp_cfg.name
    print(f"Generated config: {effective_cfg} (num_device={required_devices})")

    backend = ProxySvr()
    backend.initialize(effective_cfg)
    backend.start()

    start_wait = time.time()
    connected = 0
    while time.time() - start_wait < timeout_s:
        connected = backend.get_connection_count()
        elapsed = int(time.time() - start_wait)
        print(f"[{elapsed}s] Connected devices: {connected}/{required_devices}")
        if connected >= required_devices:
            break
        time.sleep(2)
    if connected < required_devices:
        raise RuntimeError(
            f"Insufficient devices connected: {connected}/{required_devices}"
        )

    gen = torch.Generator(device="cpu")
    _ = gen.manual_seed(seed)

    total_dispatched = 0
    global_oid = 0
    for level in sorted(grouped.keys()):
        level_gemms = grouped[level]
        print(
            f"Dispatching level {level}: {len(level_gemms)} unique GEMMs",
            flush=True,
        )
        dispatched, global_oid = _dispatch_level(
            backend=backend,
            gemms=level_gemms,
            gen=gen,
            global_oid=global_oid,
        )
        total_dispatched += dispatched
        print(
            f"Completed level {level}: dispatched {dispatched}",
            flush=True,
        )

    if hasattr(backend, "flush_perf_log"):
        backend.flush_perf_log()
    time.sleep(4.0)
    if hasattr(backend, "flush_perf_log"):
        backend.flush_perf_log()
    _copy_vtime_log(source_log=source_log, output_log=output_log)
    print(
        f"Replay complete. Total dispatched GEMMs={total_dispatched}. "
        f"VTIME log copied to {output_log}"
    )


if __name__ == "__main__":
    main()
