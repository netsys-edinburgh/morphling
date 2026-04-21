#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

PHASES = {"SEND", "DOWNLOAD", "COMPUTE", "UPLOAD", "RECEIVE"}
ALLREDUCE_TYPES = {"dp_allreduce", "tp_allreduce"}


@dataclass
class VTimeEvent:
    timestamp_us: int
    device_id: int
    gemm_id: int
    phase: str
    event: str
    vt_start_us: int
    vt_end_us: int
    vt_duration_us: int


@dataclass
class BatchRuntimeResult:
    baseline_name: str
    total_runtime_ms: float
    compute_time_ms: float
    network_time_ms: float
    pipeline_bubble_ms: float
    allreduce_time_ms: float
    optimizer_tail_ms: float
    per_device_breakdown: dict[int, dict[str, float]]
    per_level_breakdown: list[dict[str, Any]]


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def parse_vtime_log(path: str) -> list[VTimeEvent]:
    events: list[VTimeEvent] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or not stripped.startswith("VTIME"):
                continue

            parts = stripped.split(",")
            if len(parts) not in (8, 9):
                continue

            try:
                if len(parts) == 9:
                    timestamp_us = _to_int(parts[1])
                    device_id = _to_int(parts[2])
                    gemm_id = _to_int(parts[3], -1)
                    phase = parts[4].strip().upper()
                    event = parts[5].strip().upper()
                    vt_start_us = _to_int(parts[6])
                    vt_end_us = _to_int(parts[7])
                    vt_duration_us = _to_int(parts[8])
                else:
                    timestamp_us = _to_int(parts[1])
                    device_id = _to_int(parts[2])
                    gemm_id = -1
                    phase = parts[3].strip().upper()
                    event = parts[4].strip().upper()
                    vt_start_us = _to_int(parts[5])
                    vt_end_us = _to_int(parts[6])
                    vt_duration_us = _to_int(parts[7])

                if event != "END":
                    continue
                if phase not in PHASES:
                    continue

                events.append(
                    VTimeEvent(
                        timestamp_us=timestamp_us,
                        device_id=device_id,
                        gemm_id=gemm_id,
                        phase=phase,
                        event=event,
                        vt_start_us=vt_start_us,
                        vt_end_us=vt_end_us,
                        vt_duration_us=vt_duration_us,
                    )
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse VTIME line {line_no}: {stripped}"
                ) from exc

    return events


def _extract_manifest_entries(manifest: Any) -> list[dict[str, Any]]:
    if isinstance(manifest, list):
        return [x for x in manifest if isinstance(x, dict)]

    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a dict or list")

    list_keys = (
        "entries",
        "dispatch_entries",
        "dispatches",
        "sequence",
        "items",
    )
    for key in list_keys:
        value = manifest.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    levels = manifest.get("levels")
    if isinstance(levels, list):
        out: list[dict[str, Any]] = []
        for level_item in levels:
            if not isinstance(level_item, dict):
                continue
            level = level_item.get("level")
            level_entries = level_item.get("entries", [])
            if not isinstance(level_entries, list):
                continue
            for raw in level_entries:
                if not isinstance(raw, dict):
                    continue
                merged = dict(raw)
                merged.setdefault("level", level)
                out.append(merged)
        if out:
            return out

    value_dicts = [v for v in manifest.values() if isinstance(v, dict)]
    if value_dicts and all("gemm_id" in v for v in value_dicts):
        return value_dicts

    raise ValueError("Could not locate dispatch entries in manifest JSON")


def _normalize_entry(raw: dict[str, Any]) -> dict[str, Any]:
    parallelism_type = str(
        raw.get("parallelism_type", raw.get("type", ""))
    ).lower()
    return {
        "gemm_id": _to_int(raw.get("gemm_id"), -1),
        "level": _to_int(raw.get("level"), 0),
        "device_id": _to_int(raw.get("device_id"), -1),
        "alpha": _to_float(raw.get("alpha"), 0.0),
        "beta": _to_float(raw.get("beta"), 0.0),
        "n": _to_float(raw.get("n"), 0.0),
        "m_total": _to_float(raw.get("m_total"), 0.0),
        "q_total": _to_float(raw.get("q_total"), 0.0),
        "is_local": _to_bool(raw.get("is_local"), False),
        "stage_id": _to_int(raw.get("stage_id"), 0),
        "parallelism_type": parallelism_type,
        "raw": raw,
    }


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_device_profiles(path: str | None) -> dict[int, dict[str, float]]:
    if not path:
        return {}

    raw = _load_json(path)
    profiles: dict[int, dict[str, float]] = {}

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("devices"), list):
            items = raw["devices"]
        else:
            items = [v for v in raw.values() if isinstance(v, dict)]
    else:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        rank = _to_int(item.get("rank", item.get("device_id")), -1)
        if rank < 0:
            continue
        profiles[rank] = {
            "flops": _to_float(item.get("flops"), 0.0),
            "ul_bw": _to_float(item.get("ul_bw"), 0.0),
            "dl_bw": _to_float(item.get("dl_bw"), 0.0),
            "ul_lat": _to_float(item.get("ul_lat"), 0.0),
            "dl_lat": _to_float(item.get("dl_lat"), 0.0),
        }
    return profiles


def _build_entry_lookup(
    entries: list[dict[str, Any]],
) -> tuple[
    dict[tuple[int, int], dict[str, Any]],
    dict[int, list[dict[str, Any]]],
]:
    by_device_gemm: dict[tuple[int, int], dict[str, Any]] = {}
    by_gemm: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for entry in entries:
        key = (entry["device_id"], entry["gemm_id"])
        by_device_gemm[key] = entry
        by_gemm[entry["gemm_id"]].append(entry)

    return by_device_gemm, by_gemm


def _pick_entry_for_event(
    event: VTimeEvent,
    by_device_gemm: dict[tuple[int, int], dict[str, Any]],
    by_gemm: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    exact = by_device_gemm.get((event.device_id, event.gemm_id))
    if exact is not None:
        return exact

    candidates = by_gemm.get(event.gemm_id, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _entry_data_bytes(entry: dict[str, Any], bytes_per_element: float) -> float:
    raw = entry["raw"]
    for k in ("data_bytes", "gradient_bytes", "bytes"):
        v = raw.get(k)
        if v is not None:
            candidate = _to_float(v, 0.0)
            if candidate > 0:
                return candidate

    alpha = max(_to_float(entry.get("alpha"), 0.0), 0.0)
    beta = max(_to_float(entry.get("beta"), 0.0), 0.0)
    if alpha > 0 and beta > 0:
        return alpha * beta * bytes_per_element

    m_total = max(_to_float(entry.get("m_total"), 0.0), 0.0)
    q_total = max(_to_float(entry.get("q_total"), 0.0), 0.0)
    if m_total > 0 and q_total > 0:
        return m_total * q_total * bytes_per_element

    return 0.0


def _collective_group_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    raw = entry["raw"]
    group_id = raw.get("group_id")
    if group_id is None:
        group_id = raw.get("dp_group_id")
    if group_id is None:
        group_id = raw.get("tp_group_id")
    return (
        entry["parallelism_type"],
        entry["level"],
        entry["stage_id"],
        group_id,
    )


def _build_collective_groups(
    entries: list[dict[str, Any]],
) -> dict[tuple[Any, ...], set[int]]:
    groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for entry in entries:
        if entry["parallelism_type"] not in ALLREDUCE_TYPES:
            continue
        groups[_collective_group_key(entry)].add(entry["device_id"])
    return groups


def _estimate_allreduce_ms(
    entry: dict[str, Any],
    entries_in_group: set[int],
    device_profiles: dict[int, dict[str, float]],
    bytes_per_element: float,
    allreduce_bandwidth_bps: float | None,
) -> float:
    raw = entry["raw"]

    explicit_k = raw.get("group_size")
    if explicit_k is None and entry["parallelism_type"] == "dp_allreduce":
        explicit_k = raw.get("dp_size")
    if explicit_k is None and entry["parallelism_type"] == "tp_allreduce":
        explicit_k = raw.get("tp_size")

    k = _to_int(explicit_k, 0)
    if k <= 0:
        k = len(entries_in_group)
    if k <= 1:
        return 0.0

    data_bytes = _entry_data_bytes(entry, bytes_per_element)
    if data_bytes <= 0:
        return 0.0

    bw = None
    explicit_bw = raw.get("min_bandwidth_bps", raw.get("bandwidth_bps"))
    if explicit_bw is not None:
        explicit_bw_f = _to_float(explicit_bw, 0.0)
        if explicit_bw_f > 0:
            bw = explicit_bw_f

    if bw is None and allreduce_bandwidth_bps and allreduce_bandwidth_bps > 0:
        bw = allreduce_bandwidth_bps

    if bw is None and device_profiles and entries_in_group:
        group_bws: list[float] = []
        for d in entries_in_group:
            profile = device_profiles.get(d)
            if not profile:
                continue
            ul = profile.get("ul_bw", 0.0)
            dl = profile.get("dl_bw", 0.0)
            if ul > 0 and dl > 0:
                group_bws.append(min(ul, dl))
        if group_bws:
            bw = min(group_bws)

    if bw is None or bw <= 0:
        return 0.0

    sec = 2.0 * (k - 1) / k * data_bytes / bw
    return sec * 1000.0


def _compute_analytical_phase_us(
    entry: dict[str, Any],
    device_profiles: dict[int, dict[str, float]],
    bytes_per_element: float,
) -> dict[str, float]:
    profile = device_profiles.get(entry["device_id"], {})

    flops = _to_float(profile.get("flops"), 0.0)
    dl_bw = _to_float(profile.get("dl_bw"), 0.0)
    ul_bw = _to_float(profile.get("ul_bw"), 0.0)
    dl_lat = _to_float(profile.get("dl_lat"), 0.0)
    ul_lat = _to_float(profile.get("ul_lat"), 0.0)

    alpha = max(_to_float(entry.get("alpha"), 0.0), 0.0)
    beta = max(_to_float(entry.get("beta"), 0.0), 0.0)
    n = max(_to_float(entry.get("n"), 0.0), 0.0)

    compute_s = 0.0
    if flops > 0 and alpha > 0 and beta > 0 and n > 0:
        compute_s = 2.0 * alpha * beta * n / flops

    download_s = 0.0
    if dl_bw > 0 and alpha > 0 and beta > 0 and n > 0:
        download_bytes = (
            alpha * n * bytes_per_element + n * beta * bytes_per_element
        )
        download_s = download_bytes / dl_bw + dl_lat

    upload_s = 0.0
    if ul_bw > 0 and alpha > 0 and beta > 0:
        upload_bytes = alpha * beta * bytes_per_element
        upload_s = upload_bytes / ul_bw + ul_lat

    return {
        "SEND": 0.0,
        "DOWNLOAD": download_s * 1_000_000.0,
        "COMPUTE": compute_s * 1_000_000.0,
        "UPLOAD": upload_s * 1_000_000.0,
        "RECEIVE": 0.0,
    }


def _infer_pipeline_params(
    manifest: dict[str, Any] | list[dict[str, Any]],
    entries: list[dict[str, Any]],
    pp_size_arg: int | None,
    num_microbatches_arg: int | None,
) -> tuple[int, int]:
    pp_size = pp_size_arg
    num_microbatches = num_microbatches_arg

    if pp_size is None and isinstance(manifest, dict):
        pp_size = _to_int(manifest.get("pp_size"), 0)
    if num_microbatches is None and isinstance(manifest, dict):
        num_microbatches = _to_int(manifest.get("num_microbatches"), 0)

    if pp_size is None or pp_size <= 0:
        stages = sorted({entry["stage_id"] for entry in entries})
        pp_size = len(stages) if stages else 1
    if num_microbatches is None or num_microbatches <= 0:
        num_microbatches = 1

    return pp_size, num_microbatches


def _baseline_uses_pipeline_overlay(
    baseline_type: str,
    pp_size: int,
) -> bool:
    b = baseline_type.lower()
    if b == "cleave":
        return False
    if "pp" in b or b in {"dtfm", "asteroid", "confident", "alpa"}:
        return True
    return pp_size > 1


def compute_batch_runtime(
    vtime_events: list[VTimeEvent],
    manifest: dict[str, Any] | list[dict[str, Any]],
    baseline_type: str,
    *,
    pp_size: int | None = None,
    num_microbatches: int | None = None,
    optimizer_tail_ms: float | None = None,
    device_profiles: dict[int, dict[str, float]] | None = None,
    bytes_per_element: float = 2.0,
    allreduce_bandwidth_bps: float | None = None,
) -> BatchRuntimeResult:
    device_profiles = device_profiles or {}
    baseline = baseline_type.lower()

    raw_entries = _extract_manifest_entries(manifest)
    entries = [_normalize_entry(x) for x in raw_entries]
    if not entries:
        raise ValueError("Manifest has no dispatch entries")

    by_device_gemm, by_gemm = _build_entry_lookup(entries)
    collective_groups = _build_collective_groups(entries)

    phase_us_by_key: dict[
        tuple[int, int, int, str, int, bool],
        dict[str, float],
    ] = defaultdict(lambda: defaultdict(float))

    if vtime_events:
        # Build VTIME phase durations keyed by gemm_id.  The emulator
        # dispatches each unique GEMM once (via _unique_gemms_by_level)
        # regardless of how many manifest devices reference it.  We
        # map the measured durations to *every* manifest entry that
        # shares the gemm_id so the per-device / per-stage accounting
        # reflects all devices in the original placement.
        vtime_phases: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for ev in vtime_events:
            vtime_phases[ev.gemm_id][ev.phase] += max(ev.vt_duration_us, 0)

        for entry in entries:
            phases = vtime_phases.get(entry["gemm_id"])
            if phases is None:
                continue
            key = (
                entry["level"],
                entry["device_id"],
                entry["gemm_id"],
                entry["parallelism_type"],
                entry["stage_id"],
                entry["is_local"],
            )
            for phase, dur_us in phases.items():
                phase_us_by_key[key][phase] += dur_us
    else:
        for entry in entries:
            key = (
                entry["level"],
                entry["device_id"],
                entry["gemm_id"],
                entry["parallelism_type"],
                entry["stage_id"],
                entry["is_local"],
            )
            synthetic = _compute_analytical_phase_us(
                entry,
                device_profiles,
                bytes_per_element,
            )
            for phase, dur_us in synthetic.items():
                phase_us_by_key[key][phase] += max(dur_us, 0.0)

    per_device_breakdown: dict[int, dict[str, float]] = defaultdict(
        lambda: {"compute_ms": 0.0, "network_ms": 0.0, "total_ms": 0.0}
    )
    level_device_runtime_ms: dict[int, dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    stage_device_runtime_ms: dict[int, dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    stage_compute_ms: dict[int, float] = defaultdict(float)
    stage_network_ms: dict[int, float] = defaultdict(float)

    compute_time_ms = 0.0
    network_time_ms = 0.0
    allreduce_time_ms = 0.0
    pipeline_bubble_ms = 0.0

    for key, phase_map in phase_us_by_key.items():
        level, device_id, _gemm_id, ptype, stage_id, _is_local = key

        send_ms = phase_map.get("SEND", 0.0) / 1000.0
        dl_ms = phase_map.get("DOWNLOAD", 0.0) / 1000.0
        comp_ms = phase_map.get("COMPUTE", 0.0) / 1000.0
        ul_ms = phase_map.get("UPLOAD", 0.0) / 1000.0
        recv_ms = phase_map.get("RECEIVE", 0.0) / 1000.0
        xfer_ms = send_ms + dl_ms + ul_ms + recv_ms

        if baseline == "cleave":
            if ptype != "cleave_tp":
                continue

            effective_ms = max(dl_ms, comp_ms, ul_ms) + send_ms + recv_ms
            compute_time_ms += comp_ms
            network_time_ms += xfer_ms

            per_device_breakdown[device_id]["compute_ms"] += comp_ms
            per_device_breakdown[device_id]["network_ms"] += xfer_ms
            per_device_breakdown[device_id]["total_ms"] += effective_ms

            level_device_runtime_ms[level][device_id] += effective_ms
            stage_device_runtime_ms[stage_id][device_id] += effective_ms
            stage_compute_ms[stage_id] += comp_ms
            stage_network_ms[stage_id] += xfer_ms
            continue

        included_compute_ms = 0.0
        included_network_ms = 0.0

        if ptype == "pp_local":
            included_compute_ms = comp_ms
            included_network_ms = xfer_ms
        elif ptype == "pp_boundary":
            included_network_ms = comp_ms + xfer_ms
        elif ptype in ALLREDUCE_TYPES:
            pass
        else:
            pass

        included_total_ms = included_compute_ms + included_network_ms
        compute_time_ms += included_compute_ms
        network_time_ms += included_network_ms

        per_device_breakdown[device_id]["compute_ms"] += included_compute_ms
        per_device_breakdown[device_id]["network_ms"] += included_network_ms
        per_device_breakdown[device_id]["total_ms"] += included_total_ms

        level_device_runtime_ms[level][device_id] += included_total_ms
        stage_device_runtime_ms[stage_id][device_id] += included_total_ms
        stage_compute_ms[stage_id] += included_compute_ms
        stage_network_ms[stage_id] += included_network_ms

    seen_collectives_for_walltime: set[tuple[Any, ...]] = set()
    for entry in entries:
        ptype = entry["parallelism_type"]
        if ptype not in ALLREDUCE_TYPES:
            continue

        group_key = _collective_group_key(entry)
        group_devices = collective_groups.get(group_key, set())
        ar_ms = _estimate_allreduce_ms(
            entry,
            group_devices,
            device_profiles,
            bytes_per_element,
            allreduce_bandwidth_bps,
        )
        if ar_ms <= 0:
            continue

        level = entry["level"]
        stage_id = entry["stage_id"]
        device_id = entry["device_id"]

        per_device_breakdown[device_id]["network_ms"] += ar_ms
        per_device_breakdown[device_id]["total_ms"] += ar_ms

        level_device_runtime_ms[level][device_id] += ar_ms
        stage_device_runtime_ms[stage_id][device_id] += ar_ms
        stage_network_ms[stage_id] += ar_ms

        if group_key not in seen_collectives_for_walltime:
            seen_collectives_for_walltime.add(group_key)
            allreduce_time_ms += ar_ms

    per_level_breakdown: list[dict[str, Any]] = []
    level_times_ms: list[float] = []

    for level in sorted(level_device_runtime_ms.keys()):
        device_times = level_device_runtime_ms[level]
        if not device_times:
            continue
        bottleneck_device, runtime_ms = max(
            device_times.items(),
            key=lambda kv: kv[1],
        )
        level_times_ms.append(runtime_ms)
        per_level_breakdown.append(
            {
                "level": level,
                "runtime_ms": runtime_ms,
                "bottleneck_device": bottleneck_device,
            }
        )

    inferred_optimizer_tail_ms = 0.0
    if isinstance(manifest, dict):
        inferred_optimizer_tail_ms = _to_float(
            manifest.get("optimizer_tail_ms"),
            0.0,
        )
    optimizer_tail_val_ms = (
        inferred_optimizer_tail_ms
        if optimizer_tail_ms is None
        else max(optimizer_tail_ms, 0.0)
    )

    total_runtime_ms = 0.0
    if baseline == "cleave":
        total_runtime_ms = sum(level_times_ms) + optimizer_tail_val_ms
    else:
        pp_size_inferred, num_microbatches_inferred = _infer_pipeline_params(
            manifest,
            entries,
            pp_size,
            num_microbatches,
        )

        if _baseline_uses_pipeline_overlay(baseline, pp_size_inferred):
            if stage_device_runtime_ms:
                bottleneck_stage_time_ms = max(
                    max(dev_times.values())
                    for dev_times in stage_device_runtime_ms.values()
                    if dev_times
                )
            else:
                bottleneck_stage_time_ms = 0.0

            pipeline_bubble_ms = (
                max(pp_size_inferred - 1, 0) * bottleneck_stage_time_ms
            )
            total_runtime_ms = (
                num_microbatches_inferred * bottleneck_stage_time_ms
                + pipeline_bubble_ms
            )
        else:
            total_runtime_ms = sum(level_times_ms)

    per_device_breakdown_out: dict[int, dict[str, float]] = {}
    for device_id in sorted(per_device_breakdown.keys()):
        vals = per_device_breakdown[device_id]
        per_device_breakdown_out[device_id] = {
            "compute_ms": vals["compute_ms"],
            "network_ms": vals["network_ms"],
            "total_ms": vals["total_ms"],
        }

    return BatchRuntimeResult(
        baseline_name=baseline,
        total_runtime_ms=total_runtime_ms,
        compute_time_ms=compute_time_ms,
        network_time_ms=network_time_ms,
        pipeline_bubble_ms=pipeline_bubble_ms,
        allreduce_time_ms=allreduce_time_ms,
        optimizer_tail_ms=optimizer_tail_val_ms
        if baseline == "cleave"
        else 0.0,
        per_device_breakdown=per_device_breakdown_out,
        per_level_breakdown=per_level_breakdown,
    )


def _make_output_payload(
    result: BatchRuntimeResult,
    used_vtime_log: bool,
    manifest_path: str,
    baseline_type: str,
) -> dict[str, Any]:
    payload = asdict(result)
    payload["total_runtime_s"] = result.total_runtime_ms / 1000.0
    payload["compute_time_s"] = result.compute_time_ms / 1000.0
    payload["network_time_s"] = result.network_time_ms / 1000.0
    payload["pipeline_bubble_s"] = result.pipeline_bubble_ms / 1000.0
    payload["allreduce_time_s"] = result.allreduce_time_ms / 1000.0
    payload["optimizer_tail_s"] = result.optimizer_tail_ms / 1000.0
    payload["mode"] = "vtime" if used_vtime_log else "analytical"
    payload["manifest_path"] = manifest_path
    payload["baseline_type"] = baseline_type
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-batch runtime from VTIME log + dispatch manifest",
    )
    parser.add_argument(
        "--vtime-log",
        type=str,
        default=None,
        help="Path to merged perf log with VTIME entries",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to dispatch manifest JSON",
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        required=True,
        help="Baseline type (e.g., cleave, dtfm, asteroid, confident, alpa)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=None,
        help="Pipeline parallel size for 1F1B overlay",
    )
    parser.add_argument(
        "--num-microbatches",
        type=int,
        default=None,
        help="Number of microbatches for 1F1B overlay",
    )
    parser.add_argument(
        "--device-config",
        type=str,
        default=None,
        help="Device profile JSON (required for accurate analytical fallback)",
    )
    parser.add_argument(
        "--optimizer-tail-ms",
        type=float,
        default=None,
        help="Optional override for CLEAVE optimizer tail in ms",
    )
    parser.add_argument(
        "--bytes-per-element",
        type=float,
        default=2.0,
        help="Element size in bytes for analytical/fallback byte estimation",
    )
    parser.add_argument(
        "--allreduce-bandwidth-bps",
        type=float,
        default=None,
        help="Override min bandwidth (B/s) used in allreduce ring model",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _load_json(str(manifest_path))
    device_profiles = _load_device_profiles(args.device_config)

    used_vtime_log = bool(args.vtime_log)
    if args.vtime_log:
        vtime_path = Path(args.vtime_log)
        if not vtime_path.exists():
            raise FileNotFoundError(f"VTIME log not found: {vtime_path}")
        vtime_events = parse_vtime_log(str(vtime_path))
    else:
        vtime_events = []

    result = compute_batch_runtime(
        vtime_events=vtime_events,
        manifest=manifest,
        baseline_type=args.baseline_type,
        pp_size=args.pp_size,
        num_microbatches=args.num_microbatches,
        optimizer_tail_ms=args.optimizer_tail_ms,
        device_profiles=device_profiles,
        bytes_per_element=args.bytes_per_element,
        allreduce_bandwidth_bps=args.allreduce_bandwidth_bps,
    )

    payload = _make_output_payload(
        result,
        used_vtime_log=used_vtime_log,
        manifest_path=str(manifest_path),
        baseline_type=args.baseline_type,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved batch runtime to: {output_path}")
    else:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
