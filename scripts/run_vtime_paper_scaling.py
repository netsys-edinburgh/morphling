#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportDeprecated=false, reportImplicitStringConcatenation=false, reportUnusedCallResult=false
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASELINES = ["cleave", "dtfm", "asteroid", "confident", "alpa"]
SWEEP_DIMS = ("num_devices", "model_size", "batch_size")

MODEL_PARAMS_B: dict[str, float] = {
    "opt-125m": 0.125,
    "opt-1.3b": 1.3,
    "opt-13b": 13.0,
    "opt-30b": 30.0,
    "opt-66b": 66.0,
    "llama2-7b": 7.0,
    "llama2-13b": 13.0,
    "llama2-70b": 70.0,
}

PAPER_NUM_DEVICES = [64, 128, 256, 512, 1024]
PAPER_BATCH_SIZES = [16, 32, 64, 128, 256]
PAPER_MODEL_ORDER = [
    "opt-125m",
    "opt-1.3b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama2-7b",
    "llama2-13b",
    "llama2-70b",
]


@dataclass(frozen=True)
class SweepPoint:
    sweep_dim: str
    point_index: int
    param_value: int | str
    num_devices: int
    model_size: str
    batch_size: int
    seq_len: int
    num_microbatches: int
    seed: int


@dataclass(frozen=True)
class PointPaths:
    point_dir: Path
    result_path: Path
    fleet_path: Path
    planning_dir: Path
    vtime_root: Path
    runtimes_root: Path


@dataclass(frozen=True)
class BackfillTask:
    row_index: int
    point: SweepPoint
    paths: PointPaths
    missing_baselines: tuple[str, ...]
    existing_payload: dict[str, Any] | None
    rebuild: bool


def _canonical_baseline_name(name: str) -> str:
    return name.strip().lower()


def _slugify(value: Any) -> str:
    text = str(value)
    text = text.replace("/", "_")
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_") or "value"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_csv_tokens(raw_items: Sequence[str] | None) -> list[str]:
    items: list[str] = []
    if raw_items:
        for chunk in raw_items:
            items.extend(x.strip() for x in chunk.split(",") if x.strip())
    return items


def _parse_baselines(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in (x.strip() for x in raw.split(",") if x.strip()):
        canon = _canonical_baseline_name(token)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
    return out


def _parse_sweep_dims(raw_dims: Sequence[str] | None) -> list[str]:
    if not raw_dims:
        return list(SWEEP_DIMS)

    dims = _parse_csv_tokens(raw_dims)
    if not dims:
        return list(SWEEP_DIMS)

    out: list[str] = []
    seen: set[str] = set()
    for dim in dims:
        if dim not in SWEEP_DIMS:
            allowed = ", ".join(SWEEP_DIMS)
            raise ValueError(f"Unknown --sweep-dim '{dim}'. Allowed: {allowed}")
        if dim in seen:
            continue
        seen.add(dim)
        out.append(dim)
    return out


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in (x.strip() for x in raw.split(",") if x.strip()):
        out.append(int(token))
    if not out:
        raise ValueError("Expected at least one integer value")
    return out


def _build_device_fleet(num_devices: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    fleet: list[dict[str, Any]] = []
    for rank in range(num_devices):
        fleet.append(
            {
                "rank": rank,
                "flops": int(rng.uniform(5.0e12, 7.0e12)),
                "memory": int(rng.uniform(1.0e9, 3.0e9)),
                "ul_bw": float(rng.uniform(5.0e6, 10.0e6)),
                "dl_bw": float(rng.uniform(10.0e6, 100.0e6)),
                "ul_lat": 0.0,
                "dl_lat": 0.0,
            }
        )
    return fleet


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _resolve_output_root(output_dir: str, repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    results_root = (repo_root / "results").resolve()
    raw = Path(output_dir)

    def _ensure_inside_results(candidate: Path) -> Path:
        resolved = candidate.resolve()
        if resolved == results_root or results_root in resolved.parents:
            return resolved
        raise ValueError(
            "VTIME output must be inside repo_root/results because the Docker "
            "runner only mounts results/."
        )

    if raw.is_absolute():
        return _ensure_inside_results(raw)

    if raw.parts and raw.parts[0] == "results":
        return _ensure_inside_results(repo_root / raw)

    return _ensure_inside_results(results_root / raw)


def _run_command(
    cmd: Sequence[str],
    cwd: Path,
    *,
    tolerate_kill: bool = False,
) -> None:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return
    # Docker cleanup in run_vtime_experiment.sh can SIGKILL the container
    # after the server finishes, producing exit code 137 even on success.
    if tolerate_kill and proc.returncode in (137, -9):
        return
    stdout = proc.stdout.strip() or "<empty stdout>"
    stderr = proc.stderr.strip() or "<empty stderr>"
    raise RuntimeError(
        "Command failed:\n"
        f"{' '.join(cmd)}\n"
        f"stdout:\n{stdout}\n\n"
        f"stderr:\n{stderr}"
    )


def _load_optimizer_tail_ms(
    planning_dir: Path,
    baseline: str,
) -> float | None:
    runtime_json = planning_dir / "runtimes" / f"{baseline}_runtime.json"
    if not runtime_json.exists():
        return None
    try:
        payload = json.loads(runtime_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("optimizer_tail_ms")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_total_runtime_ms(runtime_json: Path) -> float | None:
    if not runtime_json.exists():
        return None
    try:
        payload = json.loads(runtime_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("total_runtime_ms")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _model_size_devices(
    model_size: str,
    base_devices_opt13b: int,
    max_devices: int,
) -> int:
    params = MODEL_PARAMS_B[model_size]
    base = MODEL_PARAMS_B["opt-13b"]
    scaled = base_devices_opt13b * (params / base)
    rounded = int(round(scaled / 8.0) * 8)
    rounded = max(8, rounded)
    rounded = min(max_devices, rounded)
    return rounded


def _build_points_for_dim(
    dim: str,
    args: argparse.Namespace,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []

    if dim == "num_devices":
        values = _parse_int_list(args.num_devices_values)
        for idx, num_devices in enumerate(values):
            batch_size = 128
            points.append(
                SweepPoint(
                    sweep_dim=dim,
                    point_index=idx,
                    param_value=num_devices,
                    num_devices=int(num_devices),
                    model_size="opt-13b",
                    batch_size=batch_size,
                    seq_len=args.seq_len,
                    num_microbatches=max(1, batch_size // args.micro_batch_size),
                    seed=args.seed,
                )
            )
        return points

    if dim == "batch_size":
        values = _parse_int_list(args.batch_size_values)
        for idx, batch_size in enumerate(values):
            num_devices = max(1, int(batch_size // 2))
            points.append(
                SweepPoint(
                    sweep_dim=dim,
                    point_index=idx,
                    param_value=batch_size,
                    num_devices=num_devices,
                    model_size="opt-13b",
                    batch_size=int(batch_size),
                    seq_len=args.seq_len,
                    num_microbatches=max(
                        1, int(batch_size) // args.micro_batch_size
                    ),
                    seed=args.seed,
                )
            )
        return points

    if dim == "model_size":
        model_values = _parse_csv_tokens(args.model_size_values.split(","))
        if not model_values:
            raise ValueError("--model-size-values must not be empty")
        for idx, model_size in enumerate(model_values):
            if model_size not in MODEL_PARAMS_B:
                allowed = ", ".join(sorted(MODEL_PARAMS_B.keys()))
                raise ValueError(
                    f"Unknown model size '{model_size}'. Allowed: {allowed}"
                )
            batch_size = 128
            num_devices = _model_size_devices(
                model_size=model_size,
                base_devices_opt13b=args.model_size_base_devices,
                max_devices=args.model_size_max_devices,
            )
            points.append(
                SweepPoint(
                    sweep_dim=dim,
                    point_index=idx,
                    param_value=model_size,
                    num_devices=num_devices,
                    model_size=model_size,
                    batch_size=batch_size,
                    seq_len=args.seq_len,
                    num_microbatches=max(1, batch_size // args.micro_batch_size),
                    seed=args.seed,
                )
            )
        return points

    raise ValueError(f"Unsupported sweep dim: {dim}")


def _normalize_rows(
    rows: Sequence[dict[str, Any]],
    baselines: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        baseline_map = row.get("baselines", {})
        if not isinstance(baseline_map, dict):
            baseline_map = {}
        cleave_entry = baseline_map.get("cleave")
        cleave = None
        if isinstance(cleave_entry, dict):
            raw = cleave_entry.get("total_runtime_ms")
            if isinstance(raw, (int, float)):
                cleave = float(raw)

        normalized_baselines: dict[str, float | None] = {}
        for baseline in baselines:
            runtime = None
            entry = baseline_map.get(baseline)
            if isinstance(entry, dict):
                raw = entry.get("total_runtime_ms")
                if isinstance(raw, (int, float)):
                    runtime = float(raw)
            if cleave is None or cleave <= 0 or runtime is None:
                normalized_baselines[baseline] = None
            else:
                normalized_baselines[baseline] = runtime / cleave

        out.append(
            {
                "param_value": row.get("param_value"),
                "baselines": normalized_baselines,
            }
        )
    return out


def _point_seed(point: SweepPoint) -> int:
    return point.seed + (point.point_index + 1) * 1009


def _point_paths(point: SweepPoint, output_root: Path) -> PointPaths:
    point_tag = f"{point.point_index:03d}_{_slugify(point.param_value)}"
    point_dir = output_root / point.sweep_dim / "points" / point_tag
    return PointPaths(
        point_dir=point_dir,
        result_path=point_dir / "result.json",
        fleet_path=point_dir / "generated_device_fleet.json",
        planning_dir=point_dir / "planning",
        vtime_root=point_dir / "vtime",
        runtimes_root=point_dir / "vtime_runtimes",
    )


def _load_existing_payload(result_path: Path) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _runtime_from_entry(entry: Any) -> float | None:
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        value = entry.get("total_runtime_ms")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _point_config_matches(
    payload: dict[str, Any],
    point: SweepPoint,
    baselines: Sequence[str],
    micro_batch_size: int,
    point_seed: int,
) -> bool:
    if payload.get("sweep_dim") != point.sweep_dim:
        return False
    if payload.get("param_value") != point.param_value:
        return False

    config = payload.get("config")
    if not isinstance(config, dict):
        return False

    expected_fields: dict[str, Any] = {
        "num_devices": point.num_devices,
        "model_size": point.model_size,
        "batch_size": point.batch_size,
        "seq_len": point.seq_len,
        "num_microbatches": point.num_microbatches,
        "micro_batch_size": micro_batch_size,
        "seed": point_seed,
    }
    for key, expected in expected_fields.items():
        if config.get(key) != expected:
            return False

    raw_baselines = config.get("baselines")
    if not isinstance(raw_baselines, list):
        return False
    existing_baselines = {_canonical_baseline_name(str(x)) for x in raw_baselines}
    requested_baselines = {_canonical_baseline_name(x) for x in baselines}
    return existing_baselines == requested_baselines


def _row_from_payload(
    payload: dict[str, Any],
    *,
    baselines: Sequence[str],
    point_result: str,
) -> dict[str, Any]:
    baselines_payload = payload.get("baselines")
    if not isinstance(baselines_payload, dict):
        baselines_payload = {}

    row_baselines: dict[str, dict[str, float | None]] = {}
    for baseline in baselines:
        value = _runtime_from_entry(baselines_payload.get(baseline))
        row_baselines[baseline] = {"total_runtime_ms": value}

    return {
        "param_value": payload.get("param_value"),
        "baselines": row_baselines,
        "point_result": point_result,
    }


def _run_missing_baselines(
    *,
    point: SweepPoint,
    paths: PointPaths,
    baselines: Sequence[str],
    missing_baselines: Sequence[str],
    existing_payload: dict[str, Any] | None,
    rebuild: bool,
    repo_root: Path,
    scripts_dir: Path,
    micro_batch_size: int,
) -> dict[str, Any]:
    point_seed = _point_seed(point)
    paths.point_dir.mkdir(parents=True, exist_ok=True)

    if rebuild or not paths.fleet_path.exists():
        fleet = _build_device_fleet(num_devices=point.num_devices, seed=point_seed)
        _write_json(paths.fleet_path, fleet)

    run_baseline_cmd = [
        sys.executable,
        str(scripts_dir / "run_baseline_comparison.py"),
        "--model",
        point.model_size,
        "--num-devices",
        str(point.num_devices),
        "--device-config",
        _relative_to_repo(paths.fleet_path, repo_root),
        "--baselines",
        ",".join(missing_baselines),
        "--batch-size",
        str(point.batch_size),
        "--seq-len",
        str(point.seq_len),
        "--micro-batch-size",
        str(micro_batch_size),
        "--num-microbatches",
        str(point.num_microbatches),
        "--output-dir",
        _relative_to_repo(paths.planning_dir, repo_root),
        "--log-level",
        "WARNING",
    ]
    _run_command(run_baseline_cmd, cwd=repo_root)

    baseline_results: dict[str, Any] = {}
    if not rebuild and existing_payload is not None:
        old_baselines = existing_payload.get("baselines")
        if isinstance(old_baselines, dict):
            for key, value in old_baselines.items():
                baseline_results[_canonical_baseline_name(str(key))] = value

    for baseline in missing_baselines:
        manifest_path = paths.planning_dir / "manifests" / f"{baseline}_manifest.json"
        if not manifest_path.exists():
            baseline_results[baseline] = {
                "total_runtime_ms": None,
                "missing_manifest": True,
            }
            continue

        baseline_vtime_dir = paths.vtime_root / baseline
        run_vtime_cmd = [
            "bash",
            str(scripts_dir / "run_vtime_experiment.sh"),
            baseline,
            _relative_to_repo(manifest_path, repo_root),
            _relative_to_repo(paths.fleet_path, repo_root),
            _relative_to_repo(baseline_vtime_dir, repo_root),
        ]
        try:
            _run_command(run_vtime_cmd, cwd=repo_root, tolerate_kill=True)
        except RuntimeError as exc:
            print(f"  [{baseline}] VTIME run failed, skipping: {exc}")
            baseline_results[baseline] = {
                "total_runtime_ms": None,
                "vtime_failed": True,
            }
            continue

        vtime_log_path = baseline_vtime_dir / "vtime.log"
        if not vtime_log_path.exists():
            print(
                f"  [{baseline}] VTIME exited but no vtime.log produced, skipping"
            )
            baseline_results[baseline] = {
                "total_runtime_ms": None,
                "vtime_no_log": True,
            }
            continue

        runtime_json_path = paths.runtimes_root / f"{baseline}_runtime.json"
        optimizer_tail_ms = _load_optimizer_tail_ms(paths.planning_dir, baseline)

        compute_cmd = [
            sys.executable,
            str(scripts_dir / "compute_batch_runtime.py"),
            "--vtime-log",
            _relative_to_repo(vtime_log_path, repo_root),
            "--manifest",
            _relative_to_repo(manifest_path, repo_root),
            "--baseline-type",
            baseline,
            "--device-config",
            _relative_to_repo(paths.fleet_path, repo_root),
            "--num-microbatches",
            str(point.num_microbatches),
            "--output",
            _relative_to_repo(runtime_json_path, repo_root),
        ]
        if optimizer_tail_ms is not None:
            compute_cmd.extend(["--optimizer-tail-ms", str(optimizer_tail_ms)])
        try:
            _run_command(compute_cmd, cwd=repo_root)
        except RuntimeError as exc:
            print(f"  [{baseline}] runtime computation failed, skipping: {exc}")
            baseline_results[baseline] = {
                "total_runtime_ms": None,
                "compute_failed": True,
            }
            continue

        total_runtime_ms = _load_total_runtime_ms(runtime_json_path)
        baseline_results[baseline] = {
            "total_runtime_ms": total_runtime_ms,
            "manifest": _relative_to_repo(manifest_path, repo_root),
            "vtime_log": _relative_to_repo(vtime_log_path, repo_root),
            "runtime_json": _relative_to_repo(runtime_json_path, repo_root),
            "optimizer_tail_ms": optimizer_tail_ms,
        }

    point_payload = {
        "sweep_dim": point.sweep_dim,
        "param_value": point.param_value,
        "config": {
            "num_devices": point.num_devices,
            "model_size": point.model_size,
            "batch_size": point.batch_size,
            "seq_len": point.seq_len,
            "num_microbatches": point.num_microbatches,
            "micro_batch_size": micro_batch_size,
            "baselines": list(baselines),
            "seed": point_seed,
        },
        "planning_dir": _relative_to_repo(paths.planning_dir, repo_root),
        "baselines": baseline_results,
    }
    _write_json(paths.result_path, point_payload)
    return point_payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-oriented VTIME scaling sweeps for num_devices, "
            "model_size, and batch_size."
        )
    )
    parser.add_argument(
        "--sweep-dim",
        action="append",
        default=None,
        help=(
            "Sweep dimensions to run (repeatable or comma-separated): "
            "num_devices, model_size, batch_size. "
            "Defaults to all three."
        ),
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default=",".join(DEFAULT_BASELINES),
        help="Comma-separated baseline list",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vtime_scaling",
        help="Output directory for per-point and aggregate JSON",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length used across paper sweeps",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro-batch size forwarded to planning/runtime scripts",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--num-devices-values",
        type=str,
        default=",".join(str(x) for x in PAPER_NUM_DEVICES),
        help="Paper num_devices sweep values",
    )
    parser.add_argument(
        "--batch-size-values",
        type=str,
        default=",".join(str(x) for x in PAPER_BATCH_SIZES),
        help="Paper batch_size sweep values",
    )
    parser.add_argument(
        "--model-size-values",
        type=str,
        default=",".join(PAPER_MODEL_ORDER),
        help="Paper model_size sweep values",
    )
    parser.add_argument(
        "--model-size-base-devices",
        type=int,
        default=256,
        help="Device count assigned to opt-13b before proportional scaling",
    )
    parser.add_argument(
        "--model-size-max-devices",
        type=int,
        default=1024,
        help="Maximum devices used in model_size sweep",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    output_root = _resolve_output_root(args.output_dir, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baselines = _parse_baselines(args.baselines)
    if not baselines:
        raise ValueError("--baselines must include at least one baseline")
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be > 0")

    dims = _parse_sweep_dims(args.sweep_dim)

    for dim in dims:
        points = _build_points_for_dim(dim, args)
        rows: list[dict[str, Any]] = []
        backfill_tasks: list[BackfillTask] = []
        reused_count = 0
        full_missing_count = 0
        partial_missing_count = 0

        print(f"[{dim}] scanning {len(points)} points for cached results")

        for idx, point in enumerate(points):
            point_seed = _point_seed(point)
            paths = _point_paths(point, output_root)
            existing_payload = _load_existing_payload(paths.result_path)

            if existing_payload is not None and not _point_config_matches(
                existing_payload,
                point,
                baselines,
                args.micro_batch_size,
                point_seed,
            ):
                existing_payload = None

            if existing_payload is None:
                backfill_tasks.append(
                    BackfillTask(
                        row_index=idx,
                        point=point,
                        paths=paths,
                        missing_baselines=tuple(baselines),
                        existing_payload=None,
                        rebuild=True,
                    )
                )
                full_missing_count += 1
                rows.append(
                    {
                        "param_value": point.param_value,
                        "baselines": {
                            baseline: {"total_runtime_ms": None}
                            for baseline in baselines
                        },
                        "point_result": _relative_to_repo(
                            paths.result_path, repo_root
                        ),
                    }
                )
                continue

            row = _row_from_payload(
                existing_payload,
                baselines=baselines,
                point_result=_relative_to_repo(paths.result_path, repo_root),
            )
            rows.append(row)

            missing: list[str] = []
            payload_baselines = existing_payload.get("baselines")
            if not isinstance(payload_baselines, dict):
                payload_baselines = {}
            for baseline in baselines:
                runtime = _runtime_from_entry(payload_baselines.get(baseline))
                if runtime is None:
                    missing.append(baseline)

            if not missing:
                reused_count += 1
                continue

            backfill_tasks.append(
                BackfillTask(
                    row_index=idx,
                    point=point,
                    paths=paths,
                    missing_baselines=tuple(missing),
                    existing_payload=existing_payload,
                    rebuild=False,
                )
            )
            partial_missing_count += 1

        print(
            f"[{dim}] reuse={reused_count} "
            f"backfill_partial={partial_missing_count} "
            f"backfill_full={full_missing_count}"
        )

        if backfill_tasks:
            print(
                f"[{dim}] running {len(backfill_tasks)} missing points as final step"
            )
            for task in backfill_tasks:
                print(
                    f"[{dim}] backfill point {task.row_index:03d} "
                    f"param={task.point.param_value} "
                    f"missing={','.join(task.missing_baselines)}"
                )
                payload = _run_missing_baselines(
                    point=task.point,
                    paths=task.paths,
                    baselines=baselines,
                    missing_baselines=task.missing_baselines,
                    existing_payload=task.existing_payload,
                    rebuild=task.rebuild,
                    repo_root=repo_root,
                    scripts_dir=scripts_dir,
                    micro_batch_size=args.micro_batch_size,
                )
                rows[task.row_index] = _row_from_payload(
                    payload,
                    baselines=baselines,
                    point_result=_relative_to_repo(task.paths.result_path, repo_root),
                )
        else:
            print(f"[{dim}] no missing VTIME data; skip execution")

        normalized_rows = _normalize_rows(rows, baselines)
        _write_json(output_root / f"{dim}.json", rows)
        _write_json(output_root / f"{dim}_normalized.json", normalized_rows)

        print(f"[{dim}] wrote {output_root / f'{dim}.json'}")
        print(f"[{dim}] wrote {output_root / f'{dim}_normalized.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
