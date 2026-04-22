#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportDeprecated=false, reportImplicitStringConcatenation=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import multiprocessing as mp
import random
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    embedding_dim: int
    num_heads: int
    d_ff: int
    vocab_size: int


MODEL_CONFIGS = {
    "opt-125m": ModelConfig(
        num_layers=12,
        embedding_dim=768,
        num_heads=12,
        d_ff=3072,
        vocab_size=50272,
    ),
    "opt-1.3b": ModelConfig(
        num_layers=24,
        embedding_dim=2048,
        num_heads=32,
        d_ff=8192,
        vocab_size=50272,
    ),
    "opt-13b": ModelConfig(
        num_layers=40,
        embedding_dim=5120,
        num_heads=40,
        d_ff=20480,
        vocab_size=50272,
    ),
    "opt-30b": ModelConfig(
        num_layers=48,
        embedding_dim=7168,
        num_heads=56,
        d_ff=28672,
        vocab_size=50272,
    ),
    "opt-66b": ModelConfig(
        num_layers=64,
        embedding_dim=9216,
        num_heads=72,
        d_ff=36864,
        vocab_size=50272,
    ),
    "llama2-7b": ModelConfig(
        num_layers=32,
        embedding_dim=4096,
        num_heads=32,
        d_ff=11008,
        vocab_size=32000,
    ),
    "llama2-13b": ModelConfig(
        num_layers=40,
        embedding_dim=5120,
        num_heads=40,
        d_ff=13824,
        vocab_size=32000,
    ),
    "llama2-70b": ModelConfig(
        num_layers=80,
        embedding_dim=8192,
        num_heads=64,
        d_ff=28672,
        vocab_size=32000,
    ),
}


SWEEP_VALUES: dict[str, list[Any]] = {
    "num_devices": [64, 128, 256, 512, 1024],
    "model_size": [
        "opt-125m",
        "opt-1.3b",
        "opt-13b",
        "opt-30b",
        "opt-66b",
        "llama2-7b",
        "llama2-13b",
        "llama2-70b",
    ],
    "batch_size": [16, 32, 64, 128, 256],
    "straggler_ratio": [0.0, 0.05, 0.10, 0.20],
    "dl_ul_ratio": [2, 5, 10, 20],
}


MODEL_NAME_MAP = {
    "opt-125m": "facebook/opt-125m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-13b": "facebook/opt-13b",
    "opt-30b": "facebook/opt-30b",
    "opt-66b": "facebook/opt-66b",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-hf",
}

DEFAULT_BASELINES = ["cleave", "dtfm", "asteroid", "confident", "alpa"]


@dataclass(frozen=True)
class SweepTask:
    sweep_dim: str
    param_value: Any
    point_index: int
    num_devices: int
    model_size: str
    batch_size: int
    straggler_ratio: float
    dl_ul_ratio: float | None
    seq_len: int
    baselines: tuple[str, ...]
    output_dir: str
    seed: int


def _parse_unit(value: str) -> float:
    value = value.strip().upper()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTP])", value)
    if not match:
        raise ValueError(f"Unsupported unit value: {value}")
    base = float(match.group(1))
    suffix = match.group(2)
    scale = {
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
    }[suffix]
    return base * scale


def _point_seed(base_seed: int, sweep_dim: str, param_value: Any) -> int:
    encoded = f"{base_seed}:{sweep_dim}:{param_value}".encode("utf-8")
    acc = 0
    for i, byte in enumerate(encoded, 1):
        acc = (acc + i * byte) & 0xFFFFFFFF
    return acc


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


def _canonical_baseline_name(name: str) -> str:
    return name.strip().lower()


def _parse_baselines(raw: str) -> list[str]:
    values = [_canonical_baseline_name(x) for x in raw.split(",") if x.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for name in values:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _parse_sweep_dims(raw_dims: list[str] | None, use_all: bool) -> list[str]:
    if use_all:
        return list(SWEEP_VALUES.keys())

    dims: list[str] = []
    if raw_dims:
        for chunk in raw_dims:
            dims.extend([x.strip() for x in chunk.split(",") if x.strip()])

    deduped: list[str] = []
    seen: set[str] = set()
    for dim in dims:
        if dim not in SWEEP_VALUES:
            allowed = ", ".join(sorted(SWEEP_VALUES.keys()))
            raise ValueError(
                f"Unknown sweep dimension '{dim}'. Allowed: {allowed}"
            )
        if dim in seen:
            continue
        seen.add(dim)
        deduped.append(dim)

    if not deduped:
        raise ValueError("Provide --sweep-dim or --all (no default full sweep)")
    return deduped


def _build_device_fleet(
    *,
    num_devices: int,
    straggler_ratio: float,
    dl_ul_ratio: float | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    flops_lb = _parse_unit("5T")
    flops_ub = _parse_unit("7T")
    dl_lb = _parse_unit("10M")
    dl_ub = _parse_unit("100M")
    ul_lb = _parse_unit("5M")
    ul_ub = _parse_unit("10M")
    mem_lb = _parse_unit("1G")
    mem_ub = _parse_unit("3G")

    fixed_ul = _parse_unit("7.5M")
    devices: list[dict[str, Any]] = []
    for rank in range(num_devices):
        if dl_ul_ratio is None:
            ul_bw = rng.uniform(ul_lb, ul_ub)
            dl_bw = rng.uniform(dl_lb, dl_ub)
        else:
            ul_bw = fixed_ul
            dl_bw = fixed_ul * float(dl_ul_ratio)
        devices.append(
            {
                "rank": rank,
                "flops": int(rng.uniform(flops_lb, flops_ub)),
                "memory": int(rng.uniform(mem_lb, mem_ub)),
                "ul_bw": float(ul_bw),
                "dl_bw": float(dl_bw),
                "ul_lat": 0.0,
                "dl_lat": 0.0,
            }
        )

    degrade_count = int(max(0.0, straggler_ratio) * num_devices)
    if degrade_count > 0:
        degrade_indices = rng.sample(range(num_devices), k=degrade_count)
        for idx in degrade_indices:
            devices[idx]["flops"] = int(devices[idx]["flops"] / 10.0)
            devices[idx]["ul_bw"] = float(devices[idx]["ul_bw"] / 10.0)
            devices[idx]["dl_bw"] = float(devices[idx]["dl_bw"] / 10.0)

    return devices


def _extract_runtime_ms(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        return None
    for key in (
        "total_runtime_ms",
        "runtime_ms",
        "batch_time_ms",
        "total_ms",
    ):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def _extract_baseline_runtimes(
    payload: Any,
    baselines: Sequence[str],
) -> dict[str, float]:
    baseline_set = {_canonical_baseline_name(x) for x in baselines}
    out: dict[str, float] = {}

    if isinstance(payload, dict):
        baselines_payload = payload.get("baselines")
        if isinstance(baselines_payload, dict):
            for key, value in baselines_payload.items():
                canon = _canonical_baseline_name(str(key))
                if canon not in baseline_set:
                    continue
                runtime = _extract_runtime_ms(value)
                if runtime is not None:
                    out[canon] = runtime

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if _canonical_baseline_name(str(key)) == "normalized":
                    continue
                canon = _canonical_baseline_name(str(key))
                if canon in baseline_set and canon not in out:
                    runtime = _extract_runtime_ms(value)
                    if runtime is not None:
                        out[canon] = runtime
                    continue
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return out


def _load_summary_runtimes(
    comparison_output_dir: Path,
    point_dir: Path,
    baselines: Sequence[str],
) -> dict[str, float]:
    candidates = [
        comparison_output_dir / "summary.json",
        point_dir / "summary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            runtimes = _extract_baseline_runtimes(payload, baselines)
            if runtimes:
                return runtimes
        except Exception:
            continue
    return {}


def _import_harness() -> tuple[Callable[..., Any] | None, str]:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        module = importlib.import_module("run_baseline_comparison")
    except Exception as exc:
        return None, f"import failed: {exc}"

    for name in (
        "run_comparison",
        "run_baseline_comparison",
        "compare_baselines",
    ):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn, name

    return None, "imported module without callable comparison entrypoint"


def _run_via_import(
    task: SweepTask,
    device_config_path: Path,
    comparison_output_dir: Path,
) -> tuple[dict[str, float], str]:
    run_fn, entry_name = _import_harness()
    if run_fn is None:
        raise ImportError(entry_name)

    model_name = MODEL_NAME_MAP.get(task.model_size, task.model_size)
    kwargs_source: dict[str, Any] = {
        "model": model_name,
        "model_name": model_name,
        "model_size": task.model_size,
        "model_config": asdict(MODEL_CONFIGS[task.model_size]),
        "num_devices": task.num_devices,
        "device_config": str(device_config_path),
        "device_config_path": str(device_config_path),
        "baselines": list(task.baselines),
        "batch_size": task.batch_size,
        "seq_len": task.seq_len,
        "seq_length": task.seq_len,
        "output_dir": str(comparison_output_dir),
        "seed": task.seed,
    }

    signature = inspect.signature(run_fn)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )

    call_kwargs: dict[str, Any] = {}
    for key, value in kwargs_source.items():
        if accepts_kwargs or key in signature.parameters:
            call_kwargs[key] = value

    try:
        payload = run_fn(**call_kwargs)
    except TypeError:
        call_kwargs["baselines"] = ",".join(task.baselines)
        payload = run_fn(**call_kwargs)

    runtimes = _extract_baseline_runtimes(payload, task.baselines)
    if not runtimes:
        runtimes = _load_summary_runtimes(
            comparison_output_dir=comparison_output_dir,
            point_dir=comparison_output_dir.parent,
            baselines=task.baselines,
        )
    if not runtimes:
        raise RuntimeError(
            "Imported harness returned no parsable baseline runtimes"
        )

    return runtimes, f"import:{entry_name}"


def _run_via_subprocess(
    task: SweepTask,
    device_config_path: Path,
    comparison_output_dir: Path,
) -> tuple[dict[str, float], str]:
    scripts_dir = Path(__file__).resolve().parent
    comparison_script = scripts_dir / "run_baseline_comparison.py"
    if not comparison_script.exists():
        raise FileNotFoundError(
            f"run_baseline_comparison.py not found at {comparison_script}"
        )

    model_name = MODEL_NAME_MAP.get(task.model_size, task.model_size)
    cmd = [
        sys.executable,
        str(comparison_script),
        "--model",
        model_name,
        "--num-devices",
        str(task.num_devices),
        "--device-config",
        str(device_config_path),
        "--baselines",
        ",".join(task.baselines),
        "--batch-size",
        str(task.batch_size),
        "--seq-len",
        str(task.seq_len),
        "--output-dir",
        str(comparison_output_dir),
    ]

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "<empty stderr>"
        stdout = proc.stdout.strip() or "<empty stdout>"
        raise RuntimeError(
            "subprocess run_baseline_comparison failed\n"
            f"stdout:\n{stdout}\n\n"
            f"stderr:\n{stderr}"
        )

    runtimes = _load_summary_runtimes(
        comparison_output_dir=comparison_output_dir,
        point_dir=comparison_output_dir.parent,
        baselines=task.baselines,
    )
    if not runtimes:
        try:
            payload = json.loads(proc.stdout)
            runtimes = _extract_baseline_runtimes(payload, task.baselines)
        except Exception:
            pass

    if not runtimes:
        raise RuntimeError(
            "subprocess succeeded but no baseline runtimes were found in summary "
            "or stdout"
        )

    return runtimes, "subprocess"


def _run_single_point(task: SweepTask) -> dict[str, Any]:
    point_seed = _point_seed(task.seed, task.sweep_dim, task.param_value)

    point_tag = f"{task.point_index:03d}_{_slugify(task.param_value)}"
    point_dir = Path(task.output_dir) / task.sweep_dim / "points" / point_tag
    comparison_output_dir = point_dir / "comparison"
    point_dir.mkdir(parents=True, exist_ok=True)

    device_fleet = _build_device_fleet(
        num_devices=task.num_devices,
        straggler_ratio=task.straggler_ratio,
        dl_ul_ratio=task.dl_ul_ratio,
        seed=point_seed,
    )
    device_config_path = point_dir / "device_config.json"
    _write_json(device_config_path, device_fleet)

    try:
        runtimes, method = _run_via_import(
            task=task,
            device_config_path=device_config_path,
            comparison_output_dir=comparison_output_dir,
        )
    except Exception as import_exc:
        try:
            runtimes, method = _run_via_subprocess(
                task=task,
                device_config_path=device_config_path,
                comparison_output_dir=comparison_output_dir,
            )
        except Exception as subprocess_exc:
            raise RuntimeError(
                f"Sweep point failed for {task.sweep_dim}={task.param_value}; "
                f"import error: {import_exc}; "
                f"subprocess error: {subprocess_exc}"
            ) from subprocess_exc

    baseline_payload = {
        baseline: (float(runtimes[baseline]) if baseline in runtimes else None)
        for baseline in task.baselines
    }

    point_payload = {
        "sweep_dim": task.sweep_dim,
        "param_value": task.param_value,
        "seed": point_seed,
        "runner": method,
        "config": {
            "num_devices": task.num_devices,
            "model_size": task.model_size,
            "model_name": MODEL_NAME_MAP.get(task.model_size, task.model_size),
            "model_config": asdict(MODEL_CONFIGS[task.model_size]),
            "batch_size": task.batch_size,
            "seq_len": task.seq_len,
            "straggler_ratio": task.straggler_ratio,
            "dl_ul_ratio": task.dl_ul_ratio,
            "baselines": list(task.baselines),
        },
        "baselines": baseline_payload,
    }
    _write_json(point_dir / "result.json", point_payload)

    return {
        "param_value": task.param_value,
        "baselines": baseline_payload,
        "point_result": str(point_dir / "result.json"),
    }


def _normalize_to_cleave(
    rows: list[dict[str, Any]],
    baselines: Sequence[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("baselines", {})
        cleave = raw.get("cleave")
        baseline_norm: dict[str, float | None] = {}
        for baseline in baselines:
            value = raw.get(baseline)
            if cleave is None or cleave == 0 or value is None:
                baseline_norm[baseline] = None
            else:
                baseline_norm[baseline] = float(value) / float(cleave)
        normalized.append(
            {
                "param_value": row.get("param_value"),
                "baselines": baseline_norm,
            }
        )
    return normalized


def _build_tasks_for_dim(
    sweep_dim: str,
    args: argparse.Namespace,
    baselines: Sequence[str],
) -> list[SweepTask]:
    tasks: list[SweepTask] = []
    for idx, value in enumerate(SWEEP_VALUES[sweep_dim]):
        num_devices = (
            int(value)
            if sweep_dim == "num_devices"
            else args.default_num_devices
        )
        model_size = (
            str(value) if sweep_dim == "model_size" else args.default_model_size
        )
        batch_size = (
            int(value) if sweep_dim == "batch_size" else args.default_batch_size
        )
        straggler_ratio = (
            float(value)
            if sweep_dim == "straggler_ratio"
            else args.default_straggler_ratio
        )
        dl_ul_ratio = (
            float(value)
            if sweep_dim == "dl_ul_ratio"
            else args.default_dl_ul_ratio
        )

        task = SweepTask(
            sweep_dim=sweep_dim,
            param_value=value,
            point_index=idx,
            num_devices=num_devices,
            model_size=model_size,
            batch_size=batch_size,
            straggler_ratio=straggler_ratio,
            dl_ul_ratio=dl_ul_ratio,
            seq_len=args.seq_len,
            baselines=tuple(baselines),
            output_dir=args.output_dir,
            seed=args.seed,
        )
        tasks.append(task)
    return tasks


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline comparison parameter sweeps across num_devices, "
            "model_size, batch_size, straggler_ratio, and DL/UL asymmetry"
        )
    )
    parser.add_argument(
        "--sweep-dim",
        action="append",
        default=None,
        help=(
            "Sweep dimension to run (repeatable or comma-separated): "
            "num_devices, model_size, batch_size, straggler_ratio, dl_ul_ratio"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all sweep dimensions",
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
        default="results/sweeps",
        help="Output directory for sweep JSON files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of sweep points to run concurrently",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length forwarded to comparison harness",
    )
    parser.add_argument(
        "--default-num-devices",
        type=int,
        default=256,
        help="Default num_devices used when not sweeping num_devices",
    )
    parser.add_argument(
        "--default-model-size",
        type=str,
        default="opt-13b",
        choices=sorted(MODEL_CONFIGS.keys()),
        help="Default model_size used when not sweeping model_size",
    )
    parser.add_argument(
        "--default-batch-size",
        type=int,
        default=128,
        help="Default batch_size used when not sweeping batch_size",
    )
    parser.add_argument(
        "--default-straggler-ratio",
        type=float,
        default=0.0,
        help="Default straggler_ratio used when not sweeping straggler_ratio",
    )
    parser.add_argument(
        "--default-dl-ul-ratio",
        type=float,
        default=None,
        help="Default DL/UL ratio used when not sweeping dl_ul_ratio",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    baselines = _parse_baselines(args.baselines)
    if not baselines:
        raise ValueError("--baselines must include at least one baseline")

    dims = _parse_sweep_dims(args.sweep_dim, args.all)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    workers = max(int(args.parallel), 1)

    for sweep_dim in dims:
        tasks = _build_tasks_for_dim(sweep_dim, args, baselines)

        if workers == 1:
            rows = [_run_single_point(task) for task in tasks]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                rows = pool.map(_run_single_point, tasks)

        sweep_rows = [
            {"param_value": row["param_value"], "baselines": row["baselines"]}
            for row in rows
        ]
        normalized_rows = _normalize_to_cleave(sweep_rows, baselines)

        _write_json(output_root / f"{sweep_dim}.json", sweep_rows)
        _write_json(
            output_root / f"{sweep_dim}_normalized.json",
            normalized_rows,
        )

        print(f"[{sweep_dim}] wrote {output_root / f'{sweep_dim}.json'}")
        print(
            f"[{sweep_dim}] wrote "
            f"{output_root / f'{sweep_dim}_normalized.json'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
