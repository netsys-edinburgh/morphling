#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false, reportMissingTypeStubs=false
# pyright: reportUnusedCallResult=false, reportImplicitStringConcatenation=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false
"""E2E single-GPU OPT-125M training benchmark.

Runs baseline and green-context modes sequentially, records per-step
metrics, writes CSV artifacts, and generates publication-style
comparison plots.
"""

from __future__ import annotations

import argparse
import csv
from importlib import import_module
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
import types
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WARMUP_STEPS = 5
WONG_PALETTE = [
    "#000000",
    "#EEBA0C",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#0000FF",
    "#FF0000",
]


class _LocalMatmulBackend:
    def __init__(self, torch_mod: Any) -> None:
        self._torch = torch_mod
        self._queue: list[Any] = []

    def async_dispatch_matmul(self, mat_a: Any, mat_b: Any) -> None:
        self._queue.append(self._torch.matmul(mat_a, mat_b.transpose(-2, -1)))

    def wait_matmul(self, _idx: int) -> Any:
        if not self._queue:
            raise RuntimeError("wait_matmul called with empty queue")
        return self._queue.pop(0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU OPT training benchmark with green context",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        default="data/ldpc_trace_with_ctrl.csv",
        help="LDPC CSV trace path (raw format)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of optimizer steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/",
        help="Output directory for CSV/plots",
    )
    parser.add_argument(
        "--total-sms",
        type=int,
        default=48,
        help="Total SM count for LDPC adapter conversion",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline training run",
    )
    parser.add_argument(
        "--skip-greenctx",
        action="store_true",
        help="Skip green-context training run",
    )
    parser.add_argument(
        "--dump-gemm-shapes",
        action="store_true",
        help="Dump unique GEMM shapes observed by autograd hooks",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of training runs",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed (incremented per run)",
    )
    parser.add_argument(
        "--max-trace-slots",
        type=int,
        default=27462,
        help=(
            "Truncate training to this many trace slots (shorter trace length)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (legacy alias for --seed-base)",
    )
    parser.add_argument(
        "--skip-violation-analysis",
        action="store_true",
        help="Skip generating per-run violation artifacts",
    )
    return parser.parse_args()


def _bootstrap_morphling_runtime() -> None:
    if "morphling" not in sys.modules:
        morphling_mod = types.ModuleType("morphling")
        morphling_mod.__path__ = [str(REPO_ROOT / "morphling")]
        morphling_mod.__package__ = "morphling"
        sys.modules["morphling"] = morphling_mod

    if "morphling.runtime" not in sys.modules:
        runtime_mod = types.ModuleType("morphling.runtime")
        runtime_mod.__path__ = [str(REPO_ROOT / "morphling" / "runtime")]
        runtime_mod.__package__ = "morphling.runtime"
        sys.modules["morphling.runtime"] = runtime_mod


def _load_greenctx_symbols() -> tuple[Any, Any, Any]:
    _bootstrap_morphling_runtime()
    green_context = import_module("morphling.runtime.green_context")
    ldpc_adapter = import_module("morphling.runtime.ldpc_trace_adapter")
    return (
        getattr(green_context, "GreenContextConfig"),
        getattr(green_context, "GreenContextController"),
        getattr(ldpc_adapter, "LdpcTraceAdapter"),
    )


def _resolve_trace_path(trace_path: str) -> Path:
    path = Path(trace_path)
    if path.exists():
        return path
    if not path.is_absolute():
        repo_path = REPO_ROOT / path
        if repo_path.exists():
            return repo_path
    raise FileNotFoundError(f"Trace file not found: {trace_path}")


def _paper_rc() -> dict[str, object]:
    return {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.3,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _save_dual_output(fig: Any, plt: Any, stem_path: Path) -> None:
    fig.savefig(stem_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _set_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _make_loader(
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    *,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = num_steps * batch_size
    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(samples, seq_len),
        dtype=torch.long,
        generator=generator,
    )
    dataset = TensorDataset(token_ids)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def _prepare_greenctx_controller(
    *,
    trace_path: Path,
    total_sms: int,
    device_id: int,
    GreenContextConfig: Any,
    GreenContextController: Any,
    LdpcTraceAdapter: Any,
) -> tuple[Any, Path]:
    adapter = LdpcTraceAdapter(trace_path, total_sms=total_sms)
    with tempfile.NamedTemporaryFile(
        prefix="ldpc_v2_",
        suffix=".csv",
        delete=False,
    ) as tmp_file:
        v2_path = Path(tmp_file.name)
    adapter.to_v2_file(v2_path)

    backend_name = "auto"
    greenctx_so = REPO_ROOT / "morphling" / "_GreenCtx.so"
    if not greenctx_so.exists():
        logging.warning(
            "_GreenCtx.so unavailable at %s; falling back to backend='off'.",
            greenctx_so,
        )
        backend_name = "off"

    cfg = GreenContextConfig(
        enabled=True,
        backend=backend_name,
        trace_path=str(v2_path),
        clock_mode="step",
    )

    try:
        ctrl = GreenContextController.from_config(device_id=device_id, cfg=cfg)
    except Exception as exc:
        logging.warning(
            "Green context init failed with backend='%s': %s. "
            "Retrying with backend='off'.",
            backend_name,
            exc,
        )
        cfg = GreenContextConfig(
            enabled=True,
            backend="off",
            trace_path=str(v2_path),
            clock_mode="step",
        )
        ctrl = GreenContextController.from_config(device_id=device_id, cfg=cfg)

    logging.info(
        "Green context backend active: %s (supported=%s)",
        ctrl.backend.__class__.__name__,
        ctrl.is_supported,
    )
    return ctrl, v2_path


def _run_mode(
    *,
    mode_name: str,
    model_name: str,
    seed: int,
    loader: Any,
    device: Any,
    total_sms: int,
    torch: Any,
    AutoModelForCausalLM: Any,
    controller: Any | None,
    get_gemm_log: Any,
    collect_gemm_entries: bool = False,
) -> tuple[list[dict[str, float | int]], list[dict[str, Any]]]:
    _set_seed(torch, seed)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    metrics: list[dict[str, float | int]] = []
    all_gemm_entries: list[dict[str, Any]] = []

    _ = get_gemm_log(reset=True)
    if controller is not None:
        controller.reset_swap_stats()

    logging.info("Starting mode: %s", mode_name)
    for step_idx, (input_ids_cpu,) in enumerate(loader):
        input_ids = input_ids_cpu.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        step_start_ns = time.perf_counter_ns()

        if controller is None:
            start_t = time.perf_counter()
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            sm_count = total_sms
        else:
            with controller.step_scope(step_idx) as bundle:
                start_t = time.perf_counter()
                with torch.cuda.stream(bundle.comp):
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                bundle.comp.synchronize()
                sm_count = int(bundle.sm_count)

        if controller is None:
            step_swap_count = 0
            step_swap_overhead_us = 0.0
        else:
            swap_stats = controller.get_swap_stats()
            python_swap = swap_stats.get("python_layer", {})
            step_swap_count = int(python_swap.get("count", 0))
            step_swap_overhead_us = float(
                python_swap.get("total_overhead_us", 0.0)
            )
            controller.reset_swap_stats()

        step_gemm_entries = get_gemm_log(reset=True)
        step_gemm_count = int(len(step_gemm_entries))
        if collect_gemm_entries and step_gemm_entries:
            all_gemm_entries.extend(step_gemm_entries)

        wall_time_s = max(time.perf_counter() - start_t, 1e-12)
        step_end_ns = time.perf_counter_ns()
        tokens_per_sec = float(input_ids.numel()) / wall_time_s
        metrics.append(
            {
                "step_idx": int(step_idx),
                "step_start_ns": int(step_start_ns),
                "step_end_ns": int(step_end_ns),
                "wall_time_ms": wall_time_s * 1000.0,
                "loss": float(loss.detach().item()),
                "tokens_per_sec": tokens_per_sec,
                "gpu_mem_mb": float(torch.cuda.memory_allocated(device) / 1e6),
                "sm_count": int(sm_count),
                "swap_count": int(step_swap_count),
                "swap_overhead_us": float(step_swap_overhead_us),
                "gemm_count": int(step_gemm_count),
                "is_warmup": int(step_idx < WARMUP_STEPS),
            }
        )

    del model
    torch.cuda.empty_cache()
    return metrics, all_gemm_entries


def _write_violation_artifacts(
    *,
    run_dir: Path,
    metrics_df: Any,
    gemm_log: list[dict[str, Any]],
    max_sm_count: int,
) -> None:
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "analyze_violations",
        Path(__file__).resolve().parent / "analyze_violations.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    sys.modules.setdefault("analyze_violations", _mod)
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    av = _mod

    required_cols = {"step_idx", "step_start_ns", "step_end_ns", "sm_count"}
    if not required_cols.issubset(set(metrics_df.columns)):
        raise ValueError(
            "Missing required columns for violation artifacts: "
            + ", ".join(sorted(required_cols - set(metrics_df.columns)))
        )

    step_path = run_dir / "step_boundaries.csv"
    step_df = metrics_df[
        ["step_idx", "step_start_ns", "step_end_ns", "sm_count"]
    ].copy()
    step_df.columns = ["step", "start_ns", "end_ns", "sm_count"]
    step_df.to_csv(step_path, index=False)

    gemm_path = run_dir / "gemm_log.csv"
    epoch_ns = int(step_df["start_ns"].iloc[0]) if len(step_df) > 0 else 0
    with open(gemm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "function_name",
                "start_ns",
                "end_ns",
                "duration_ns",
                "m",
                "n",
                "k",
            ],
        )
        writer.writeheader()
        for entry in gemm_log:
            start_ns = epoch_ns + int(
                float(entry.get("start_us", 0.0)) * 1000.0
            )
            end_ns = epoch_ns + int(float(entry.get("end_us", 0.0)) * 1000.0)
            if end_ns < start_ns:
                end_ns = start_ns
            writer.writerow(
                {
                    "function_name": str(entry.get("phase", "gemm")),
                    "start_ns": int(start_ns),
                    "end_ns": int(end_ns),
                    "duration_ns": int(end_ns - start_ns),
                    "m": int(entry.get("m", 0)),
                    "n": int(entry.get("n", 0)),
                    "k": int(entry.get("k", 0)),
                }
            )

    gemms = av.parse_gemm_log(gemm_path)
    slots = av.parse_step_boundaries(step_path)
    step_gemms = av.assign_gemms_to_steps(gemms, slots)
    summaries = av.detect_violations(slots, step_gemms, max_sm_count)
    violation_time_ns = av.compute_violation_time_ns(summaries, slots)

    report_path = run_dir / "violation_report.txt"
    with open(report_path, "w", encoding="utf-8") as out:
        av.write_report(
            out=out,
            gemms=gemms,
            slots=slots,
            summaries=summaries,
            violation_time_ns=violation_time_ns,
            max_sm=max_sm_count,
            top_n=10,
            verbose=False,
        )

    av.write_csv_summary(run_dir / "violation_summary.csv", summaries)
    av.write_json_summary(
        run_dir / "violations.json",
        gemms,
        slots,
        summaries,
        violation_time_ns,
    )


def _timed_rows(df: Any) -> Any:
    timed = df.loc[df["step_idx"] >= WARMUP_STEPS]
    if len(timed) == 0:
        return df
    return timed


def _summarize_run(df: Any) -> dict[str, float]:
    timed = _timed_rows(df)
    return {
        "tokens_per_sec": float(timed["tokens_per_sec"].mean()),
        "step_time_ms": float(timed["wall_time_ms"].mean()),
        "swap_count": float(timed["swap_count"].mean()),
        "swap_overhead_us": float(timed["swap_overhead_us"].mean()),
    }


def _dump_gemm_shapes(
    output_dir: str,
    num_steps: int,
    gemm_log: list[dict[str, Any]] | None = None,
) -> None:
    from collections import Counter
    import os

    if gemm_log is None:
        from morphling.hooks.autograd import get_gemm_log

        log = get_gemm_log()
    else:
        log = gemm_log
    shape_counts = Counter()
    for entry in log:
        key = (entry["m"], entry["n"], entry["k"], entry["phase"])
        shape_counts[key] += 1

    shapes = []
    for (m, n, k, phase), count in shape_counts.most_common():
        shapes.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "phase": phase,
                "count_per_step": count / max(num_steps, 1),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "gemm_shapes.json")
    with open(path, "w") as f:
        json.dump(shapes, f, indent=2)
    print(f"Saved {len(shapes)} GEMM shapes to {path}")


def _print_summary(results: dict[str, Any]) -> None:
    print("\nBenchmark Summary (warmup skipped: first 5 steps)")
    print("=" * 78)
    print(
        f"{'mode':<12}"
        f"{'mean_iter_ms':>22}"
        f"{'mean_throughput':>22}"
        f"{'final_loss':>22}"
    )
    print("-" * 78)

    for mode in ("baseline", "greenctx"):
        if mode not in results:
            continue
        df = results[mode]
        timed = _timed_rows(df)
        mean_iter_ms = float(timed["wall_time_ms"].mean())
        mean_tput = float(timed["tokens_per_sec"].mean())
        final_loss = float(df["loss"].iloc[-1])
        print(
            f"{mode:<12}"
            f"{mean_iter_ms:>22.3f}"
            f"{mean_tput:>22.2f}"
            f"{final_loss:>22.6f}"
        )
    print("=" * 78)


def _plot_comparison(
    *,
    results: dict[str, Any],
    output_dir: Path,
    plt: Any,
    np: Any,
) -> None:
    color_map = {
        "baseline": WONG_PALETTE[5],
        "greenctx": WONG_PALETTE[1],
    }
    marker_map = {"baseline": "o", "greenctx": "s"}

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.33, 2.5))
        for mode in ("baseline", "greenctx"):
            if mode not in results:
                continue
            df = _timed_rows(results[mode])
            ax.plot(
                df["step_idx"],
                df["tokens_per_sec"],
                color=color_map[mode],
                marker=marker_map[mode],
                markersize=3,
                linewidth=1.2,
                label=mode,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Training Throughput")
        ax.legend()
        _save_dual_output(fig, plt, output_dir / "eval_throughput")

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.33, 2.5))
        for mode in ("baseline", "greenctx"):
            if mode not in results:
                continue
            df = results[mode]
            ax.plot(
                df["step_idx"],
                df["loss"],
                color=color_map[mode],
                marker=marker_map[mode],
                markersize=3,
                linewidth=1.2,
                label=mode,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        _save_dual_output(fig, plt, output_dir / "eval_loss")

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.33, 2.5))
        for mode in ("baseline", "greenctx"):
            if mode not in results:
                continue
            df = _timed_rows(results[mode])
            vals = df["wall_time_ms"].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            bins = max(5, min(20, int(np.sqrt(len(vals)) * 2)))
            ax.hist(
                vals,
                bins=bins,
                alpha=0.60,
                color=color_map[mode],
                edgecolor="black",
                linewidth=0.4,
                label=mode,
            )
        ax.set_xlabel("Step wall time (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Iteration Time Distribution")
        ax.legend()
        _save_dual_output(fig, plt, output_dir / "eval_iter_time")


def _run_benchmark(args: argparse.Namespace) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import morphling
        from morphling.hooks import apply_hooks, set_greenctx
        from morphling.hooks.autograd import get_gemm_log
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for benchmark runtime: "
            f"{exc}. Install torch/transformers/matplotlib/numpy/pandas."
        ) from exc

    GreenContextConfig, GreenContextController, LdpcTraceAdapter = (
        _load_greenctx_symbols()
    )

    if args.skip_baseline and args.skip_greenctx:
        raise ValueError("Cannot skip both modes")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be > 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.max_trace_slots <= 0:
        raise ValueError("--max-trace-slots must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    local_backend = _LocalMatmulBackend(torch)
    orig_linear_forward = torch.nn.Linear.forward
    orig_functional_linear = torch.nn.functional.linear
    orig_tensor_matmul = torch.Tensor.__matmul__
    orig_bmm = torch.bmm
    hooks_installed = False

    def install_linear_hooks() -> None:
        nonlocal hooks_installed
        if hooks_installed:
            return
        morphling.set_backend(local_backend)
        apply_hooks("linear")
        hooks_installed = True

    base_seed = int(args.seed_base)
    if args.seed_base == 42 and args.seed != 42:
        base_seed = int(args.seed)

    effective_num_steps = min(int(args.num_steps), int(args.max_trace_slots))
    if effective_num_steps < int(args.num_steps):
        logging.info(
            "Truncating training from %d to %d steps due to --max-trace-slots",
            args.num_steps,
            effective_num_steps,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = int(tokenizer.vocab_size)

    run_summaries: dict[str, list[dict[str, float]]] = {}
    single_run_results: dict[str, Any] = {}
    single_run_gemm_log: list[dict[str, Any]] = []

    for run_idx in range(args.runs):
        seed = base_seed + run_idx
        logging.info(
            "Starting run %d/%d (seed=%d)",
            run_idx + 1,
            args.runs,
            seed,
        )

        loader = _make_loader(
            torch,
            DataLoader,
            TensorDataset,
            num_steps=effective_num_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            seed=seed,
        )

        if len(loader) != effective_num_steps:
            raise RuntimeError(
                "Unexpected synthetic loader length: "
                f"got {len(loader)}, expected {effective_num_steps}"
            )

        run_dir = output_dir / f"run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}
        run_gemm_log: list[dict[str, Any]] = []

        if not args.skip_baseline:
            collect_baseline_gemm = bool(
                args.dump_gemm_shapes
                or (not args.skip_violation_analysis and args.skip_greenctx)
            )
            if collect_baseline_gemm:
                install_linear_hooks()
                set_greenctx(None, reset_log=True)
            baseline_metrics, baseline_gemm_log = _run_mode(
                mode_name="baseline",
                model_name=args.model,
                seed=seed,
                loader=loader,
                device=device,
                total_sms=args.total_sms,
                torch=torch,
                AutoModelForCausalLM=AutoModelForCausalLM,
                controller=None,
                get_gemm_log=get_gemm_log,
                collect_gemm_entries=collect_baseline_gemm,
            )
            run_gemm_log.extend(baseline_gemm_log)
            baseline_df = pd.DataFrame(baseline_metrics)
            baseline_df.to_csv(
                run_dir / "eval_metrics_baseline.csv",
                index=False,
            )
            if args.runs == 1:
                baseline_df.to_csv(
                    output_dir / "eval_metrics_baseline.csv",
                    index=False,
                )
            results["baseline"] = baseline_df
            run_summaries.setdefault("baseline", []).append(
                _summarize_run(baseline_df)
            )
            logging.info("Saved baseline metrics CSV for run %d", run_idx)

            if not args.skip_violation_analysis and args.skip_greenctx:
                _write_violation_artifacts(
                    run_dir=run_dir,
                    metrics_df=baseline_df,
                    gemm_log=baseline_gemm_log,
                    max_sm_count=args.total_sms,
                )
                logging.info(
                    "Saved baseline violation artifacts for run %d", run_idx
                )

        if not args.skip_greenctx:
            collect_greenctx_gemm = bool(
                args.dump_gemm_shapes or (not args.skip_violation_analysis)
            )
            trace_path = _resolve_trace_path(args.trace_path)
            ctrl = None
            v2_path: Path | None = None
            try:
                if collect_greenctx_gemm:
                    install_linear_hooks()
                ctrl, v2_path = _prepare_greenctx_controller(
                    trace_path=trace_path,
                    total_sms=args.total_sms,
                    device_id=0,
                    GreenContextConfig=GreenContextConfig,
                    GreenContextController=GreenContextController,
                    LdpcTraceAdapter=LdpcTraceAdapter,
                )
                if collect_greenctx_gemm:
                    set_greenctx(ctrl, reset_log=True)
                greenctx_metrics, greenctx_gemm_log = _run_mode(
                    mode_name="greenctx",
                    model_name=args.model,
                    seed=seed,
                    loader=loader,
                    device=device,
                    total_sms=args.total_sms,
                    torch=torch,
                    AutoModelForCausalLM=AutoModelForCausalLM,
                    controller=ctrl,
                    get_gemm_log=get_gemm_log,
                    collect_gemm_entries=collect_greenctx_gemm,
                )
            finally:
                if collect_greenctx_gemm:
                    set_greenctx(None, reset_log=True)
                if ctrl is not None:
                    ctrl.close()
                if v2_path is not None and v2_path.exists():
                    v2_path.unlink()

            run_gemm_log.extend(greenctx_gemm_log)
            greenctx_df = pd.DataFrame(greenctx_metrics)
            greenctx_df.to_csv(
                run_dir / "eval_metrics_greenctx.csv",
                index=False,
            )
            if args.runs == 1:
                greenctx_df.to_csv(
                    output_dir / "eval_metrics_greenctx.csv",
                    index=False,
                )
            results["greenctx"] = greenctx_df
            run_summaries.setdefault("greenctx", []).append(
                _summarize_run(greenctx_df)
            )
            logging.info("Saved greenctx metrics CSV for run %d", run_idx)

            if not args.skip_violation_analysis:
                _write_violation_artifacts(
                    run_dir=run_dir,
                    metrics_df=greenctx_df,
                    gemm_log=greenctx_gemm_log,
                    max_sm_count=args.total_sms,
                )
                logging.info("Saved violation artifacts for run %d", run_idx)

        combined_frames = []
        for mode_name, df in results.items():
            mode_df = df.copy()
            mode_df.insert(0, "mode", mode_name)
            mode_df.insert(1, "seed", int(seed))
            mode_df.insert(2, "run_idx", int(run_idx))
            combined_frames.append(mode_df)
        if combined_frames:
            run_metrics_df = pd.concat(combined_frames, ignore_index=True)
            run_metrics_df.to_csv(run_dir / "metrics.csv", index=False)
            if args.runs == 1:
                run_metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        if args.dump_gemm_shapes:
            _dump_gemm_shapes(
                str(run_dir),
                effective_num_steps,
                gemm_log=run_gemm_log,
            )

        if args.runs == 1:
            single_run_results = results
            single_run_gemm_log = run_gemm_log

    if args.runs > 1:
        aggregated: dict[str, Any] = {
            "runs": int(args.runs),
            "seed_base": int(base_seed),
            "num_steps": int(effective_num_steps),
            "metrics": {},
        }
        for mode_name, summaries in run_summaries.items():
            mode_payload: dict[str, Any] = {}
            for metric_name in (
                "tokens_per_sec",
                "step_time_ms",
                "swap_count",
                "swap_overhead_us",
            ):
                values = np.asarray(
                    [row[metric_name] for row in summaries],
                    dtype=float,
                )
                mode_payload[metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }
            aggregated["metrics"][mode_name] = mode_payload

        aggregated_path = output_dir / "aggregated.json"
        with open(aggregated_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        logging.info("Saved multi-run aggregation to %s", aggregated_path)
        if hooks_installed:
            set_greenctx(None, reset_log=True)
            torch.nn.Linear.forward = orig_linear_forward
            torch.nn.functional.linear = orig_functional_linear
            torch.Tensor.__matmul__ = orig_tensor_matmul
            torch.bmm = orig_bmm
        return

    if args.dump_gemm_shapes:
        _dump_gemm_shapes(
            str(output_dir),
            effective_num_steps,
            gemm_log=single_run_gemm_log,
        )

    _print_summary(single_run_results)
    _plot_comparison(
        results=single_run_results,
        output_dir=output_dir,
        plt=plt,
        np=np,
    )
    if hooks_installed:
        set_greenctx(None, reset_log=True)
        torch.nn.Linear.forward = orig_linear_forward
        torch.nn.functional.linear = orig_functional_linear
        torch.Tensor.__matmul__ = orig_tensor_matmul
        torch.bmm = orig_bmm
    logging.info("Saved comparison plots in %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    _run_benchmark(args)


if __name__ == "__main__":
    main()
