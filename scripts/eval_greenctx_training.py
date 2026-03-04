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
from importlib import import_module
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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
        runtime_mod.__path__ = [
            str(REPO_ROOT / "morphling" / "runtime")
        ]
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
) -> list[dict[str, float | int]]:
    _set_seed(torch, seed)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    metrics: list[dict[str, float | int]] = []

    logging.info("Starting mode: %s", mode_name)
    for step_idx, (input_ids_cpu,) in enumerate(loader):
        input_ids = input_ids_cpu.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

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

        wall_time_s = max(time.perf_counter() - start_t, 1e-12)
        tokens_per_sec = float(input_ids.numel()) / wall_time_s
        metrics.append(
            {
                "step_idx": int(step_idx),
                "wall_time_ms": wall_time_s * 1000.0,
                "loss": float(loss.detach().item()),
                "tokens_per_sec": tokens_per_sec,
                "gpu_mem_mb": float(torch.cuda.memory_allocated(device) / 1e6),
                "sm_count": int(sm_count),
                "is_warmup": int(step_idx < WARMUP_STEPS),
            }
        )

    del model
    torch.cuda.empty_cache()
    return metrics


def _timed_rows(df: Any) -> Any:
    timed = df.loc[df["step_idx"] >= WARMUP_STEPS]
    if len(timed) == 0:
        return df
    return timed


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
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    _set_seed(torch, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = int(tokenizer.vocab_size)

    loader = _make_loader(
        torch,
        DataLoader,
        TensorDataset,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        seed=args.seed,
    )

    if len(loader) != args.num_steps:
        raise RuntimeError(
            "Unexpected synthetic loader length: "
            f"got {len(loader)}, expected {args.num_steps}"
        )

    results: dict[str, Any] = {}

    if not args.skip_baseline:
        baseline_metrics = _run_mode(
            mode_name="baseline",
            model_name=args.model,
            seed=args.seed,
            loader=loader,
            device=device,
            total_sms=args.total_sms,
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            controller=None,
        )
        baseline_df = pd.DataFrame(baseline_metrics)
        baseline_df.to_csv(
            output_dir / "eval_metrics_baseline.csv",
            index=False,
        )
        results["baseline"] = baseline_df
        logging.info("Saved baseline metrics CSV")

    if not args.skip_greenctx:
        trace_path = _resolve_trace_path(args.trace_path)
        ctrl = None
        v2_path: Path | None = None
        try:
            ctrl, v2_path = _prepare_greenctx_controller(
                trace_path=trace_path,
                total_sms=args.total_sms,
                device_id=0,
                GreenContextConfig=GreenContextConfig,
                GreenContextController=GreenContextController,
                LdpcTraceAdapter=LdpcTraceAdapter,
            )
            greenctx_metrics = _run_mode(
                mode_name="greenctx",
                model_name=args.model,
                seed=args.seed,
                loader=loader,
                device=device,
                total_sms=args.total_sms,
                torch=torch,
                AutoModelForCausalLM=AutoModelForCausalLM,
                controller=ctrl,
            )
        finally:
            if ctrl is not None:
                ctrl.close()
            if v2_path is not None and v2_path.exists():
                v2_path.unlink()

        greenctx_df = pd.DataFrame(greenctx_metrics)
        greenctx_df.to_csv(
            output_dir / "eval_metrics_greenctx.csv",
            index=False,
        )
        results["greenctx"] = greenctx_df
        logging.info("Saved greenctx metrics CSV")

    _print_summary(results)
    _plot_comparison(
        results=results,
        output_dir=output_dir,
        plt=plt,
        np=np,
    )
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
