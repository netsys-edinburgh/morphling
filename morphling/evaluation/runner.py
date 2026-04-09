from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifacts import (
    _dump_gemm_shapes,
    _summarize_run,
    _write_violation_artifacts,
)
from .bootstrap import REPO_ROOT, _load_greenctx_symbols, _resolve_trace_path
from .plotting import WARMUP_STEPS, _plot_comparison, _print_summary


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
        dataset, batch_size=batch_size, shuffle=False, drop_last=True
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
        prefix="ldpc_v2_", suffix=".csv", delete=False
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
            "Green context init failed with backend='%s': %s. Retrying with backend='off'.",
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
            "Starting run %d/%d (seed=%d)", run_idx + 1, args.runs, seed
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
                run_dir / "eval_metrics_baseline.csv", index=False
            )
            if args.runs == 1:
                baseline_df.to_csv(
                    output_dir / "eval_metrics_baseline.csv", index=False
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
                run_dir / "eval_metrics_greenctx.csv", index=False
            )
            if args.runs == 1:
                greenctx_df.to_csv(
                    output_dir / "eval_metrics_greenctx.csv", index=False
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
                str(run_dir), effective_num_steps, gemm_log=run_gemm_log
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
                    [row[metric_name] for row in summaries], dtype=float
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
            str(output_dir), effective_num_steps, gemm_log=single_run_gemm_log
        )

    _print_summary(single_run_results)
    _plot_comparison(
        results=single_run_results, output_dir=output_dir, plt=plt, np=np
    )
    if hooks_installed:
        set_greenctx(None, reset_log=True)
        torch.nn.Linear.forward = orig_linear_forward
        torch.nn.functional.linear = orig_functional_linear
        torch.Tensor.__matmul__ = orig_tensor_matmul
        torch.bmm = orig_bmm
    logging.info("Saved comparison plots in %s", output_dir)
