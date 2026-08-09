#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false, reportUnusedCallResult=false
"""D1: real single-coordinator training loop through the proxy backend + fleet.

Runs a multi-step AdamW training loop where the transformer's ``nn.Linear``
GEMMs are dispatched to a real ``ProxySvr`` backend + fake-device fleet (the
same path the paper uses), while attention ``bmm`` and the wide ``lm_head`` stay
native so a full forward is tractable. It first trains the model natively
(golden) and then dispatched, from the same seed/inputs, and reports whether the
dispatched loss trajectory tracks golden.

Usage (inside the Docker image, with scripts/ mounted):
  python3 scripts/run_single_coordinator_training.py \
      --num_devices 2 --model_name facebook/opt-125m \
      --steps 6 --seq_length 128 --batch_size 1
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import morphling.hooks.autograd as hooks_autograd
from morphling.runtime.coordinator_metrics import (
    CoordinatorMetricsCollector,
    PhaseRecorder,
    add_metrics_arguments,
    metrics_config_from_args,
    start_metrics_collector,
    track_phase,
)
from scripts._runtime_common import (
    load_model_and_tokenizer,
    start_backend,
    wait_for_connections,
)
from scripts._single_coordinator_fleet import (
    cleanup_fake_fleet,
)
from scripts._single_coordinator_fleet import (
    spawn_fake_fleet as _spawn_fake_fleet,
)
from scripts._single_coordinator_training_workload import (
    apply_restricted_hook as _apply_restricted_hook,
)
from scripts._single_coordinator_training_workload import (
    infer_vocab_size as _infer_vocab_size,
)
from scripts._single_coordinator_training_workload import (
    make_tiny_model as _make_tiny_model,
)
from scripts._single_coordinator_training_workload import (
    remove_restricted_hook as _remove_restricted_hook,
)
from scripts._single_coordinator_training_workload import (
    train_loop as _train_loop,
)
from scripts._single_coordinator_training_workload import (
    train_loop_tiny as _train_loop_tiny,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_devices", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--backend", type=str, default="proxy")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--proxy_host", type=str, default="")
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a small MLP + regression loop so a full dispatched run "
        "completes quickly (correctness check without the opt-125m dispatch "
        "overhead).",
    )
    add_metrics_arguments(parser)
    args = parser.parse_args()

    device = torch.device("cpu")
    vocab_size = 0
    os.environ["NUM_DEVICES"] = str(args.num_devices)

    phase_recorder: PhaseRecorder | None = None
    metrics_collector: CoordinatorMetricsCollector | None = None
    metrics_config = metrics_config_from_args(args)
    if metrics_config is not None:
        phase_recorder = PhaseRecorder()
        metrics_collector = start_metrics_collector(
            metrics_config, phase_recorder
        )
        if metrics_collector is None:
            phase_recorder = None

    # ---- 1. Golden: native (no dispatch), same seed/inputs -----------------
    _remove_restricted_hook()
    if args.tiny:
        model_g = _make_tiny_model(args.seed).to(device)
        golden = _train_loop_tiny(
            model_g,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            device=device,
        )
    else:
        model_g, tokenizer = load_model_and_tokenizer(
            args.model_name, dtype=torch.float32
        )
        model_g = model_g.to(device)
        vocab_size = _infer_vocab_size(model_g, tokenizer)
        golden = _train_loop(
            model_g,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            device=device,
            vocab_size=vocab_size,
        )
    del model_g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GOLDEN (native) losses:", [round(x, 4) for x in golden], flush=True)

    # ---- 2. Start real proxy backend + fake fleet --------------------------
    backend = start_backend(
        backend_name=args.backend,
        block_size=args.block_size,
        cfg_path=args.cfg,
    )
    hooks_autograd._backend = backend
    with track_phase(phase_recorder, "idle_wait"):
        time.sleep(5)
    print(f"Spawning {args.num_devices} fast fake devices...", flush=True)
    _spawn_fake_fleet(args.num_devices, args.backend, args.proxy_host)
    if args.backend == "proxy":
        with track_phase(phase_recorder, "idle_wait"):
            connected = wait_for_connections(
                backend, min_devices=args.num_devices, timeout=120
            )
        if connected < args.num_devices:
            print(
                f"WARNING: only {connected}/{args.num_devices} devices connected",
                flush=True,
            )
    with track_phase(phase_recorder, "idle_wait"):
        time.sleep(5)

    # ---- 3. Dispatched: Linear GEMMs through the fleet ---------------------
    if args.tiny:
        model_d = _make_tiny_model(args.seed).to(device)
        _apply_restricted_hook()
        t0 = time.time()
        dispatched = _train_loop_tiny(
            model_d,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            device=device,
            phase_recorder=phase_recorder,
        )
        elapsed = time.time() - t0
    else:
        model_d, _ = load_model_and_tokenizer(
            args.model_name, dtype=torch.float32
        )
        model_d = model_d.to(device)
        _apply_restricted_hook()
        t0 = time.time()
        dispatched = _train_loop(
            model_d,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            device=device,
            vocab_size=vocab_size,
            phase_recorder=phase_recorder,
        )
        elapsed = time.time() - t0
    _remove_restricted_hook()
    print(
        "DISPATCHED losses:",
        [round(x, 4) if x == x else x for x in dispatched],
        flush=True,
    )
    print(
        f"dispatched wall time: {elapsed:.1f}s for {args.warmup_steps} warmup "
        f"and {args.steps} measured steps",
        flush=True,
    )

    # ---- 4. Verdict --------------------------------------------------------
    any_nan = any(x != x for x in dispatched)
    decreasing = (not any_nan) and dispatched[-1] < dispatched[0]
    tracks = (not any_nan) and all(
        abs(d - g) <= args.rtol * max(abs(g), 1e-6)
        for d, g in zip(dispatched, golden)
    )
    max_rel = (
        max(abs(d - g) / max(abs(g), 1e-6) for d, g in zip(dispatched, golden))
        if not any_nan
        else float("nan")
    )
    print(f"any NaN: {any_nan}", flush=True)
    print(f"loss decreasing: {decreasing}", flush=True)
    print(
        f"full trajectory tracks golden (rtol={args.rtol}): {tracks} "
        f"(max rel err {max_rel})",
        flush=True,
    )

    try:
        if hasattr(backend, "stop"):
            backend.stop()
    except Exception:
        pass
    cleanup_fake_fleet()

    ok = (not any_nan) and decreasing and tracks
    print("D1 RESULT:", "PASS" if ok else "FAIL", flush=True)
    if metrics_collector is not None:
        metrics_collector.stop()
    # ProxySvr's libevent loop thread is non-daemon with no stop binding, so a
    # normal return leaves the process alive; force-exit after the verdict.
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
