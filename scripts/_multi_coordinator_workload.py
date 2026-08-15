"""Tiny and OPT rank workloads sharing the single-coordinator model helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from morphling.runtime.coordinator_metrics import PhaseRecorder, track_phase
from scripts._runtime_common import load_model_and_tokenizer
from scripts._single_coordinator_training_workload import (
    TokenBatchSpec,
    apply_restricted_hook,
    extract_loss,
    infer_vocab_size,
    make_token_batch,
    remove_restricted_hook,
)
from scripts.multi_coordinator_scaling_config import ScalingConfig

LossFunction = Callable[[torch.nn.Module], torch.Tensor]


@dataclass(frozen=True, slots=True)
class ModelStep:
    model: torch.nn.Module
    loss_function: LossFunction


def _optimizer(
    config: ScalingConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def _native_losses(
    config: ScalingConfig,
    step_function: ModelStep,
) -> tuple[float, ...]:
    optimizer = _optimizer(config, step_function.model)
    losses: list[float] = []
    for step in range(config.warmup_iterations + config.measured_iterations):
        optimizer.zero_grad()
        loss = step_function.loss_function(step_function.model)
        loss.backward()
        if step >= config.warmup_iterations:
            losses.append(float(loss.detach()))
        optimizer.step()
    return tuple(losses)


def _overlap_from_env() -> bool:
    return os.environ.get(
        "MORPHLING_OVERLAP_GRADIENT_SYNC", "0"
    ).strip().lower() not in ("", "0", "false", "no")


def _register_async_gradient_reduction(
    parameters: list[torch.nn.Parameter],
    pending: list[tuple[torch.nn.Parameter, object]],
) -> list[object]:
    def make_hook(
        _target: torch.nn.Parameter,
    ) -> Callable[[torch.nn.Parameter], None]:
        def hook(param: torch.nn.Parameter) -> None:
            work = dist.all_reduce(
                param.grad, op=dist.ReduceOp.SUM, async_op=True
            )
            pending.append((param, work))

        return hook

    return [
        parameter.register_post_accumulate_grad_hook(make_hook(parameter))
        for parameter in parameters
        if parameter.requires_grad
    ]


def _finish_async_gradient_reduction(
    pending: list[tuple[torch.nn.Parameter, object]], coordinators: int
) -> None:
    for parameter, work in pending:
        work.wait()
        parameter.grad /= coordinators
    pending.clear()


def _reduce_gradients_blocking(
    parameters: list[torch.nn.Parameter], coordinators: int
) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad /= coordinators


def _distributed_losses(
    config: ScalingConfig,
    step_function: ModelStep,
    recorder: PhaseRecorder,
    overlap_gradient_sync: bool = False,
) -> tuple[float, ...]:
    optimizer = _optimizer(config, step_function.model)
    parameters = list(step_function.model.parameters())
    losses: list[float] = []
    pending: list[tuple[torch.nn.Parameter, object]] = []
    # With overlap enabled, each parameter launches its inter-coordinator
    # AllReduce as soon as backward finishes accumulating that gradient, so the
    # reduction runs concurrently with the remaining backward device dispatch
    # rather than as a serial phase afterward. Waiting for the async result
    # yields the identical reduced gradient, so training semantics are unchanged.
    hooks = (
        _register_async_gradient_reduction(parameters, pending)
        if overlap_gradient_sync
        else []
    )
    try:
        for step in range(
            config.warmup_iterations + config.measured_iterations
        ):
            warmup = step < config.warmup_iterations
            measured_recorder = None if warmup else recorder
            phase_name = "warmup_iteration" if warmup else "iteration_total"
            with track_phase(recorder, phase_name):
                optimizer.zero_grad()
                pending.clear()
                with track_phase(
                    measured_recorder, "forward_device_dispatch_aggregation"
                ):
                    loss = step_function.loss_function(step_function.model)
                with track_phase(
                    measured_recorder, "backward_device_dispatch_aggregation"
                ):
                    loss.backward()
                with track_phase(measured_recorder, "gradient_sync"):
                    if overlap_gradient_sync:
                        _finish_async_gradient_reduction(
                            pending, config.coordinators
                        )
                    else:
                        _reduce_gradients_blocking(
                            parameters, config.coordinators
                        )
                    global_loss = loss.detach().clone()
                    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
                    global_loss /= config.coordinators
                if not warmup:
                    losses.append(float(global_loss))
                with track_phase(measured_recorder, "optimizer"):
                    optimizer.step()
    finally:
        for handle in hooks:
            handle.remove()
    return tuple(losses)


def _tiny_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )


def _tiny_inputs(
    seed: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (
        torch.randn(batch_size, 8, generator=generator),
        torch.randn(batch_size, 4, generator=generator),
    )


def _run_tiny(
    rank: int,
    config: ScalingConfig,
    recorder: PhaseRecorder,
    overlap_gradient_sync: bool = False,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    inputs, targets = _tiny_inputs(42, config.global_batch)
    golden = _native_losses(
        config,
        ModelStep(
            _tiny_model(42),
            lambda model: F.mse_loss(model(inputs), targets),
        ),
    )
    lower = rank * config.local_batch
    local_inputs = inputs[lower : lower + config.local_batch]
    local_targets = targets[lower : lower + config.local_batch]
    dispatched = _distributed_losses(
        config,
        ModelStep(
            _tiny_model(42),
            lambda model: F.mse_loss(model(local_inputs), local_targets),
        ),
        recorder,
        overlap_gradient_sync,
    )
    return dispatched, golden


def _load_full_model(config: ScalingConfig) -> tuple[torch.nn.Module, int]:
    torch.manual_seed(42)
    model, tokenizer = load_model_and_tokenizer(
        config.model_name, dtype=torch.float32
    )
    return model.to(torch.device("cpu")), infer_vocab_size(model, tokenizer)


def _run_full(
    rank: int,
    config: ScalingConfig,
    recorder: PhaseRecorder,
    overlap_gradient_sync: bool = False,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    golden_model, vocab_size = _load_full_model(config)
    tokens = make_token_batch(
        TokenBatchSpec(
            seed=42,
            batch_size=config.global_batch,
            sequence_length=config.sequence_length,
            vocab_size=vocab_size,
        ),
        device=torch.device("cpu"),
    )
    golden = _native_losses(
        config,
        ModelStep(
            golden_model,
            lambda model: extract_loss(
                model(input_ids=tokens, labels=tokens), tokens
            ),
        ),
    )
    del golden_model
    dispatched_model, _ = _load_full_model(config)
    lower = rank * config.local_batch
    local_tokens = tokens[lower : lower + config.local_batch]
    apply_restricted_hook()
    try:
        dispatched = _distributed_losses(
            config,
            ModelStep(
                dispatched_model,
                lambda model: extract_loss(
                    model(input_ids=local_tokens, labels=local_tokens),
                    local_tokens,
                ),
            ),
            recorder,
            overlap_gradient_sync,
        )
    finally:
        remove_restricted_hook()
    return dispatched, golden


def run_workload(
    rank: int, config: ScalingConfig, recorder: PhaseRecorder
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    overlap_gradient_sync = _overlap_from_env()
    if config.tiny:
        return _run_tiny(rank, config, recorder, overlap_gradient_sync)
    return _run_full(rank, config, recorder, overlap_gradient_sync)
