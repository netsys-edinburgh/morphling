from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest
import torch

from morphling.hooks import autograd
from scripts import _single_coordinator_training_workload as workload
from scripts import multi_coordinator_scaling_results as results
from scripts import run_multi_coordinator
from scripts import run_multi_coordinator_scaling as runner


class FakeRankProcess:
    def __init__(self, *, status: int = 0, timeout: bool = False) -> None:
        self.status = status
        self.timeout = timeout
        self.terminated = False
        self.killed = False
        self.wait_count = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_count += 1
        if self.timeout and self.wait_count == 1:
            raise subprocess.TimeoutExpired("rank", timeout)
        return self.status

    def terminate_group(self) -> None:
        self.terminated = True

    def kill_group(self) -> None:
        self.killed = True


def test_timeout_terminates_waits_and_kills_all_rank_groups() -> None:
    # Given
    ranks = (FakeRankProcess(timeout=True), FakeRankProcess())

    # When / Then
    with pytest.raises(subprocess.TimeoutExpired):
        runner._wait_for_rank_processes(ranks, timeout_seconds=1)
    assert all(rank.terminated and rank.killed for rank in ranks)
    assert all(rank.wait_count >= 2 for rank in ranks)


def test_rank_error_terminates_all_other_rank_groups() -> None:
    # Given
    ranks = (FakeRankProcess(status=7), FakeRankProcess())

    # When / Then
    with pytest.raises(subprocess.CalledProcessError):
        runner._wait_for_rank_processes(ranks, timeout_seconds=1)
    assert all(rank.terminated and rank.killed for rank in ranks)


def test_distributed_timeout_flows_to_command_and_config(tmp_path: Path) -> None:
    # Given
    config = dataclasses.replace(
        run_multi_coordinator.build_scaling_configs("strong")[0],
        distributed_timeout_seconds=37,
    )

    # When
    launch = run_multi_coordinator.build_rank_launch(config, 0, tmp_path)

    # Then
    assert launch.command[launch.command.index("--distributed-timeout-seconds") + 1] == "37"
    assert dataclasses.asdict(config)["distributed_timeout_seconds"] == 37


class EvalTrackingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.eval_called = False

    def eval(self) -> EvalTrackingModel:
        self.eval_called = True
        return self

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": torch.nn.functional.one_hot(input_ids, 4).float() * self.weight}


def test_single_coordinator_full_workload_uses_eval_mode() -> None:
    # Given
    model = EvalTrackingModel()

    # When
    workload.train_loop(
        model,
        steps=1,
        seed=1,
        batch_size=1,
        seq_length=2,
        device=torch.device("cpu"),
        vocab_size=4,
    )

    # Then
    assert model.eval_called is True


def test_matching_rising_trajectory_still_passes_equivalence() -> None:
    # Given
    config = run_multi_coordinator.build_scaling_configs("strong")[0]

    # When
    result = results.build_rank_result(
        config,
        results.RankMeasurement(0, {"iteration_total": 1.0}, (1.0, 1.01), (1.0, 1.01)),
    )

    # Then
    assert result.loss_correctness.passed is True
    assert result.loss_correctness.decreasing is False


@pytest.mark.parametrize("measured", [(float("nan"), 1.0), (1.0, 2.0)])
def test_nan_or_mismatch_fails_equivalence(measured: tuple[float, float]) -> None:
    config = run_multi_coordinator.build_scaling_configs("strong")[0]
    result = results.build_rank_result(
        config,
        results.RankMeasurement(0, {"iteration_total": 1.0}, measured, (1.0, 1.0)),
    )
    assert result.loss_correctness.passed is False


def test_reference_grad_input_uses_supported_matmul_signature() -> None:
    grad = torch.ones((2, 3))
    weight = torch.ones((3, 4))
    assert torch.equal(autograd._reference_grad_input(grad, weight), torch.matmul(grad, weight))
