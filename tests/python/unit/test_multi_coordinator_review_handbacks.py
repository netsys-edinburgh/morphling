from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from scripts import _multi_coordinator_training as training
from scripts import run_multi_coordinator
from scripts import run_multi_coordinator_scaling as scaling_runner


class FakeFleetProcess:
    def __init__(self, *, running: bool = True) -> None:
        self.running = running
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminated = True
        self.running = False

    def kill(self) -> None:
        self.killed = True
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


def _rank_result(
    rank: int, *, passed: bool = True
) -> run_multi_coordinator.RankResult:
    return run_multi_coordinator.RankResult(
        rank=rank,
        warmup_iteration_seconds=1.0,
        iteration_total_seconds=3.0 + rank,
        device_dispatch_aggregation_seconds=1.0,
        gradient_sync_seconds=0.5,
        optimizer_seconds=0.25,
        idle_other_seconds=1.25 + rank,
        throughput_samples_per_second=8.0,
        measured_losses=(3.0, 2.0, 1.0),
        golden_losses=(3.0, 2.0, 1.0),
        loss_correctness=run_multi_coordinator.LossCorrectness(
            passed=passed,
            decreasing=passed,
            tracks_golden=passed,
            max_relative_error=0.0 if passed else 0.2,
        ),
    )


def test_summary_serializes_executed_affinity_config(tmp_path: Path) -> None:
    # Given
    planned = run_multi_coordinator.build_experiment_plan(
        modes=("strong",), output_root=tmp_path
    )
    executed = run_multi_coordinator.apply_affinity_mode(planned, "taskset")
    results = tuple(
        run_multi_coordinator.build_global_result(
            run.config, (_rank_result(0),)
        )
        for run in executed
    )

    # When
    scaling_runner._write_summaries(tmp_path, executed, results, ("test",))

    # Then
    summary = json.loads((tmp_path / "strong.json").read_text(encoding="utf-8"))
    assert {
        point["configuration"]["affinity_mode"] for point in summary["points"]
    } == {"taskset"}


def test_relative_tolerance_controls_loss_correctness() -> None:
    # Given
    config = dataclasses.replace(
        run_multi_coordinator.build_scaling_configs("strong")[0],
        relative_tolerance=0.01,
    )

    # When
    result = run_multi_coordinator.build_rank_result(
        config,
        run_multi_coordinator.RankMeasurement(
            rank=0,
            phase_durations={"iteration_total": 3.0},
            measured_losses=(3.12, 2.08, 1.04),
            golden_losses=(3.0, 2.0, 1.0),
        ),
    )

    # Then
    assert result.loss_correctness.tracks_golden is False
    assert result.loss_correctness.passed is False


def test_rank_command_and_result_provenance_include_relative_tolerance(
    tmp_path: Path,
) -> None:
    # Given
    config = dataclasses.replace(
        run_multi_coordinator.build_scaling_configs("strong")[0],
        relative_tolerance=0.0125,
    )
    result = run_multi_coordinator.build_rank_result(
        config,
        run_multi_coordinator.RankMeasurement(
            rank=0,
            phase_durations={"iteration_total": 3.0},
            measured_losses=(3.0, 2.0, 1.0),
            golden_losses=(3.0, 2.0, 1.0),
        ),
    )

    # When
    launch = run_multi_coordinator.build_rank_launch(config, 0, tmp_path)
    payload = run_multi_coordinator.rank_result_payload(config, result)

    # Then
    assert launch.command[launch.command.index("--rtol") + 1] == "0.0125"
    assert payload["relative_tolerance"] == 0.0125


def test_rank_cleanup_terminates_only_owned_fleet_processes() -> None:
    # Given
    owned = (FakeFleetProcess(), FakeFleetProcess())
    unrelated = FakeFleetProcess()

    # When
    training._terminate_fleet(owned)

    # Then
    assert all(process.terminated and process.waited for process in owned)
    assert unrelated.terminated is False
    assert unrelated.waited is False


def test_global_correctness_aggregates_every_rank() -> None:
    # Given
    config = run_multi_coordinator.build_scaling_configs("strong")[1]

    # When
    result = run_multi_coordinator.build_global_result(
        config,
        (_rank_result(0, passed=True), _rank_result(1, passed=False)),
    )

    # Then
    assert result.loss_correctness.passed is False
    assert result.loss_correctness.decreasing is False
    assert result.loss_correctness.tracks_golden is False
    assert result.loss_correctness.max_relative_error == 0.2
