from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from scripts import coordinator_scaling_analysis, run_multi_coordinator


def test_strong_scaling_keeps_global_work_fixed() -> None:
    # Given / When
    configs = run_multi_coordinator.build_scaling_configs("strong")

    # Then
    assert [config.coordinators for config in configs] == [1, 2]
    assert {config.total_devices for config in configs} == {8}
    assert {config.global_batch for config in configs} == {16}
    assert [
        (config.devices_per_coordinator, config.local_batch)
        for config in configs
    ] == [
        (8, 16),
        (4, 8),
    ]
    assert all(config.warmup_iterations == 1 for config in configs)
    assert all(config.measured_iterations == 3 for config in configs)


def test_weak_scaling_keeps_local_work_fixed() -> None:
    # Given / When
    configs = run_multi_coordinator.build_scaling_configs("weak")

    # Then
    assert [config.coordinators for config in configs] == [1, 2]
    assert {config.devices_per_coordinator for config in configs} == {4}
    assert {config.local_batch for config in configs} == {8}
    assert [
        (config.total_devices, config.global_batch) for config in configs
    ] == [
        (4, 8),
        (8, 16),
    ]


@pytest.mark.parametrize(
    ("rank", "expected_node", "expected_range"),
    [(0, 0, "0-27"), (1, 1, "28-55")],
)
def test_rank_command_pins_each_rank_to_its_numa_node(
    rank: int,
    expected_node: int,
    expected_range: str,
    tmp_path: Path,
) -> None:
    # Given
    config = run_multi_coordinator.build_scaling_configs("strong")[1]

    # When
    launch = run_multi_coordinator.build_rank_launch(
        config=config,
        rank=rank,
        output_directory=tmp_path,
    )

    # Then
    assert launch.numa_node == expected_node
    assert launch.cpu_range == expected_range
    assert launch.command[:4] == (
        "numactl",
        f"--cpunodebind={expected_node}",
        f"--membind={expected_node}",
        "python3",
    )
    assert launch.environment["RANK"] == str(rank)
    assert launch.environment["WORLD_SIZE"] == "2"
    assert launch.substrate == (
        "same-host Gloo loopback lower bound; NUMA CPU and memory binding"
    )


def test_affinity_fallback_labels_cpu_first_touch_substrate(
    tmp_path: Path,
) -> None:
    # Given
    config = dataclasses.replace(
        run_multi_coordinator.build_scaling_configs("weak")[1],
        affinity_mode="taskset",
    )

    # When
    launch = run_multi_coordinator.build_rank_launch(
        config=config,
        rank=1,
        output_directory=tmp_path,
    )

    # Then
    assert launch.command[:3] == ("taskset", "--cpu-list", "28-55")
    assert launch.substrate == (
        "same-host Gloo loopback lower bound; CPU affinity with first-touch memory"
    )


def test_affinity_selection_falls_back_when_memory_binding_is_unavailable() -> (
    None
):
    # Given / When
    privileged = run_multi_coordinator.select_affinity_mode(
        numactl_available=True, memory_binding_available=True
    )
    restricted = run_multi_coordinator.select_affinity_mode(
        numactl_available=True, memory_binding_available=False
    )

    # Then
    assert privileged == "numactl"
    assert restricted == "taskset"


def test_scaling_efficiency_uses_strong_and_weak_definitions() -> None:
    # Given / When
    strong = coordinator_scaling_analysis.scaling_efficiency(
        mode="strong",
        baseline_seconds=12.0,
        measured_seconds=7.5,
        coordinators=2,
    )
    weak = coordinator_scaling_analysis.scaling_efficiency(
        mode="weak",
        baseline_seconds=12.0,
        measured_seconds=15.0,
        coordinators=2,
    )

    # Then
    assert strong == pytest.approx(0.8)
    assert weak == pytest.approx(0.8)


def test_breakdown_reconciles_device_communication_optimizer_and_residual() -> (
    None
):
    # Given / When
    breakdown = coordinator_scaling_analysis.reconcile_breakdown(
        iteration_total_seconds=10.0,
        forward_device_seconds=2.0,
        backward_device_seconds=3.0,
        gradient_sync_seconds=1.5,
        optimizer_seconds=0.5,
    )

    # Then
    assert breakdown.device_dispatch_aggregation_seconds == 5.0
    assert breakdown.gradient_sync_seconds == 1.5
    assert breakdown.optimizer_seconds == 0.5
    assert breakdown.idle_other_seconds == 3.0
    assert breakdown.reconciles is True


def test_breakdown_rejects_components_exceeding_iteration_total() -> None:
    # Given / When / Then
    with pytest.raises(
        coordinator_scaling_analysis.BreakdownReconciliationError
    ):
        coordinator_scaling_analysis.reconcile_breakdown(
            iteration_total_seconds=1.0,
            forward_device_seconds=0.5,
            backward_device_seconds=0.4,
            gradient_sync_seconds=0.3,
            optimizer_seconds=0.2,
        )


def test_rank_result_records_warmup_separately_and_measured_throughput() -> (
    None
):
    # Given
    phases = {
        "warmup_iteration": 4.0,
        "iteration_total": 6.0,
        "forward_device_dispatch_aggregation": 1.0,
        "backward_device_dispatch_aggregation": 2.0,
        "gradient_sync": 1.0,
        "optimizer": 0.5,
    }

    # When
    result = run_multi_coordinator.build_rank_result(
        run_multi_coordinator.build_scaling_configs("strong")[1],
        run_multi_coordinator.RankMeasurement(
            rank=0,
            phase_durations=phases,
            measured_losses=(2.0, 1.5, 1.0),
            golden_losses=(2.0, 1.5, 1.0),
        ),
    )

    # Then
    assert result.warmup_iteration_seconds == 4.0
    assert result.iteration_total_seconds == 6.0
    assert result.device_dispatch_aggregation_seconds == 3.0
    assert result.gradient_sync_seconds == 1.0
    assert result.optimizer_seconds == 0.5
    assert result.idle_other_seconds == 1.5
    assert result.throughput_samples_per_second == 4.0
    assert result.loss_correctness.passed is True


def test_global_result_uses_slowest_rank_as_iteration_makespan() -> None:
    # Given
    config = run_multi_coordinator.build_scaling_configs("weak")[1]
    first = run_multi_coordinator.build_rank_result(
        config,
        run_multi_coordinator.RankMeasurement(
            rank=0,
            phase_durations={"iteration_total": 6.0},
            measured_losses=(2.0, 1.5, 1.0),
            golden_losses=(2.0, 1.5, 1.0),
        ),
    )
    second = dataclasses.replace(
        first,
        rank=1,
        iteration_total_seconds=8.0,
        throughput_samples_per_second=3.0,
    )

    # When
    result = run_multi_coordinator.build_global_result(config, (first, second))

    # Then
    assert result.iteration_total_seconds == 8.0
    assert result.throughput_samples_per_second == 6.0
    assert result.loss_correctness.passed is True
    assert result.substrate.startswith("same-host Gloo loopback lower bound")


def test_iteration_plan_has_one_warmup_before_three_measured_iterations() -> (
    None
):
    # Given
    config = run_multi_coordinator.build_scaling_configs("strong")[0]

    # When
    plan = run_multi_coordinator.build_iteration_plan(config)

    # Then
    assert plan == ("warmup", "measured", "measured", "measured")


def test_rank_result_payload_contains_required_phase_and_loss_fields() -> None:
    # Given
    config = run_multi_coordinator.build_scaling_configs("strong")[0]
    result = run_multi_coordinator.build_rank_result(
        config,
        run_multi_coordinator.RankMeasurement(
            rank=0,
            phase_durations={
                "warmup_iteration": 1.0,
                "iteration_total": 3.0,
                "forward_device_dispatch_aggregation": 0.5,
                "backward_device_dispatch_aggregation": 1.0,
                "gradient_sync": 0.5,
                "optimizer": 0.25,
            },
            measured_losses=(3.0, 2.0, 1.0),
            golden_losses=(3.0, 2.0, 1.0),
        ),
    )

    # When
    payload = run_multi_coordinator.rank_result_payload(config, result)

    # Then
    assert payload["warmup_iterations"] == 1
    assert payload["measured_iterations"] == 3
    assert payload["loss_correctness"]["passed"] is True
    assert payload["phases_seconds"] == {
        "iteration_total": 3.0,
        "device_dispatch_aggregation": 1.5,
        "gradient_sync": 0.5,
        "optimizer": 0.25,
        "idle_other": 0.75,
    }


def test_experiment_plan_covers_strong_and_weak_without_writing_results(
    tmp_path: Path,
) -> None:
    # Given / When
    plan = run_multi_coordinator.build_experiment_plan(
        modes=("strong", "weak"), output_root=tmp_path
    )

    # Then
    assert [(run.config.mode, run.config.coordinators) for run in plan] == [
        ("strong", 1),
        ("strong", 2),
        ("weak", 1),
        ("weak", 2),
    ]
    assert len({run.master_port for run in plan}) == 4
    assert plan[0].run_directory == tmp_path / "strong" / "coordinators-1"
    assert (tmp_path / "strong").exists() is False


def test_scaling_conclusion_reports_speedup_and_efficiency() -> None:
    # Given
    config_one, config_two = run_multi_coordinator.build_scaling_configs(
        "strong"
    )
    baseline = run_multi_coordinator.GlobalResult(
        coordinators=1,
        iteration_total_seconds=12.0,
        throughput_samples_per_second=4.0,
        loss_correctness=run_multi_coordinator.LossCorrectness(
            True, True, True, 0.0
        ),
        substrate="same-host Gloo loopback lower bound",
        ranks=(),
    )
    scaled = dataclasses.replace(
        baseline,
        coordinators=2,
        iteration_total_seconds=7.5,
        throughput_samples_per_second=6.4,
    )

    # When
    conclusion = run_multi_coordinator.build_scaling_conclusion(
        config_one, config_two, baseline, scaled
    )

    # Then
    assert conclusion.speedup == pytest.approx(1.6)
    assert conclusion.efficiency == pytest.approx(0.8)
    assert conclusion.throughput_ratio == pytest.approx(1.6)
