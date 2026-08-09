from __future__ import annotations

from pathlib import Path

import torch

from scripts import _single_coordinator_training_workload as shared_workload
from scripts import run_multi_coordinator


def test_opt_strong_scaling_keeps_full_model_global_work_fixed() -> None:
    # Given / When
    configs = run_multi_coordinator.build_scaling_configs(
        "strong", run_multi_coordinator.OPT_125M_WORKLOAD
    )

    # Then
    assert [
        (config.coordinators, config.local_batch) for config in configs
    ] == [
        (1, 2),
        (2, 1),
    ]
    assert {config.total_devices for config in configs} == {8}
    assert {config.global_batch for config in configs} == {2}
    assert [(config.devices_per_coordinator) for config in configs] == [8, 4]
    assert all(config.model_name == "facebook/opt-125m" for config in configs)
    assert all(config.sequence_length == 32 for config in configs)
    assert all(config.measured_iterations == 2 for config in configs)
    assert all(config.block_size == 256 for config in configs)


def test_opt_weak_scaling_keeps_full_model_local_work_fixed() -> None:
    # Given / When
    configs = run_multi_coordinator.build_scaling_configs(
        "weak", run_multi_coordinator.OPT_125M_WORKLOAD
    )

    # Then
    assert [
        (config.coordinators, config.global_batch) for config in configs
    ] == [
        (1, 1),
        (2, 2),
    ]
    assert {config.local_batch for config in configs} == {1}
    assert {config.devices_per_coordinator for config in configs} == {4}
    assert [(config.total_devices) for config in configs] == [4, 8]


def test_full_model_rank_command_carries_model_sequence_and_block_size(
    tmp_path: Path,
) -> None:
    # Given
    config = run_multi_coordinator.build_scaling_configs(
        "strong", run_multi_coordinator.OPT_125M_WORKLOAD
    )[0]

    # When
    launch = run_multi_coordinator.build_rank_launch(config, 0, tmp_path)

    # Then
    assert launch.command[launch.command.index("--model-name") + 1] == (
        "facebook/opt-125m"
    )
    assert launch.command[launch.command.index("--sequence-length") + 1] == "32"
    assert launch.command[launch.command.index("--block_size") + 1] == "256"
    assert "--tiny" not in launch.command


def test_tiny_rank_command_preserves_quick_workload_mode(
    tmp_path: Path,
) -> None:
    # Given
    config = run_multi_coordinator.build_scaling_configs(
        "strong", run_multi_coordinator.TINY_WORKLOAD
    )[0]

    # When
    launch = run_multi_coordinator.build_rank_launch(config, 0, tmp_path)

    # Then
    assert "--tiny" in launch.command
    assert config.measured_iterations == 3
    assert config.global_batch == 16


def test_result_provenance_records_model_and_sequence_length() -> None:
    # Given
    config = run_multi_coordinator.build_scaling_configs(
        "weak", run_multi_coordinator.OPT_125M_WORKLOAD
    )[0]
    result = run_multi_coordinator.build_rank_result(
        config,
        run_multi_coordinator.RankMeasurement(
            rank=0,
            phase_durations={"iteration_total": 4.0},
            measured_losses=(2.0, 1.0),
            golden_losses=(2.0, 1.0),
        ),
    )

    # When
    payload = run_multi_coordinator.rank_result_payload(config, result)

    # Then
    assert payload["model_name"] == "facebook/opt-125m"
    assert payload["sequence_length"] == 32
    assert result.throughput_samples_per_second == 0.5


def test_deterministic_token_batch_supports_global_rank_split() -> None:
    # Given / When
    first = shared_workload.make_token_batch(
        shared_workload.TokenBatchSpec(42, 2, 4, 32),
        device=torch.device("cpu"),
    )
    second = shared_workload.make_token_batch(
        shared_workload.TokenBatchSpec(42, 2, 4, 32),
        device=torch.device("cpu"),
    )

    # Then
    assert torch.equal(first, second)
    assert first.shape == (2, 4)
    assert torch.equal(torch.cat((first[:1], first[1:])), first)


def test_worker_parser_accepts_full_model_flags() -> None:
    # Given / When
    args = run_multi_coordinator._parser().parse_args(
        [
            "--model-name",
            "facebook/opt-125m",
            "--sequence-length",
            "32",
            "--block_size",
            "256",
        ]
    )

    # Then
    assert args.model_name == "facebook/opt-125m"
    assert args.sequence_length == 32
    assert args.block_size == 256
