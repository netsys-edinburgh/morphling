from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import coordinator_scaling_execution, run_coordinator_scaling


def test_render_proxy_config_updates_device_count_and_port() -> None:
    # Given
    template = """[network]\nlisten_ip = 0.0.0.0\nlisten_port = 39000\n\n[worker]\nnum_device = 3\n"""

    # When
    rendered = coordinator_scaling_execution.render_proxy_config(
        template, device_count=8, port=39108
    )

    # Then
    assert "listen_ip = 127.0.0.1" in rendered
    assert "listen_port = 39108" in rendered
    assert "num_device = 8" in rendered
    assert "listen_port = 39000" not in rendered
    assert "num_device = 3" not in rendered


def test_build_inner_command_uses_proven_d1_settings(tmp_path: Path) -> None:
    # Given
    run = run_coordinator_scaling.RunSpec(
        device_count=4,
        repetition=1,
        port=39104,
        run_directory=tmp_path,
        container_directory=Path("/scaling-output/devices-4/rep-1"),
    )

    # When
    command = run_coordinator_scaling.build_inner_command(
        run, metrics_interval=0.2
    )

    # Then
    assert command[:2] == [
        "python3",
        "scripts/run_single_coordinator_training.py",
    ]
    assert command[command.index("--num_devices") + 1] == "4"
    assert command[command.index("--model_name") + 1] == "facebook/opt-125m"
    assert command[command.index("--warmup_steps") + 1] == "1"
    assert command[command.index("--steps") + 1] == "3"
    assert command[command.index("--batch_size") + 1] == "1"
    assert command[command.index("--seq_length") + 1] == "32"
    assert command[command.index("--block_size") + 1] == "256"
    assert command[command.index("--metrics_nics") + 1] == "lo"
    assert command[command.index("--cfg") + 1].endswith("proxy.ini")
    assert command[command.index("--metrics_output") + 1].endswith(
        "metrics.jsonl"
    )


def test_ports_are_unique_across_device_counts_and_repetitions() -> None:
    # Given
    run_count = 48

    # When
    ports = [
        run_coordinator_scaling.port_for_run(index)
        for index in range(run_count)
    ]

    # Then
    assert len(set(ports)) == run_count
    assert ports == sorted(ports)


def test_docker_command_has_deterministic_unique_name_and_auto_remove(
    tmp_path: Path,
) -> None:
    # Given
    run = run_coordinator_scaling.RunSpec(
        device_count=4,
        repetition=2,
        port=39107,
        run_directory=tmp_path / "run",
        container_directory=Path("/scaling-output/devices-4/rep-2"),
    )

    # When
    command = coordinator_scaling_execution.build_docker_command(
        run, tmp_path, "device-emulator:latest", 0.2
    )

    # Then
    assert command[command.index("--name") + 1] == "morphling-d2-d4-r2-p39107"
    assert "--rm" in command


def test_timeout_forcibly_removes_only_named_docker_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    run = run_coordinator_scaling.RunSpec(
        device_count=4,
        repetition=1,
        port=39103,
        run_directory=tmp_path / "run",
        container_directory=Path("/scaling-output/devices-4/rep-1"),
    )
    calls: list[list[str]] = []
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if path == Path("/.dockerenv"):
            return False
        return original_exists(path)

    def fake_run(command: list[str], **_kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(
                command, 1, output="partial", stderr="late"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(
        coordinator_scaling_execution.subprocess, "run", fake_run
    )

    # When
    coordinator_scaling_execution.run_command(
        run, tmp_path, "device-emulator:latest", 0.2, 1
    )

    # Then
    assert calls[1] == ["docker", "rm", "-f", "morphling-d2-d4-r1-p39103"]
    result = json.loads((run.run_directory / "result.json").read_text())
    assert result["timed_out"] is True


def test_normal_docker_completion_relies_only_on_auto_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    run = run_coordinator_scaling.RunSpec(
        device_count=2,
        repetition=1,
        port=39101,
        run_directory=tmp_path / "run",
        container_directory=Path("/scaling-output/devices-2/rep-1"),
    )
    calls: list[list[str]] = []
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if path == Path("/.dockerenv"):
            return False
        return original_exists(path)

    def fake_run(command: list[str], **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(
        coordinator_scaling_execution.subprocess, "run", fake_run
    )

    # When
    coordinator_scaling_execution.run_command(
        run, tmp_path, "device-emulator:latest", 0.2, 1
    )

    # Then
    assert len(calls) == 1
    assert "--rm" in calls[0]
    assert calls[0][0:3] != ["docker", "rm", "-f"]


def test_summary_keeps_good_repetition_when_other_metrics_are_invalid(
    tmp_path: Path,
) -> None:
    # Given
    for repetition in (1, 2):
        run_directory = tmp_path / "devices-1" / f"rep-{repetition}"
        run_directory.mkdir(parents=True)
        (run_directory / "result.json").write_text(
            json.dumps({"exit_status": 0}), encoding="utf-8"
        )
    valid = tmp_path / "devices-1" / "rep-1" / "metrics.jsonl"
    valid.write_text(
        json.dumps(
            {
                "record_type": "sample",
                "process_cpu_percent": 10.0,
                "rss_bytes": 100,
                "nic_tx_bytes_per_sec": 1.0,
                "nic_rx_bytes_per_sec": 2.0,
                "phase_counts": {"iteration_total": 1},
                "phase_durations_seconds": {"iteration_total": 3.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    invalid = tmp_path / "devices-1" / "rep-2" / "metrics.jsonl"
    invalid.write_text("{bad\n{}\n", encoding="utf-8")

    # When
    summary_path = run_coordinator_scaling._write_summary(
        tmp_path, (1,), 2, "image", ("command",)
    )
    summary = json.loads(summary_path.read_text())

    # Then
    assert summary["rows"][0]["successful_repetitions"] == 1
    assert summary["rows"][0]["runs"][1]["measurement"]["error"] is not None
