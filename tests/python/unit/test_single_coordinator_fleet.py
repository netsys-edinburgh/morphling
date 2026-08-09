from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import _single_coordinator_fleet as fleet


class FakeProcess:
    def __init__(self, *, requires_kill: bool = False) -> None:
        self.requires_kill = requires_kill
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.requires_kill and self.kill_calls == 0:
            raise subprocess.TimeoutExpired("fake", timeout)
        return 0


def test_spawn_cleans_only_owned_processes_and_closes_parent_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    stale = FakeProcess()
    created: list[FakeProcess] = []
    parent_logs = []
    fleet._device_processes[:] = [stale]
    monkeypatch.setenv("CFG_PATH", str(tmp_path / "proxy.ini"))
    monkeypatch.setenv("MORPHLING_DEV_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(fleet.torch.cuda, "device_count", lambda: 2)

    def fail_global_command(*_args, **_kwargs) -> None:
        raise AssertionError("fleet lifecycle must not run global commands")

    def fake_popen(*_args, **kwargs):
        process = FakeProcess()
        created.append(process)
        parent_logs.append(kwargs["stdout"])
        return process

    monkeypatch.setattr(fleet.subprocess, "run", fail_global_command)
    monkeypatch.setattr(fleet.subprocess, "Popen", fake_popen)

    # When
    fleet.spawn_fake_fleet(2, "proxy", "")

    # Then
    assert stale.terminate_calls == 1
    assert stale.wait_calls == 1
    assert fleet._device_processes == created
    assert all(log.closed for log in parent_logs)
    fleet.cleanup_fake_fleet()
    assert all(process.terminate_calls == 1 for process in created)
    assert fleet._device_processes == []


def test_cleanup_kills_owned_process_that_does_not_terminate() -> None:
    # Given
    process = FakeProcess(requires_kill=True)
    fleet._device_processes[:] = [process]

    # When
    fleet.cleanup_fake_fleet()

    # Then
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 2
    assert fleet._device_processes == []
