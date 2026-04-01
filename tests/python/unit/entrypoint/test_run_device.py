# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false, reportUnknownLambdaType=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

RUN_DEVICE_PATH = (
    Path(__file__).resolve().parents[4]
    / "morphling"
    / "entrypoint"
    / "run_device.py"
)


class _WorkerStub:
    def __init__(self):
        self.initialize_args = None
        self.started = False

    def initialize(self, cfg, device_id):
        self.initialize_args = (cfg, device_id)

    def start(self):
        self.started = True


class _AutoWorkerStub:
    calls = []
    workers = []

    @classmethod
    def from_name(cls, *args):
        cls.calls.append(args)
        worker = _WorkerStub()
        cls.workers.append(worker)
        return worker


def _load_run_device_module(monkeypatch):
    fake_morphling = types.ModuleType("morphling")
    fake_morphling.__path__ = []

    fake_common = types.ModuleType("morphling.common")
    fake_common.human2bytes = lambda value: int(value)
    fake_common.bytes2human = lambda value: str(value)

    fake_backend = types.ModuleType("morphling.backend")
    _AutoWorkerStub.calls = []
    _AutoWorkerStub.workers = []
    fake_backend.AutoWorker = _AutoWorkerStub

    monkeypatch.setitem(sys.modules, "morphling", fake_morphling)
    monkeypatch.setitem(sys.modules, "morphling.common", fake_common)
    monkeypatch.setitem(sys.modules, "morphling.backend", fake_backend)

    spec = importlib.util.spec_from_file_location(
        "run_device_under_test", RUN_DEVICE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proxy_backend_loads_cfg_and_sets_proxy_env(monkeypatch):
    module = _load_run_device_module(monkeypatch)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "morphling_emulator",
            "--id",
            "7",
            "--flops",
            "1000",
            "--memory",
            "2048",
            "--ul_bw",
            "3000",
            "--dl_bw",
            "4000",
            "--ul_lat",
            "0.01",
            "--dl_lat",
            "0.02",
            "--backend",
            "proxy",
            "--cfg",
            "/tmp/device.ini",
            "--proxy_host",
            "127.0.0.1:39000",
        ],
    )

    with mock.patch.object(module.time, "sleep", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            module.main()

    assert os.environ["MORPHLING_PIN_SIZE"] == "2048"
    assert os.environ["MORPHLING_PROXY_HOST"] == "127.0.0.1"
    assert os.environ["MORPHLING_PROXY_PORT"] == "39000"

    assert _AutoWorkerStub.calls == [("proxy",)]
    assert len(_AutoWorkerStub.workers) == 1
    worker = _AutoWorkerStub.workers[0]
    assert worker.initialize_args == ("/tmp/device.ini", 7)
    assert worker.started is True


def test_proxy_backend_rejects_invalid_proxy_host(monkeypatch, capsys):
    module = _load_run_device_module(monkeypatch)

    monkeypatch.delenv("MORPHLING_PROXY_HOST", raising=False)
    monkeypatch.delenv("MORPHLING_PROXY_PORT", raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "morphling_emulator",
            "--id",
            "1",
            "--flops",
            "1",
            "--memory",
            "2",
            "--ul_bw",
            "3",
            "--dl_bw",
            "4",
            "--ul_lat",
            "0.001",
            "--dl_lat",
            "0.002",
            "--backend",
            "proxy",
            "--proxy_host",
            "bad-format",
        ],
    )

    with mock.patch.object(module.time, "sleep") as sleep_mock:
        module.main()

    out = capsys.readouterr().out
    assert "Invalid proxy_host format" in out
    assert _AutoWorkerStub.calls == []
    sleep_mock.assert_not_called()
