from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[4]
_TRACE_PATH = _ROOT / "tests" / "data" / "greenctx" / "second_level_wall.trace"


class _LocalMatmulBackend:
    def __init__(self) -> None:
        self._queue = []

    def async_dispatch_matmul(self, mat_a, mat_b) -> None:
        self._queue.append(torch.matmul(mat_a, mat_b.transpose(-2, -1)))

    def wait_matmul(self, _idx: int):
        if not self._queue:
            raise RuntimeError("wait_matmul called without pending result")
        return self._queue.pop(0)


class _DeactivateRecorder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def deactivate(self, sm_count: int) -> None:
        self.calls.append(sm_count)


class _TraceDrivenGreenCtx:
    def __init__(self, trace_path: Path) -> None:
        self._entries = self._load_trace_entries(trace_path)
        self._generation = 0
        self._active_sm: int | None = None
        self._backend = _DeactivateRecorder()

    @staticmethod
    def _load_trace_entries(trace_path: Path) -> list[tuple[int, int]]:
        unit = "us"
        entries: list[tuple[int, int]] = []

        for line in trace_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if "time_unit=" in stripped:
                    unit = stripped.split("time_unit=", maxsplit=1)[1].strip()
                continue
            if stripped.startswith("timestamp"):
                continue

            ts_text, sm_text, *_ = stripped.split(",")
            entries.append((int(ts_text), int(sm_text)))

        scale = 1 if unit == "us" else 1_000_000
        return [(ts * scale, sm) for ts, sm in entries]

    def activate_for_time(self, elapsed_us: int) -> tuple[int, int]:
        sm_count = self._entries[0][1]
        for ts_us, trace_sm in self._entries:
            if elapsed_us < ts_us:
                break
            sm_count = trace_sm

        if sm_count != self._active_sm:
            self._generation += 1
            self._active_sm = sm_count
        return sm_count, self._generation


def _bootstrap_backend_stub() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    if "morphling" not in sys.modules:
        morphling_mod = types.ModuleType("morphling")
        morphling_mod.__path__ = [str(_ROOT / "morphling")]
        morphling_mod.__package__ = "morphling"
        sys.modules["morphling"] = morphling_mod

    if "morphling.backend" not in sys.modules:
        backend_mod = types.ModuleType("morphling.backend")

        class _BaseBackend:
            pass

        setattr(backend_mod, "BaseBackend", _BaseBackend)
        backend_mod.__package__ = "morphling"
        sys.modules["morphling.backend"] = backend_mod


@pytest.fixture(scope="module")
def hooks_api():
    if not _TRACE_PATH.exists():
        pytest.fail("Wall-clock trace file not found")

    _bootstrap_backend_stub()

    try:
        hooks = importlib.import_module("morphling.hooks")
        autograd = importlib.import_module("morphling.hooks.autograd")
    except Exception as exc:
        pytest.fail(f"Unable to import morphling hooks: {exc}")
    return hooks, autograd


@pytest.fixture
def hooked_linear(hooks_api):
    hooks, autograd = hooks_api

    orig_linear_forward = torch.nn.Linear.forward
    orig_functional_linear = torch.nn.functional.linear
    orig_tensor_matmul = torch.Tensor.__matmul__
    orig_bmm = torch.bmm

    autograd._backend = _LocalMatmulBackend()
    greenctx = _TraceDrivenGreenCtx(_TRACE_PATH)
    hooks.apply_hooks("linear", greenctx=greenctx)

    try:
        yield hooks, greenctx
    finally:
        hooks.set_greenctx(None, reset_log=True)
        torch.nn.Linear.forward = orig_linear_forward
        torch.nn.functional.linear = orig_functional_linear
        torch.Tensor.__matmul__ = orig_tensor_matmul
        torch.bmm = orig_bmm


@pytest.fixture
def hooked_linear_real_controller(hooks_api):
    hooks, autograd = hooks_api

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is required for real GreenContextController select_backend('auto') integration"
        )

    try:
        runtime_mod = importlib.import_module("morphling.runtime.green_context")
        backends_mod = importlib.import_module(
            "morphling.runtime.green_context_backends"
        )
    except Exception as exc:
        pytest.skip(f"Unable to import green context runtime modules: {exc}")

    GreenContextConfig = runtime_mod.GreenContextConfig
    GreenContextController = runtime_mod.GreenContextController
    select_backend = backends_mod.select_backend

    backend = select_backend("auto", 0, strict=False)
    cfg = GreenContextConfig(
        enabled=True,
        backend="auto",
        trace_path=str(_TRACE_PATH),
        clock_mode="wall",
        strict=False,
    )
    controller = GreenContextController(backend=backend, config=cfg)

    if not controller.backend.load_trace(str(_TRACE_PATH)):
        controller.close()
        pytest.skip(
            "Selected green context backend could not load second_level_wall.trace"
        )

    if not controller.backend.is_supported():
        controller.close()
        pytest.skip(
            "No supported green context backend available for real-controller integration"
        )

    orig_linear_forward = torch.nn.Linear.forward
    orig_functional_linear = torch.nn.functional.linear
    orig_tensor_matmul = torch.Tensor.__matmul__
    orig_bmm = torch.bmm

    autograd._backend = _LocalMatmulBackend()
    hooks.apply_hooks("linear", greenctx=controller)

    try:
        yield hooks, controller
    finally:
        hooks.set_greenctx(None, reset_log=True)
        torch.nn.Linear.forward = orig_linear_forward
        torch.nn.functional.linear = orig_functional_linear
        torch.Tensor.__matmul__ = orig_tensor_matmul
        torch.bmm = orig_bmm
        controller.close()


def _build_model():
    _ = torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def test_per_gemm_log_has_forward_and_backward_phases(
    hooked_linear,
) -> None:
    hooks, greenctx = hooked_linear
    model = _build_model()

    x = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 4)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()

    log = hooks.get_gemm_log(reset=True)
    forward_entries = [entry for entry in log if entry["phase"] == "forward"]
    phases = {entry["phase"] for entry in log}

    assert len(forward_entries) == 2
    assert "forward" in phases
    assert "backward_grad_input" in phases
    assert "backward_grad_weight" in phases
    assert all(entry["sm_count"] is not None for entry in log)
    assert all(entry["duration_us"] >= 0.0 for entry in log)
    assert any(entry["greenctx_enabled"] for entry in log)
    assert len(greenctx._backend.calls) >= len(log)


def test_wall_trace_changes_sm_count_with_controlled_delay(
    hooked_linear,
) -> None:
    hooks, _greenctx = hooked_linear
    model = _build_model()

    x = torch.randn(2, 8)
    _ = model(x)
    time.sleep(1.1)
    _ = model(x)

    log = hooks.get_gemm_log(reset=True)
    forward_sm_counts = [
        entry["sm_count"]
        for entry in log
        if entry["phase"] == "forward" and entry["sm_count"] is not None
    ]

    assert len(forward_sm_counts) >= 4
    assert len(set(forward_sm_counts)) >= 2


def test_real_controller_auto_backend_wall_trace_path(
    hooked_linear_real_controller,
) -> None:
    hooks, controller = hooked_linear_real_controller
    model = _build_model()

    x = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 4)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()

    log = hooks.get_gemm_log(reset=True)
    phases = {entry["phase"] for entry in log}

    assert "forward" in phases
    assert "backward_grad_input" in phases
    assert "backward_grad_weight" in phases
    assert all(entry["greenctx_enabled"] for entry in log)
    assert all(entry["sm_count"] is not None for entry in log)
    assert controller.backend.switch_count() >= 0


def test_no_greenctx_no_log(hooks_api) -> None:
    hooks, autograd = hooks_api

    orig_linear_forward = torch.nn.Linear.forward
    orig_functional_linear = torch.nn.functional.linear
    orig_tensor_matmul = torch.Tensor.__matmul__
    orig_bmm = torch.bmm

    autograd._backend = _LocalMatmulBackend()
    hooks.apply_hooks("linear", greenctx=None)
    hooks.set_greenctx(None, reset_log=True)

    try:
        model = _build_model()
        x = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 4)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()

        assert hooks.get_gemm_log(reset=False) == []
        assert hooks.get_gemm_log(reset=True) == []
    finally:
        hooks.set_greenctx(None, reset_log=True)
        torch.nn.Linear.forward = orig_linear_forward
        torch.nn.functional.linear = orig_functional_linear
        torch.Tensor.__matmul__ = orig_tensor_matmul
        torch.bmm = orig_bmm
