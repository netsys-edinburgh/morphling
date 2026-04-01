from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from typing import Callable, NoReturn, Protocol, TypeVar, cast

_FuncT = TypeVar("_FuncT", bound=Callable[..., object])


class _PytestModuleProto(Protocol):
    def skip(self, reason: str) -> NoReturn: ...

    def fixture(
        self, *, scope: str | None = ...
    ) -> Callable[[_FuncT], _FuncT]: ...


class _CudaStreamProto(Protocol):
    def synchronize(self) -> None: ...


class _CudaProto(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def set_device(self, device: int) -> None: ...

    def stream(
        self, stream: _CudaStreamProto
    ) -> AbstractContextManager[None]: ...


class _LossScalarProto(Protocol):
    def item(self) -> float: ...


class _LossProto(Protocol):
    def backward(self) -> None: ...

    def detach(self) -> _LossScalarProto: ...


class _ModelProto(Protocol):
    def to(self, device: object) -> "_ModelProto": ...

    def parameters(self) -> object: ...

    def __call__(self, x: object) -> object: ...


class _CriterionProto(Protocol):
    def __call__(self, output: object, target: object) -> _LossProto: ...


class _OptimizerProto(Protocol):
    def step(self) -> None: ...

    def zero_grad(self, set_to_none: bool = ...) -> None: ...


class _OptimProto(Protocol):
    def SGD(self, params: object, lr: float) -> _OptimizerProto: ...


class _TorchModuleProto(Protocol):
    cuda: _CudaProto
    optim: _OptimProto

    def manual_seed(self, seed: int) -> object: ...

    def device(self, spec: str) -> object: ...

    def randn(self, *size: int, device: object = ...) -> object: ...


class _NNModuleProto(Protocol):
    def Linear(self, in_features: int, out_features: int) -> object: ...

    def ReLU(self) -> object: ...

    def Sequential(self, *modules: object) -> _ModelProto: ...

    def MSELoss(self) -> _CriterionProto: ...


pytest = cast(_PytestModuleProto, cast(object, import_module("pytest")))
torch = cast(_TorchModuleProto, cast(object, import_module("torch")))
nn = cast(_NNModuleProto, cast(object, import_module("torch.nn")))


def _bootstrap_morphling() -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if "morphling" not in sys.modules:
        morphling_mod = types.ModuleType("morphling")
        morphling_mod.__path__ = [str(root / "morphling")]
        morphling_mod.__package__ = "morphling"
        sys.modules["morphling"] = morphling_mod

    if "morphling.runtime" not in sys.modules:
        runtime_mod = types.ModuleType("morphling.runtime")
        runtime_mod.__path__ = [str(root / "morphling" / "runtime")]
        runtime_mod.__package__ = "morphling.runtime"
        sys.modules["morphling.runtime"] = runtime_mod

    so_path = root / "morphling" / "_GreenCtx.so"
    if so_path.exists() and "morphling._GreenCtx" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "morphling._GreenCtx", str(so_path)
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load extension: {so_path}")
        gc_module = importlib.util.module_from_spec(spec)
        sys.modules["morphling._GreenCtx"] = gc_module
        spec.loader.exec_module(gc_module)


_bootstrap_morphling()

from morphling.runtime.green_context import (  # noqa: E402
    GreenContextConfig,
    GreenContextController,
)
from morphling.runtime.green_context_backends import (  # noqa: E402
    CppBackend,
    StreamBundle,
)


def _get_preferred_gpu() -> int:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPUs available")

    for gid in (7, 6, 5, 0):
        if gid < torch.cuda.device_count():
            return gid

    pytest.skip("No preferred CUDA GPU found")


def _greenctx_supported(gpu_id: int) -> bool:
    try:
        backend = CppBackend(gpu_id=gpu_id)
        supported = backend.is_supported()
        backend.close()
        return supported
    except Exception:
        return False


@pytest.fixture(scope="module")
def gpu_id() -> int:
    gid = _get_preferred_gpu()
    if not _greenctx_supported(gid):
        pytest.skip(f"Green contexts not supported on GPU {gid}")
    return gid


@pytest.fixture(scope="module")
def trace_path() -> str:
    root = Path(__file__).resolve().parents[3]
    path = root / "tests" / "data" / "greenctx" / "second_level.trace"
    if not path.exists():
        pytest.skip("Trace file not found")
    return str(path)


def test_greenctx_training_trace_e2e(gpu_id: int, trace_path: str) -> None:
    _ = torch.manual_seed(7)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    cfg = GreenContextConfig(
        enabled=True,
        trace_path=trace_path,
        clock_mode="step",
    )
    ctrl = GreenContextController.from_config(device_id=gpu_id, cfg=cfg)

    if not ctrl.is_supported:
        ctrl.close()
        pytest.skip(f"Green contexts not supported on GPU {gpu_id}")

    expected_sm = {0: 8, 1: 32, 2: 64, 3: 16, 4: 48, 5: 64}
    observed_sm: dict[int, int] = {}
    losses: list[float] = []

    model: _ModelProto = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)
    criterion: _CriterionProto = nn.MSELoss()
    optimizer: _OptimizerProto = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(64, 256, device=device)
    target = torch.randn(64, 64, device=device)

    switch_count = 0
    try:
        optimizer.zero_grad(set_to_none=True)
        for step in range(6):
            with ctrl.step_scope(step) as bundle:
                assert isinstance(bundle, StreamBundle)
                observed_sm[step] = bundle.sm_count
                assert bundle.sm_count == expected_sm[step]
                comp_stream = cast(_CudaStreamProto, bundle.comp)

                with torch.cuda.stream(comp_stream):
                    output = model(x)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                comp_stream.synchronize()

            losses.append(float(loss.detach().item()))

        switch_count = ctrl.switch_count()
    finally:
        ctrl.close()

    assert observed_sm == expected_sm
    assert losses[-1] < losses[0]
    assert switch_count >= 4
