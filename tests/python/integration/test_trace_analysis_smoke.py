from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


class _AdapterProto(Protocol):
    def __init__(self, csv_path: str | Path, total_sms: int = 48) -> None: ...

    def to_v2_file(self, path: str | Path) -> None: ...


class _BackendProto(Protocol):
    def load_trace(self, path: str) -> bool: ...


class _BundleProto(Protocol):
    sm_count: int


class _StepScopeProto(Protocol):
    def __enter__(self) -> _BundleProto: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool | None: ...


class _ControllerProto(Protocol):
    backend: _BackendProto

    def step_scope(self, step_idx: int) -> _StepScopeProto: ...

    def close(self) -> None: ...


class _ControllerClassProto(Protocol):
    @classmethod
    def from_config(cls, device_id: int, cfg: object) -> _ControllerProto: ...


class _ConfigClassProto(Protocol):
    def __call__(
        self,
        *,
        enabled: bool,
        backend: str,
        trace_path: str,
    ) -> object: ...


def _bootstrap_morphling() -> None:
    if "morphling" not in sys.modules:
        morphling_mod = types.ModuleType("morphling")
        morphling_mod.__path__ = [str(REPO_ROOT / "morphling")]
        morphling_mod.__package__ = "morphling"
        sys.modules["morphling"] = morphling_mod

    if "morphling.runtime" not in sys.modules:
        runtime_mod = types.ModuleType("morphling.runtime")
        runtime_mod.__path__ = [str(REPO_ROOT / "morphling" / "runtime")]
        runtime_mod.__package__ = "morphling.runtime"
        sys.modules["morphling.runtime"] = runtime_mod


def _load_adapter_cls() -> type[_AdapterProto]:
    module_path = REPO_ROOT / "morphling" / "runtime" / "ldpc_trace_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "ldpc_trace_adapter", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load adapter module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(type[_AdapterProto], getattr(module, "LdpcTraceAdapter"))


def _run_python(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *cmd],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def _write_first_n_rows(src: Path, dst: Path, n: int) -> None:
    with src.open("r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        with dst.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()
            for idx, row in enumerate(reader):
                if idx >= n:
                    break
                writer.writerow(row)


def test_plot_script_runs(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "plot_real_traces.py"
    assert script.exists(), f"missing script: {script}"

    output_dir = tmp_path / "figures"
    with_ctrl = REPO_ROOT / "data" / "ldpc_trace_with_ctrl.csv"
    without_ctrl = REPO_ROOT / "data" / "ldpc_trace_without_ctrl.csv"
    if not with_ctrl.exists() or not without_ctrl.exists():
        pytest.skip("LDPC trace CSV files not present in data/")

    help_result = _run_python([str(script), "--help"])
    has_num_samples = "--num-samples" in help_result.stdout

    if has_num_samples:
        result = _run_python(
            [
                str(script),
                "--with-ctrl",
                str(with_ctrl),
                "--without-ctrl",
                str(without_ctrl),
                "--output-dir",
                str(output_dir),
                "--num-samples",
                "100",
            ]
        )
    else:
        with_ctrl_small = tmp_path / "with_ctrl_100.csv"
        without_ctrl_small = tmp_path / "without_ctrl_100.csv"
        _write_first_n_rows(with_ctrl, with_ctrl_small, 100)
        _write_first_n_rows(without_ctrl, without_ctrl_small, 100)
        result = _run_python(
            [
                str(script),
                "--with-ctrl",
                str(with_ctrl_small),
                "--without-ctrl",
                str(without_ctrl_small),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert result.returncode == 0, (
        "plot_real_traces.py failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    stems = [
        "deadline_compliance",
        "decode_latency_cdf",
        "timeline_decode_and_sm",
        "workload_vs_latency",
        "violation_inefficiency_events",
    ]
    for stem in stems:
        assert (output_dir / f"{stem}.pdf").exists()
        assert (output_dir / f"{stem}.png").exists()


def test_adapter_round_trip(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is required for GreenContextController off backend")

    class _CudaProto(Protocol):
        def is_available(self) -> bool: ...

    class _TorchProto(Protocol):
        cuda: _CudaProto

    torch = cast(_TorchProto, cast(object, import_module("torch")))
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for OffBackend stream creation")

    _bootstrap_morphling()
    green_context_mod = import_module("morphling.runtime.green_context")
    GreenContextConfig = cast(
        _ConfigClassProto,
        getattr(green_context_mod, "GreenContextConfig"),
    )
    GreenContextController = cast(
        _ControllerClassProto,
        getattr(green_context_mod, "GreenContextController"),
    )

    source_csv = REPO_ROOT / "data" / "ldpc_trace_with_ctrl.csv"
    if not source_csv.exists():
        pytest.skip("LDPC trace CSV not present in data/")
    v2_csv = tmp_path / "trace_v2.csv"

    LdpcTraceAdapter = _load_adapter_cls()
    adapter: _AdapterProto = LdpcTraceAdapter(source_csv, total_sms=48)
    adapter.to_v2_file(v2_csv)
    assert v2_csv.exists()

    with v2_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert list(rows[0].keys()) == ["timestamp_ns", "num_sms", "tag"]

    cfg: object = GreenContextConfig(
        enabled=True,
        backend="off",
        trace_path=str(v2_csv),
    )
    ctrl = GreenContextController.from_config(device_id=0, cfg=cfg)
    try:
        assert ctrl.backend.load_trace(str(v2_csv))
        with ctrl.step_scope(0) as bundle:
            assert bundle.sm_count == 0
    finally:
        ctrl.close()


def test_eval_script_runs(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "eval_greenctx_training.py"
    if not script.exists():
        pytest.skip(f"missing script: {script}")

    output_dir = tmp_path / "eval_output"
    result = _run_python(
        [
            str(script),
            "--num-steps",
            "5",
            "--skip-greenctx",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result.returncode == 0, (
        "eval_greenctx_training.py failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    baseline_csv = output_dir / "eval_metrics_baseline.csv"
    assert baseline_csv.exists()

    with baseline_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
