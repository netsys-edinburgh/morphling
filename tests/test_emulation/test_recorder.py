import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from morphling.emulation.barrier import StructuralBarrier
from morphling.emulation.dilation import DilationMatrix
from morphling.emulation.recorder import EmulationRecorder
from morphling.emulation.step_runner import (
    BSPStepRunner,
    EmulationConfig,
    EmulationStepResult,
)
from morphling.simulator.output import export_json, format_comparison_table


def _make_step_result(step_idx: int, rank: int = 0) -> EmulationStepResult:
    return EmulationStepResult(
        step_idx=step_idx,
        wall_compute_s={rank: 0.010},
        wall_comm_s={rank: 0.005},
        virtual_compute_s={rank: 0.020},
        virtual_comm_s={rank: 0.005},
        virtual_total_s={rank: 0.025},
        step_virtual_time_s=0.025,
        barrier_overhead_s={"compute": 0.001, "comm": 0.001},
    )


class TestFinalize:
    def test_produces_simulation_result(self):
        recorder = EmulationRecorder(topology_name="test-emu")
        for i in range(3):
            recorder.record_step(_make_step_result(i))
        result = recorder.finalize()
        assert len(result.steps) == 3
        assert result.mean_step_time_s > 0
        assert result.topology == "test-emu"


class TestOutputCompatibility:
    def test_comparison_table(self):
        recorder = EmulationRecorder()
        for i in range(2):
            recorder.record_step(_make_step_result(i))
        result = recorder.finalize()
        table = format_comparison_table({"emulation": result})
        assert "emulation" in table
        assert "Mean Step (ms)" in table

    def test_json_export(self):
        recorder = EmulationRecorder()
        for i in range(2):
            recorder.record_step(_make_step_result(i))
        result = recorder.finalize()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        export_json({"emulation": result}, path)
        data = json.loads(Path(path).read_text())
        assert "emulation" in data
        assert "mean_step_time_s" in data["emulation"]


class TestWallTimeSummary:
    def test_returns_positive_values(self):
        recorder = EmulationRecorder()
        for i in range(2):
            recorder.record_step(_make_step_result(i))
        summary = recorder.wall_time_summary()
        assert summary["total_wall_time_s"] > 0
        assert summary["total_barrier_overhead_s"] >= 0
        assert summary["barrier_overhead_pct"] >= 0
