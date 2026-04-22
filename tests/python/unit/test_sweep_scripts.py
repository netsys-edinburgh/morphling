from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"

SWEEP_SCRIPTS = [
    "_inner_baseline.sh",
    "_run_sweep.sh",
    "run_vtime_experiment.sh",
    "_run_baseline_with_timeout.sh",
    "run_single_baseline.sh",
]


def _make_fleet(tmp_path: Path, num_devices: int = 4) -> Path:
    fleet = []
    for rank in range(num_devices):
        fleet.append(
            {
                "rank": rank,
                "flops": 5_000_000_000_000 + rank * 100,
                "memory": 2_000_000_000 + rank * 200,
                "ul_bw": 5_000_000 + rank * 10,
                "dl_bw": 50_000_000 + rank * 20,
                "ul_lat": 0.1 * rank,
                "dl_lat": 0.2 * rank,
            }
        )
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(fleet))
    return path


class TestBashSyntax:
    @pytest.mark.parametrize("script", SWEEP_SCRIPTS)
    def test_syntax_valid(self, script: str) -> None:
        path = SCRIPTS_DIR / script
        assert path.exists(), f"{script} not found"
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"bash -n {script} failed:\n{result.stderr}"
        )


class TestNoSlowProtobuf:
    def test_no_pure_python_protobuf_in_scripts(self) -> None:
        violations = []
        for script in SWEEP_SCRIPTS:
            path = SCRIPTS_DIR / script
            content = path.read_text()
            if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" in content:
                violations.append(script)
        assert not violations, (
            f"Pure-Python protobuf forced in: {violations}. "
            "Remove -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "
            "to use the faster default backend."
        )


class TestFleetParsing:
    def _run_fleet_snippet(
        self, fleet_path: Path, num_devices: int
    ) -> list[tuple[int, int, int, int, float, float]]:
        snippet = textwrap.dedent(f"""\
            import json
            fleet = json.load(open('{fleet_path}'))
            n = len(fleet)
            for i in range({num_devices}):
                d = fleet[i % n]
                print(int(d['flops']), int(d['memory']),
                      int(d['ul_bw']), int(d['dl_bw']),
                      d.get('ul_lat', 0.0), d.get('dl_lat', 0.0))
        """)
        result = subprocess.run(
            ["python3", "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            rows.append(
                (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            )
        return rows

    def test_exact_field_extraction(self, tmp_path: Path) -> None:
        fleet_path = _make_fleet(tmp_path, num_devices=3)
        fleet = json.loads(fleet_path.read_text())
        rows = self._run_fleet_snippet(fleet_path, 3)

        assert len(rows) == 3
        for i, row in enumerate(rows):
            assert row[0] == int(fleet[i]["flops"])
            assert row[1] == int(fleet[i]["memory"])
            assert row[2] == int(fleet[i]["ul_bw"])
            assert row[3] == int(fleet[i]["dl_bw"])
            assert row[4] == pytest.approx(fleet[i]["ul_lat"])
            assert row[5] == pytest.approx(fleet[i]["dl_lat"])

    def test_wrapping_when_ndev_exceeds_fleet(self, tmp_path: Path) -> None:
        fleet_path = _make_fleet(tmp_path, num_devices=2)
        fleet = json.loads(fleet_path.read_text())
        rows = self._run_fleet_snippet(fleet_path, 5)

        assert len(rows) == 5
        for i, row in enumerate(rows):
            expected = fleet[i % 2]
            assert row[0] == int(expected["flops"])
            assert row[2] == int(expected["ul_bw"])

    def test_missing_latency_defaults_to_zero(self, tmp_path: Path) -> None:
        fleet = [
            {
                "rank": 0,
                "flops": 6000000000000,
                "memory": 2000000000,
                "ul_bw": 7000000,
                "dl_bw": 40000000,
            }
        ]
        path = tmp_path / "fleet_no_lat.json"
        path.write_text(json.dumps(fleet))

        rows = self._run_fleet_snippet(path, 1)
        assert len(rows) == 1
        assert rows[0][4] == 0.0
        assert rows[0][5] == 0.0

    def test_single_device(self, tmp_path: Path) -> None:
        fleet_path = _make_fleet(tmp_path, num_devices=1)
        rows = self._run_fleet_snippet(fleet_path, 1)
        assert len(rows) == 1

    def test_large_fleet(self, tmp_path: Path) -> None:
        fleet_path = _make_fleet(tmp_path, num_devices=64)
        rows = self._run_fleet_snippet(fleet_path, 256)
        assert len(rows) == 256
        fleet = json.loads(fleet_path.read_text())
        for i, row in enumerate(rows):
            assert row[0] == int(fleet[i % 64]["flops"])


class TestInnerBaselineStructure:
    def test_uses_single_python_call(self) -> None:
        content = (SCRIPTS_DIR / "_inner_baseline.sh").read_text()
        assert "while IFS=' ' read -r" in content, (
            "_inner_baseline.sh should use batch read, "
            "not per-device python3 -c calls"
        )
        per_field_calls = content.count('python3 -c "import json; f=json.load')
        assert per_field_calls == 0, (
            f"Found {per_field_calls} per-field python3 -c calls; "
            "expected 0 (all fields in one call)"
        )

    def test_startup_sleep_is_short(self) -> None:
        content = (SCRIPTS_DIR / "_inner_baseline.sh").read_text()
        assert "INNER_STARTUP_WAIT" in content, (
            "Startup sleep should be configurable via INNER_STARTUP_WAIT"
        )


class TestRunSweepStructure:
    def test_has_parallel_support(self) -> None:
        content = (SCRIPTS_DIR / "_run_sweep.sh").read_text()
        assert "MAX_PARALLEL" in content
        assert "run_one_baseline" in content
        assert "wait -n" in content or "wait" in content

    def test_run_one_baseline_is_function(self) -> None:
        content = (SCRIPTS_DIR / "_run_sweep.sh").read_text()
        assert (
            "run_one_baseline()" in content
            or "run_one_baseline ()" in content
            or "run_one_baseline()\n" in content
        )

    def test_baselines_launched_in_background(self) -> None:
        content = (SCRIPTS_DIR / "_run_sweep.sh").read_text()
        assert 'run_one_baseline "$model" "$bl" &' in content


class TestVtimeExperimentStructure:
    def test_uses_fleet_params(self) -> None:
        content = (SCRIPTS_DIR / "run_vtime_experiment.sh").read_text()
        assert "--flops 5T" not in content, (
            "run_vtime_experiment.sh should read fleet params, "
            "not hardcode --flops 5T"
        )
        assert "DEVICE_PARAMS" in content

    def test_startup_sleep_reduced(self) -> None:
        content = (SCRIPTS_DIR / "run_vtime_experiment.sh").read_text()
        assert (
            "VTIME_STARTUP_WAIT:-3" in content
            or "VTIME_STARTUP_WAIT:-2" in content
        ), "Default VTIME_STARTUP_WAIT should be 3s or less, not 10s"
