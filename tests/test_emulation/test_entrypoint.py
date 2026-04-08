import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestCLISleepMode:
    def test_runs_successfully(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "morphling.entrypoint.emulate",
                "--num-devices",
                "2",
                "--num-steps",
                "3",
                "--alpha",
                "1.0,2.0",
                "--beta",
                "1.0,1.0",
                "--compute-fn",
                "sleep",
                "--comm-fn",
                "loopback",
                "--sleep-compute-s",
                "0.005",
                "--sleep-comm-s",
                "0.002",
                "--output-json",
                json_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(Path(json_path).read_text())
        assert "emulation" in data
        assert "mean_step_time_s" in data["emulation"]
        assert data["emulation"]["mean_step_time_s"] > 0


class TestCLIMismatchedAlpha:
    def test_errors_on_mismatch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "morphling.entrypoint.emulate",
                "--num-devices",
                "3",
                "--alpha",
                "1.0,2.0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert (
            "alpha" in result.stderr.lower()
            or "device" in result.stderr.lower()
        )
