from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from scripts import deep_numerical_verification as script

pytestmark = pytest.mark.smoke


def test_script_cli_help() -> None:
    completed = subprocess.run(
        ["python", "scripts/deep_numerical_verification.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--model" in completed.stdout
    assert "--steps" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_report_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_pytest(verbose: bool) -> tuple[int, str]:
        assert verbose is False
        return (
            0,
            "================== 12 passed, 3 skipped in 1.23s ==================\n",
        )

    monkeypatch.setattr(script, "_run_pytest", fake_run_pytest)

    exit_code = script.main(["--output-dir", str(tmp_path)])
    assert exit_code == 0

    report_path = tmp_path / "deep_verification_report.json"
    assert report_path.exists()

    report = cast(
        dict[str, object], json.loads(report_path.read_text(encoding="utf-8"))
    )
    required_keys = {
        "overall_result",
        "timestamp",
        "model",
        "steps",
        "test_results",
        "note",
    }
    assert required_keys.issubset(set(report.keys()))
    assert report["overall_result"] == "PASS"
    assert report["note"]
    test_results = cast(dict[str, object], report["test_results"])
    assert set(test_results.keys()) == {
        "total",
        "passed",
        "failed",
        "skipped",
    }
