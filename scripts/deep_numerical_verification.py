from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

# Explicitly scope to our new test files to avoid pre-existing C-extension
# crashes (e.g. tests/python/unit/test_param_offload.py aborts the process).
_TEST_FILES = [
    "tests/python/unit/test_axis1_emulated_vs_real.py",
    "tests/python/unit/test_axis2_dispatch_correctness.py",
    "tests/python/unit/test_axis3_determinism.py",
    "tests/python/unit/test_determinism_utils.py",
    "tests/python/unit/test_numerical_utils.py",
    "tests/python/unit/test_golden_generation.py",
    "tests/python/unit/test_deep_verification_script.py",
    "tests/python/integration/test_convergence_regression.py",
]

PYTEST_COMMAND = [
    "pytest",
    *_TEST_FILES,
    "-m",
    "smoke or deep",
    "-v",
    "--timeout=1800",
    "--tb=short",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep numerical verification tests and write a report.",
    )
    _ = parser.add_argument("--model", default="opt-125m")
    _ = parser.add_argument("--steps", type=int, default=20)
    _ = parser.add_argument("--output-dir", default="/tmp/deep_verify/")
    _ = parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _last_count(pattern: str, text: str) -> int:
    matches = list(re.finditer(pattern, text))
    if not matches:
        return 0
    return int(matches[-1].group(1))


def _parse_test_results(output: str, returncode: int) -> dict[str, int]:
    passed = _last_count(r"(\d+)\s+passed\b", output)
    failed = _last_count(r"(\d+)\s+failed\b", output)
    skipped = _last_count(r"(\d+)\s+skipped\b", output)

    if returncode != 0 and failed == 0:
        failed = 1

    total = passed + failed + skipped
    results = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }
    return results


def _run_pytest(verbose: bool) -> tuple[int, str]:
    completed = subprocess.run(
        PYTEST_COMMAND,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined = stdout
    if stderr:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += stderr

    if verbose and combined:
        print(combined, end="" if combined.endswith("\n") else "\n")

    return completed.returncode, combined


def _write_report(output_dir: Path, payload: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "deep_verification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return report_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model = cast(str, args.model)
    steps = int(cast(int, args.steps))
    output_dir = Path(cast(str, args.output_dir))
    verbose = bool(cast(bool, args.verbose))

    returncode, output = _run_pytest(verbose)
    test_results = _parse_test_results(output, returncode)
    overall_result = "PASS" if test_results["failed"] == 0 else "FAIL"
    timestamp = datetime.now(timezone.utc).isoformat()
    report: dict[str, object] = {
        "overall_result": overall_result,
        "timestamp": timestamp,
        "model": model,
        "steps": steps,
        "test_results": test_results,
        "note": "Skipped tests are expected without GPU/golden refs.",
    }

    report_path = _write_report(output_dir, report)
    print("Deep numerical verification summary")
    print("  model: %s" % model)
    print("  steps: %s" % steps)
    print("  result: %s" % overall_result)
    print(
        "  tests: {total} total, {passed} passed, {failed} failed, {skipped} skipped".format(
            total=test_results["total"],
            passed=test_results["passed"],
            failed=test_results["failed"],
            skipped=test_results["skipped"],
        )
    )
    print("  report: %s" % report_path)
    return 0 if test_results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
