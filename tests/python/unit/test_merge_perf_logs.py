"""Regression tests for the perf-log pipeline documented in
docs/GEMM_ID_ISSUES.md.

That doc captured three historical issues:

  1. ``gemm_id`` always 0  -> gemm_id is a real, position-3/4 field that the
     merge pipeline carries through and that takes distinct, increasing values.
  2. Log files missing ``#`` header comments -> headers must survive a merge.
  3. ``merge_perf_logs.py`` syntax error -> the module must import and run.

The synthetic rows below mirror the exact C++ emit format in
``csrc/backend/device_tracker.cpp``:

  VTIME:      ``snprintf(... "VTIME,%lu,%ld,%ld,%s,%s,%lu,%lu,%lu\\n" ...)``
  Throughput: ``snprintf(... "%lu,%ld,%ld,%s,%lu,%.2f,%lu,%lu,%lu\\n" ...)``

and the ``# VTIME format:`` / ``# Throughput format:`` headers written by
``DevicePartitionTracker::InitSeparatePerfLog``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import merge_perf_logs as script

pytestmark = pytest.mark.smoke

# Exact header lines emitted by InitSeparatePerfLog (device_tracker.cpp).
VTIME_HEADER = (
    "# VTIME format: VTIME,timestamp_us,device_id,gemm_id,phase,event,"
    "vt_start_us,vt_end_us,vt_duration_us"
)
THROUGHPUT_HEADER = (
    "# Throughput format: timestamp_us,device_id,gemm_id,direction,bytes,"
    "throughput_b_s,epoch_start_us,epoch_end_us,packet_duration_us"
)


def _server_log(path: Path) -> None:
    """A server log: headers + VTIME rows with gemm_id (field 4) = 0,1,2."""
    lines = [
        VTIME_HEADER,
        THROUGHPUT_HEADER,
        "# Separate performance log for server",
        # VTIME,timestamp_us,device_id,gemm_id,phase,event,vts,vte,vtd
        "VTIME,1000,0,0,COMPUTE,START,10,10,0",
        "VTIME,1003,0,0,COMPUTE,END,10,20,10",
        "VTIME,1005,0,1,COMPUTE,START,20,20,0",
        "VTIME,1009,0,1,COMPUTE,END,20,35,15",
        "VTIME,1011,0,2,COMPUTE,START,35,35,0",
        "VTIME,1014,0,2,COMPUTE,END,35,55,20",
    ]
    path.write_text("\n".join(lines) + "\n")


def _device_log(path: Path) -> None:
    """A device log: headers + throughput rows with gemm_id (field 3) = 0,1,2."""
    lines = [
        VTIME_HEADER,
        THROUGHPUT_HEADER,
        "# Separate performance log for device 0",
        # timestamp_us,device_id,gemm_id,direction,bytes,tp,es,ee,dur
        "1001,0,0,DOWNLOAD,131072,70022.96,1000,1000,0",
        "1004,0,1,DOWNLOAD,131072,80000.00,1000,1100,100",
        "1012,0,2,DOWNLOAD,131072,90000.00,1100,1250,150",
    ]
    path.write_text("\n".join(lines) + "\n")


def _build_logs(log_dir: Path) -> None:
    _server_log(log_dir / "perf_server.log")
    _device_log(log_dir / "perf_device_0.log")


# --- Claim 3: the module imports and the merge runs without error. ----------


def test_merge_runs_and_writes_output(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _build_logs(log_dir)
    out = tmp_path / "perf_merged.log"

    ok = script.merge_logs(str(log_dir), str(out))

    assert ok is True
    assert out.exists()
    assert out.read_text().strip() != ""


# --- Claim 2: header comments survive the merge. ----------------------------


def test_headers_preserved_in_merged_output(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _build_logs(log_dir)
    out = tmp_path / "perf_merged.log"

    assert script.merge_logs(str(log_dir), str(out)) is True
    merged = out.read_text().splitlines()

    assert VTIME_HEADER in merged
    assert THROUGHPUT_HEADER in merged


# --- Claim 1: gemm_id is a real field carrying distinct, increasing values. --


def test_vtime_gemm_id_is_field_4_and_increments(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _build_logs(log_dir)
    out = tmp_path / "perf_merged.log"

    assert script.merge_logs(str(log_dir), str(out)) is True
    merged = out.read_text().splitlines()

    vtime_rows = [ln for ln in merged if ln.startswith("VTIME,")]
    # gemm_id is the 4th CSV field (index 3) per the documented schema.
    gemm_ids = sorted({int(ln.split(",")[3]) for ln in vtime_rows})
    assert gemm_ids == [0, 1, 2], "gemm_id must increment, not be stuck at 0"


def test_throughput_gemm_id_is_field_3_and_increments(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _build_logs(log_dir)
    out = tmp_path / "perf_merged.log"

    assert script.merge_logs(str(log_dir), str(out)) is True
    merged = out.read_text().splitlines()

    tput_rows = [
        ln for ln in merged if ln and ln[0].isdigit() and "DOWNLOAD" in ln
    ]
    # gemm_id is the 3rd CSV field (index 2) in the throughput schema.
    gemm_ids = sorted({int(ln.split(",")[2]) for ln in tput_rows})
    assert gemm_ids == [0, 1, 2], "gemm_id must increment, not be stuck at 0"


# --- Bonus: events end up sorted by timestamp across both files. ------------


def test_events_sorted_by_timestamp(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _build_logs(log_dir)
    out = tmp_path / "perf_merged.log"

    assert script.merge_logs(str(log_dir), str(out)) is True
    merged = out.read_text().splitlines()

    def _ts(line: str) -> int:
        if line.startswith("VTIME,"):
            return int(line.split(",")[1])
        return int(line.split(",")[0])

    data_rows = [
        ln
        for ln in merged
        if ln.startswith("VTIME,") or (ln and ln[0].isdigit())
    ]
    timestamps = [_ts(ln) for ln in data_rows]
    assert timestamps == sorted(timestamps)
