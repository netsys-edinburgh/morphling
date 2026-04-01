from __future__ import annotations

import csv
import importlib
import importlib.util
import sys
import tempfile
import types
import warnings
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bootstrap_morphling() -> None:
    root = _repo_root()
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
        if spec is not None and spec.loader is not None:
            greenctx = importlib.util.module_from_spec(spec)
            sys.modules["morphling._GreenCtx"] = greenctx
            spec.loader.exec_module(greenctx)


_bootstrap_morphling()

LdpcTraceAdapter = importlib.import_module(
    "morphling.runtime.ldpc_trace_adapter"
).LdpcTraceAdapter


def _resolve_ldpc_csv(name: str) -> Path:
    root = _repo_root()
    primary = root / "data" / name
    if primary.exists():
        return primary
    fallback = root.parent.parent / "DeviceEmulator" / "data" / name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find LDPC trace file: {name}")


def _basic_rows() -> list[dict[str, int]]:
    return [
        {
            "time_slot_sched_ns": 100,
            "sm_count": 8,
            "time_decode_start_actual_ns": 120,
            "profile_idx": 1,
        },
        {
            "time_slot_sched_ns": 200,
            "sm_count": 12,
            "time_decode_start_actual_ns": 235,
            "profile_idx": 2,
        },
        {
            "time_slot_sched_ns": 300,
            "sm_count": 10,
            "time_decode_start_actual_ns": 331,
            "profile_idx": 3,
        },
        {
            "time_slot_sched_ns": 400,
            "sm_count": 10,
            "time_decode_start_actual_ns": 470,
            "profile_idx": 4,
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, int]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_with_ctrl() -> None:
    csv_path = _resolve_ldpc_csv("ldpc_trace_with_ctrl.csv")
    line_count = sum(1 for _ in csv_path.open(encoding="utf-8"))
    assert line_count == 29816
    adapter = LdpcTraceAdapter(csv_path)
    v2 = adapter.to_v2_dataframe()
    assert list(v2.columns) == ["timestamp_ns", "num_sms", "tag"]


def test_parse_without_ctrl() -> None:
    csv_path = _resolve_ldpc_csv("ldpc_trace_without_ctrl.csv")
    line_count = sum(1 for _ in csv_path.open(encoding="utf-8"))
    assert line_count == 27463
    adapter = LdpcTraceAdapter(csv_path)
    assert len(adapter.to_v2_dataframe()) > 0


def test_v2_conversion() -> None:
    rows = _basic_rows()
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "trace.csv"
        _write_csv(csv_path, rows)
        adapter = LdpcTraceAdapter(csv_path, total_sms=48)
        v2 = adapter.to_v2_dataframe()
        expected = [48 - row["sm_count"] for row in rows]
        assert list(v2["num_sms"].astype(int)) == expected


def test_detect_violations() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "trace.csv"
        _write_csv(csv_path, _basic_rows())
        adapter = LdpcTraceAdapter(csv_path)
        violations = adapter.detect_violations()
        assert list(violations["row_idx"].astype(int)) == [1]
        assert (violations["curr_sm"] > violations["prev_sm"]).all()


def test_detect_inefficiencies() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "trace.csv"
        _write_csv(csv_path, _basic_rows())
        adapter = LdpcTraceAdapter(csv_path)
        ineff = adapter.detect_inefficiencies()
        assert list(ineff["row_idx"].astype(int)) == [2]
        assert (ineff["curr_sm"] < ineff["prev_sm"]).all()


def test_switch_gap_calculation() -> None:
    rows = _basic_rows()
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "trace.csv"
        _write_csv(csv_path, rows)
        adapter = LdpcTraceAdapter(csv_path)
        violations = adapter.detect_violations()
        first = violations.iloc[0]
        expected_gap = (
            rows[1]["time_decode_start_actual_ns"]
            - rows[1]["time_slot_sched_ns"]
        )
        assert int(first.switch_gap_ns) == expected_gap


def test_empty_csv() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "empty.csv"
        csv_path.write_text(
            "time_slot_sched_ns,sm_count,time_decode_start_actual_ns\n",
            encoding="utf-8",
        )
        try:
            LdpcTraceAdapter(csv_path)
            raise AssertionError("Expected ValueError for empty CSV")
        except ValueError as err:
            assert "empty" in str(err)


def test_missing_columns() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "missing_cols.csv"
        pd.DataFrame([{"time_slot_sched_ns": 1, "sm_count": 2}]).to_csv(
            csv_path, index=False
        )
        try:
            LdpcTraceAdapter(csv_path)
            raise AssertionError("Expected ValueError for missing columns")
        except ValueError as err:
            msg = str(err)
            assert "missing required columns" in msg
            assert "time_decode_start_actual_ns" in msg


def test_sm_count_clamping() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "clamp.csv"
        _write_csv(
            csv_path,
            [
                {
                    "time_slot_sched_ns": 10,
                    "sm_count": 20,
                    "time_decode_start_actual_ns": 30,
                    "profile_idx": 0,
                }
            ],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter = LdpcTraceAdapter(csv_path, total_sms=16)
        assert any("clamped to total_sms" in str(w.message) for w in caught)
        assert int(adapter.to_v2_dataframe().loc[0, "num_sms"]) == 0


def test_iterator() -> None:
    rows = _basic_rows()
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "iter.csv"
        _write_csv(csv_path, rows)
        adapter = LdpcTraceAdapter(csv_path, total_sms=48)
        out = list(adapter)
        assert len(out) == len(rows)
        assert all(len(item) == 3 for item in out)
        v2 = adapter.to_v2_dataframe()
        expected = [tuple(row) for row in v2.itertuples(index=False, name=None)]
        assert out == expected


def test_to_v2_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        in_csv = Path(td) / "in.csv"
        out_csv = Path(td) / "out_v2.csv"
        _write_csv(in_csv, _basic_rows())
        adapter = LdpcTraceAdapter(in_csv, total_sms=48)
        adapter.to_v2_file(out_csv)
        written = pd.read_csv(out_csv)
        assert list(written.columns) == ["timestamp_ns", "num_sms", "tag"]
        assert len(written.columns) == 3
        with out_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader)
        assert header == ["timestamp_ns", "num_sms", "tag"]
        assert len(first_row) == 3
        assert int(first_row[0]) == int(written.loc[0, "timestamp_ns"])
        assert int(first_row[1]) == int(written.loc[0, "num_sms"])
