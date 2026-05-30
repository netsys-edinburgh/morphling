"""LDPC trace adapter for green context trace conversion.

Provides utilities to:
  - Parse LDPC CSV traces with scheduling and decode timing data
  - Convert LDPC-format traces to v2 format (timestamp, SM count, tag)
  - Detect violations and inefficiencies in SM allocation

The adapter validates required columns and converts timestamps to
nanosecond granularity for downstream processing.

Usage:
    from morphling.runtime.ldpc_trace_adapter import LdpcTraceAdapter

    adapter = LdpcTraceAdapter("ldpc_trace.csv", total_sms=48)
    for timestamp_ns, sms, tag in adapter:
        print(f"t={timestamp_ns}, sm={sms}, tag={tag}")
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd


class LdpcTraceAdapter:
    """Adapter for LDPC trace data to v2 format.

    Loads LDPC CSV traces and converts them to the v2 format expected
    by green context backends. Handles SM count clamping and validates
    required columns.

    Attributes:
        csv_path: Path to the input LDPC CSV file.
        total_sms: Total number of SMs available on the GPU.

    Required CSV columns:
        - time_slot_sched_ns: Scheduled time in nanoseconds
        - sm_count: Number of SMs allocated
        - time_decode_start_actual_ns: Actual decode start time
    """

    REQUIRED_COLUMNS: set[str] = {
        "time_slot_sched_ns",
        "sm_count",
        "time_decode_start_actual_ns",
    }

    def __init__(self, csv_path: str | Path, total_sms: int = 48):
        """Initialize adapter with LDPC trace file.

        Args:
            csv_path: Path to the LDPC CSV trace file.
            total_sms: Total number of SMs on the GPU (default: 48).

        Raises:
            ValueError: If total_sms <= 0, CSV is empty, or required columns are missing.
        """
        self.csv_path: Path = Path(csv_path)
        self.total_sms: int = int(total_sms)

        if self.total_sms <= 0:
            raise ValueError("total_sms must be > 0")

        self._df: Any = cast(Any, pd.read_csv(self.csv_path))

        if self._df.empty:
            raise ValueError(
                f"LDPC CSV '{self.csv_path}' is empty; expected at least one row"
            )

        missing = sorted(self.REQUIRED_COLUMNS - set(self._df.columns))
        if missing:
            available = ", ".join(map(str, self._df.columns))
            raise ValueError(
                "LDPC CSV is missing required columns: "
                + f"{', '.join(missing)}. "
                f"Available columns: {available}"
            )

        self._df["time_slot_sched_ns"] = self._to_int_series(
            self._df["time_slot_sched_ns"], "time_slot_sched_ns"
        )
        self._df["sm_count"] = self._to_int_series(
            self._df["sm_count"], "sm_count"
        )
        self._df["time_decode_start_actual_ns"] = self._to_int_series(
            self._df["time_decode_start_actual_ns"],
            "time_decode_start_actual_ns",
        )

        if "profile_idx" in self._df.columns:
            self._df["profile_idx"] = (
                cast(
                    Any, pd.to_numeric(self._df["profile_idx"], errors="coerce")
                )
                .fillna(0)
                .astype(int)
            )
        else:
            self._df["profile_idx"] = 0

        self._df["effective_sm_count"] = self._df["sm_count"].clip(
            lower=0, upper=self.total_sms
        )

        over_total = self._df["sm_count"] > self.total_sms
        if over_total.any():
            warnings.warn(
                "Found sm_count values greater than total_sms; "
                + "clamped to total_sms for v2 conversion",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _to_int_series(series: Any, col_name: str) -> Any:
        numeric = cast(Any, pd.to_numeric(series, errors="coerce"))
        if numeric.isna().any():
            bad_idx = numeric[numeric.isna()].index.tolist()
            raise ValueError(
                f"Column '{col_name}' contains non-numeric values at "
                + f"rows: {bad_idx[:10]}"
            )
        return numeric.astype(int)

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        """Iterate over trace data as (timestamp_ns, num_sms, tag) tuples."""
        v2 = self.to_v2_dataframe()
        timestamps = cast(list[Any], v2["timestamp_ns"].tolist())
        num_sms = cast(list[Any], v2["num_sms"].tolist())
        tags = cast(list[Any], v2["tag"].tolist())
        for timestamp_ns, sms, tag in zip(timestamps, num_sms, tags):
            yield int(timestamp_ns), int(sms), int(tag)

    def to_v2_dataframe(self) -> pd.DataFrame:
        """Convert LDPC trace to v2 DataFrame format.

        Returns:
            DataFrame with columns: timestamp_ns, num_sms, tag.
        """
        v2_df = pd.DataFrame(
            {
                "timestamp_ns": self._df["time_slot_sched_ns"].astype(int),
                "num_sms": (
                    self.total_sms - self._df["effective_sm_count"]
                ).astype(int),
                "tag": self._df["profile_idx"].astype(int),
            }
        )
        return cast(pd.DataFrame, v2_df)

    def to_v2_file(self, path: str | Path) -> None:
        """Write v2 trace to CSV file.

        Args:
            path: Output path for the v2 CSV file.
        """
        out_path = Path(path)
        self.to_v2_dataframe().to_csv(out_path, index=False)

    def detect_violations(self) -> pd.DataFrame:
        """Detect SM count increases (potential violations).

        Returns:
            DataFrame rows where curr_sm > prev_sm, with columns: row_idx,
            prev_sm, curr_sm, switch_gap_ns.
        """
        return self._detect_deltas(increase=True)

    def detect_inefficiencies(self) -> pd.DataFrame:
        """Detect SM count decreases (potential inefficiencies).

        Returns:
            DataFrame rows where curr_sm < prev_sm, with columns: row_idx,
            prev_sm, curr_sm, switch_gap_ns.
        """
        return self._detect_deltas(increase=False)

    def _detect_deltas(self, increase: bool) -> pd.DataFrame:
        cols = ["row_idx", "prev_sm", "curr_sm", "switch_gap_ns"]
        if len(self._df) < 2:
            return pd.DataFrame(columns=pd.Index(cols))

        curr_sm = self._df["sm_count"]
        prev_sm = curr_sm.shift(1)

        if increase:
            mask = curr_sm > prev_sm
        else:
            mask = curr_sm < prev_sm

        hits = self._df.loc[mask].copy()
        if hits.empty:
            return pd.DataFrame(columns=pd.Index(cols))

        hits["row_idx"] = hits.index.astype(int)
        hits["prev_sm"] = prev_sm.loc[hits.index].astype(int)
        hits["curr_sm"] = hits["sm_count"].astype(int)
        hits["switch_gap_ns"] = (
            hits["time_decode_start_actual_ns"] - hits["time_slot_sched_ns"]
        ).astype(int)

        return hits[cols].reset_index(drop=True)
