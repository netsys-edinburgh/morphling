#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false, reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportOptionalMemberAccess=false, reportIndexIssue=false
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportUnusedCallResult=false

"""Validate LDPC trace CSV files for paper experiments."""

import argparse
import sys

import numpy as np
import pandas as pd

VALID_SM_COUNTS = {8, 16, 24}


def validate_trace(path: str) -> list[str]:
    errors = []
    df = pd.read_csv(path)

    required = [
        "time_slot_sched_ns",
        "sm_count",
        "time_decode_start_actual_ns",
    ]
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return errors

    invalid = set(df["sm_count"].unique()) - VALID_SM_COUNTS
    if invalid:
        errors.append(f"Invalid sm_count values: {invalid}")

    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values")

    ts = df["time_slot_sched_ns"].to_numpy(dtype=np.int64)
    if not np.all(ts[1:] >= ts[:-1]):
        errors.append("time_slot_sched_ns is not monotonically increasing")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory with trace CSVs",
    )
    args = parser.parse_args()

    traces = {
        "with_ctrl": f"{args.data_dir}/ldpc_trace_with_ctrl.csv",
        "without_ctrl": f"{args.data_dir}/ldpc_trace_without_ctrl.csv",
    }

    all_ok = True
    for name, path in traces.items():
        print(f"\n=== Validating {name}: {path} ===")
        try:
            df = pd.read_csv(path)
            print(f"  Rows: {len(df)}")
            print(f"  SM counts: {sorted(df['sm_count'].unique())}")
            print(
                f"  SM distribution: {df['sm_count'].value_counts().to_dict()}"
            )
            transitions = int((df["sm_count"].diff() != 0).sum() - 1)
            print(f"  SM transitions: {transitions}")

            errors = validate_trace(path)
            if errors:
                for err in errors:
                    print(f"  ERROR: {err}")
                all_ok = False
            else:
                print("  PASSED")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            all_ok = False

    try:
        df_with = pd.read_csv(traces["with_ctrl"])
        df_without = pd.read_csv(traces["without_ctrl"])
        shorter = min(len(df_with), len(df_without))
        print("\n=== Length Mismatch ===")
        print(f"  with_ctrl: {len(df_with)} rows")
        print(f"  without_ctrl: {len(df_without)} rows")
        print(f"  Truncation strategy: use {shorter} rows for both runs")
    except Exception:
        pass

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
