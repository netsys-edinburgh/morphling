#!/usr/bin/env python3
"""D1: Compute per-method, per-component correction factors.

Compares vtime emulation results (paper_summary.json) against
analytical model predictions (device_scaling/opt-125m/n64/summary.json)
to derive multiplicative correction factors for upscaling.

Confident is calibrated at 12 devices (its vtime baseline).
All other methods are calibrated at 64 devices.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VTIME_PATH = REPO / "results/vtime/summary/paper_summary.json"
ANALYTICAL_PATH = REPO / "results/device_scaling/opt-125m/n64/summary.json"
OUTPUT_DIR = REPO / "results/analytical_scaling"
OUTPUT_PATH = OUTPUT_DIR / "calibration.json"

METHODS = ["cleave", "dtfm", "asteroid", "confident", "alpa"]

# Field mapping: vtime key -> analytical key
COMPONENT_MAP = {
    "total_ms": "total_runtime_ms",
    "compute_ms": "compute_time_ms",
    "network_ms": "network_time_ms",
    "allreduce_ms": "allreduce_time_ms",
    "bubble_ms": "pipeline_bubble_ms",
}

# Short names used in correction_factors output
COMPONENT_NAMES = ["total", "compute", "network", "allreduce", "bubble"]


def _load_vtime():
    with open(VTIME_PATH) as f:
        data = json.load(f)
    return data["results"]


def _load_analytical():
    with open(ANALYTICAL_PATH) as f:
        data = json.load(f)
    return data["baselines"]


def _correction(vtime_val, analytical_val):
    """Compute correction factor. Returns 1.0 when analytical is zero."""
    if analytical_val == 0 or analytical_val is None:
        return 1.0
    return vtime_val / analytical_val


# Per-method straggler sensitivity exponent.
# CF_target = CF_mild × (strag_target / strag_mild) ^ exponent
# DP methods: exponent=1.0 (full straggler penalty at sync barrier)
# Cleave: exponent=0.3 (solver mitigates but doesn't eliminate)
# Confident: uses compute straggler, not bandwidth
STRAGGLER_SENSITIVITY = {
    "cleave": ("dl", 0.3),
    "dtfm": ("dl", 1.0),
    "asteroid": ("dl", 1.0),
    "confident": ("compute", 1.0),
    "alpa": ("dl", 1.0),
}


def _straggler_sensitivity(method):
    dim, exp = STRAGGLER_SENSITIVITY[method]
    return {"dimension": dim, "exponent": exp}


def calibrate():
    vtime = _load_vtime()
    analytical = _load_analytical()

    per_method = {}
    for method in METHODS:
        vt = vtime[method]
        an = analytical[method]

        cal_n = vt["n_devs"]  # 64 for most, 12 for confident

        vtime_components = {
            "total_ms": vt["total_ms"],
            "compute_ms": vt["compute_ms"],
            "network_ms": vt["network_ms"],
            "allreduce_ms": vt["allreduce_ms"],
            "bubble_ms": vt["bubble_ms"],
        }

        analytical_components = {
            "total_ms": an["total_runtime_ms"],
            "compute_ms": an["compute_time_ms"],
            "network_ms": an["network_time_ms"],
            "allreduce_ms": an["allreduce_time_ms"],
            "bubble_ms": an["pipeline_bubble_ms"],
        }

        corrections = {}
        for short_name, vt_key in zip(
            COMPONENT_NAMES,
            [
                "total_ms",
                "compute_ms",
                "network_ms",
                "allreduce_ms",
                "bubble_ms",
            ],
        ):
            corrections[short_name] = _correction(
                vtime_components[vt_key],
                analytical_components[vt_key],
            )

        per_method[method] = {
            "calibration_n_devs": cal_n,
            "vtime": vtime_components,
            "analytical": analytical_components,
            "correction_factors": corrections,
            "straggler_sensitivity": _straggler_sensitivity(method),
        }

    result = {
        "calibration_point": {"model": "opt-125m"},
        "per_method": per_method,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Calibration written to {OUTPUT_PATH}")
    for m, d in per_method.items():
        n = d["calibration_n_devs"]
        cf = d["correction_factors"]
        print(
            f"  {m:>10s} (n={n:>3d}): "
            f"total={cf['total']:8.2f}x  "
            f"compute={cf['compute']:8.2f}x  "
            f"network={cf['network']:8.2f}x  "
            f"allreduce={cf['allreduce']:8.2f}x  "
            f"bubble={cf['bubble']:8.2f}x"
        )


if __name__ == "__main__":
    calibrate()
