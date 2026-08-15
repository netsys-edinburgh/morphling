#!/usr/bin/env python3
"""B1 verifier: recompute + validate the absolute per-batch times in
tab:absolute-times (results/analytical_scaling/absolute_times.json).

Reproduces the analytical cloud-A100 baseline for OPT-13B, Llama2-13B, and
Llama2-70B, recomputes the OPT-13B/256 sys (CLEAVE) time from the planner
manifest with a uniform median-edge fleet, and asserts the matched envelope
(batch=128, seq=1024) so every plotted point is checked against a common model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_batch_runtime import _load_json, compute_batch_runtime

BATCH = 128
SEQ = 1024
TOKENS = BATCH * SEQ
A100_BF16_PEAK = 3.12e14
A100_UTIL = 0.975
MEDIAN_EDGE = {
    "flops": 6e12,
    "ul_bw": 7.5e6,
    "dl_bw": 55e6,
    "ul_lat": 0.0,
    "dl_lat": 0.0,
}

CLOUD_PARAMS = {"opt-13b": 13e9, "llama2-13b": 13e9, "llama2-70b": 70e9}
CLOUD_EXPECTED = {"opt-13b": 33.6, "llama2-13b": 33.6, "llama2-70b": 180.8}

OPT13B_256_MANIFESTS = (
    "results/vtime_scaling/model_size/points/002_opt-13b/planning/manifests"
)


def cloud_seconds(num_params: float) -> float:
    return 6.0 * num_params * TOKENS / (A100_BF16_PEAK * A100_UTIL)


def main() -> int:
    ok = True

    print(
        "== Cloud A100 analytical baseline (6*P*tokens / (312 TFLOPS * 0.975)) =="
    )
    for model, params in CLOUD_PARAMS.items():
        got = cloud_seconds(params)
        exp = CLOUD_EXPECTED[model]
        close = abs(got - exp) <= 0.6
        ok = ok and close
        print(
            f"  {model:11s}: {got:6.1f} s  (paper {exp})  {'OK' if close else 'MISMATCH'}"
        )

    print(
        "\n== sys (CLEAVE) OPT-13B/256, uniform median-edge fleet, bytes/elem=1 =="
    )
    manifest = _load_json(f"{OPT13B_256_MANIFESTS}/cleave_manifest.json")
    meta = manifest.get("metadata", {}).get("model", {})
    profiles = {i: dict(MEDIAN_EDGE) for i in range(256)}
    res = compute_batch_runtime(
        [], manifest, "cleave", device_profiles=profiles, bytes_per_element=1.0
    )
    cleave_s = res.total_runtime_ms / 1000.0
    cleave_close = abs(cleave_s - 37.3) <= 2.0
    ok = ok and cleave_close
    print(
        f"  cleave: {cleave_s:.1f} s  (paper 37.3)  {'OK' if cleave_close else 'MISMATCH'}"
    )

    print("\n== matched envelope ==")
    seq_ok = int(meta.get("seq_length", 0)) == SEQ
    ok = ok and seq_ok
    print(
        f"  OPT-13B/256 manifest: model={meta.get('model_name')} "
        f"seq={meta.get('seq_length')} (expected {SEQ})  {'OK' if seq_ok else 'MISMATCH'}"
    )
    print(f"  batch={BATCH}, tokens={TOKENS} used for the cloud FLOP count")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
