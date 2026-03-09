from __future__ import annotations

import pytest

from baselines.scripts.run_asteroid_planner import (
    resolve_profiles_for_required_mps,
)


def test_resolve_profiles_for_required_mps_selects_expected() -> None:
    profiles = {
        0: {
            "layer_profiles_by_mps": {
                "20": {"1": {"latency_ms_mean": 12.0}},
                "80": {"1": {"latency_ms_mean": 44.0}},
            },
            "network": {},
        },
        1: {
            "layer_profiles_by_mps": {
                "20": {"1": {"latency_ms_mean": 18.0}},
                "80": {"1": {"latency_ms_mean": 66.0}},
            },
            "network": {},
        },
    }
    cluster_nodes = {
        "10.0.0.1": {"rank": 0, "active_thread_pct": 20},
        "10.0.0.2": {"rank": 1, "active_thread_pct": 80},
    }

    resolved = resolve_profiles_for_required_mps(
        profiles=profiles,
        cluster_nodes=cluster_nodes,
    )

    assert resolved[0]["layer_profile"]["1"]["latency_ms_mean"] == 12.0
    assert resolved[1]["layer_profile"]["1"]["latency_ms_mean"] == 66.0


def test_resolve_profiles_for_required_mps_missing_required_key_fails() -> None:
    profiles = {
        0: {
            "layer_profiles_by_mps": {
                "20": {"1": {"latency_ms_mean": 12.0}},
            }
        }
    }
    cluster_nodes = {
        "10.0.0.1": {"rank": 0, "active_thread_pct": 40},
    }

    with pytest.raises(RuntimeError) as exc:
        resolve_profiles_for_required_mps(
            profiles=profiles,
            cluster_nodes=cluster_nodes,
        )

    message = str(exc.value)
    assert "rank 0" in message
    assert "layer_profiles_by_mps[40]" in message
    assert "available=['20']" in message


def test_resolve_profiles_for_required_mps_legacy_layer_profile() -> None:
    profiles = {
        0: {
            "layer_profile": {
                "1": {"latency_ms_mean": 15.0},
            }
        }
    }
    cluster_nodes = {
        "10.0.0.1": {"rank": 0, "active_thread_pct": 70},
    }

    resolved = resolve_profiles_for_required_mps(
        profiles=profiles,
        cluster_nodes=cluster_nodes,
    )
    assert resolved[0]["layer_profile"]["1"]["latency_ms_mean"] == 15.0

