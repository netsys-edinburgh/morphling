"""Defaults check for the [device_measurement] INI section (#55 step 5b).

The probe gates land in the shipped `config/proxy/svr.ini` and are also
overridable via `MORPHLING_MEASURE_*` env vars. This test verifies the
on-disk defaults (all three enable_* flags off, payload/dim/timeout values
match what `csrc/core/env_cfg.cpp` falls back to) so a stray INI edit can't
silently flip probes on in production.
"""

import configparser
import pathlib


def _load_svr_ini():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    ini_path = repo_root / "config" / "proxy" / "svr.ini"
    assert ini_path.exists(), f"svr.ini missing at {ini_path}"
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    return parser


def test_device_measurement_section_present():
    parser = _load_svr_ini()
    assert parser.has_section("device_measurement"), (
        "[device_measurement] section missing from config/proxy/svr.ini"
    )


def test_default_probe_gates_disabled():
    parser = _load_svr_ini()
    section = parser["device_measurement"]
    assert section.getint("enable_latency") == 0
    assert section.getint("enable_bandwidth") == 0
    assert section.getint("enable_flops") == 0


def test_default_probe_sizes_match_cpp_defaults():
    parser = _load_svr_ini()
    section = parser["device_measurement"]
    assert section.getint("latency_payload_bytes") == 64
    assert section.getint("bandwidth_payload_bytes") == 4 * 1024 * 1024
    assert section.getint("flops_matrix_dim") == 256
    assert float(section["probe_timeout_sec"]) == 5.0
    assert float(section["flops_tolerance"]) == 1e-3
