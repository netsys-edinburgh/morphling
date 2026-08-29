def test_public_api_surface_is_importable():
    import morphling.api as api

    expected = {
        "set_backend",
        "AutoBackend",
        "apply_hooks",
        "LinearFunction",
        "DeviceConfigArguments",
        "ModelConfigArguments",
        "add_metrics_arguments",
        "metrics_config_from_args",
        "start_metrics_collector",
        "CoordinatorMetricsCollector",
        "PhaseRecorder",
        "track_phase",
        "GreenContextRuntime",
    }
    assert expected <= set(api.__all__)
    for name in expected:
        assert hasattr(api, name), f"missing public symbol: {name}"


def test_greencontextruntime_public_alias_matches_compiled():
    from morphling._GreenCtx import GreenContextRuntime as compiled
    from morphling.runtime import GreenContextRuntime as public

    assert public is compiled
