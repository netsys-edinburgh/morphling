from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_overlap_env_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    from scripts._multi_coordinator_workload import _overlap_from_env

    # When / Then
    for enabled in ("1", "true", "TRUE", "Yes"):
        monkeypatch.setenv("MORPHLING_OVERLAP_GRADIENT_SYNC", enabled)
        assert _overlap_from_env() is True
    for disabled in ("0", "", "false", "no"):
        monkeypatch.setenv("MORPHLING_OVERLAP_GRADIENT_SYNC", disabled)
        assert _overlap_from_env() is False
    monkeypatch.delenv("MORPHLING_OVERLAP_GRADIENT_SYNC", raising=False)
    assert _overlap_from_env() is False


def _equivalence_worker(rank: int, world_size: int, out_path: str) -> None:
    import torch.distributed as dist
    import torch.nn.functional as F

    from morphling.runtime.coordinator_metrics import PhaseRecorder
    from scripts._multi_coordinator_workload import (
        ModelStep,
        _distributed_losses,
        _native_losses,
        _tiny_inputs,
        _tiny_model,
    )
    from scripts.multi_coordinator_scaling_config import ScalingConfig

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29563")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        config = ScalingConfig(
            mode="strong",
            coordinators=world_size,
            total_devices=world_size,
            global_batch=16,
            devices_per_coordinator=1,
            local_batch=16 // world_size,
            warmup_iterations=1,
            measured_iterations=5,
            tiny=True,
            model_name="tiny-mlp",
            learning_rate=0.01,
        )
        inputs, targets = _tiny_inputs(42, config.global_batch)
        golden = _native_losses(
            config,
            ModelStep(
                _tiny_model(42), lambda m: F.mse_loss(m(inputs), targets)
            ),
        )
        lower = rank * config.local_batch
        local_inputs = inputs[lower : lower + config.local_batch]
        local_targets = targets[lower : lower + config.local_batch]

        def build_step() -> ModelStep:
            return ModelStep(
                _tiny_model(42),
                lambda m: F.mse_loss(m(local_inputs), local_targets),
            )

        recorder = PhaseRecorder()
        blocking = _distributed_losses(config, build_step(), recorder, False)
        overlap = _distributed_losses(config, build_step(), recorder, True)
        if rank == 0:
            Path(out_path).write_text(
                json.dumps(
                    {
                        "golden": list(golden),
                        "blocking": list(blocking),
                        "overlap": list(overlap),
                    }
                ),
                encoding="utf-8",
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_gradient_sync_overlap_matches_blocking_and_golden(
    tmp_path: Path,
) -> None:
    # Given
    torch_mp = pytest.importorskip("torch.multiprocessing")
    out_path = tmp_path / "losses.json"

    # When
    torch_mp.spawn(
        _equivalence_worker, args=(2, str(out_path)), nprocs=2, join=True
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    # Then
    golden, blocking, overlap = (
        payload["golden"],
        payload["blocking"],
        payload["overlap"],
    )
    assert max(abs(a - b) for a, b in zip(overlap, blocking)) < 1e-6
    assert max(abs(a - b) for a, b in zip(overlap, golden)) < 1e-4
    assert max(abs(a - b) for a, b in zip(blocking, golden)) < 1e-4
