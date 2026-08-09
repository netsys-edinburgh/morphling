from __future__ import annotations

import torch

from morphling.runtime.coordinator_metrics import PhaseRecorder
from scripts._single_coordinator_training_workload import (
    make_tiny_model,
    train_loop_tiny,
)


def test_tiny_training_excludes_warmup_from_measured_phases() -> None:
    # Given
    recorder = PhaseRecorder()
    model = make_tiny_model(seed=42)

    # When
    losses = train_loop_tiny(
        model,
        steps=3,
        warmup_steps=1,
        seed=42,
        device=torch.device("cpu"),
        phase_recorder=recorder,
    )

    # Then
    snapshot = recorder.snapshot()
    assert len(losses) == 4
    assert snapshot.counts["warmup_iteration"] == 1
    assert snapshot.counts["iteration_total"] == 3
    assert snapshot.counts["forward_device_dispatch_aggregation"] == 3
    assert snapshot.counts["backward_device_dispatch_aggregation"] == 3
    assert snapshot.counts["optimizer"] == 3
