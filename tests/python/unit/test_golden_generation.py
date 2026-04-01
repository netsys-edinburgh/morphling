from __future__ import annotations
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

from scripts.generate_golden_references import generate_golden_references


class TinyTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        hidden = torch.relu(self.hidden(hidden))
        logits = self.projection(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)


def _run_mock_generation(output_dir: Path, steps: int = 3) -> None:
    generate_golden_references(
        model="tiny-mock",
        steps=steps,
        seed=42,
        output_dir=str(output_dir),
        batch_size=2,
        seq_length=8,
        verify=True,
        model_factory=lambda _: TinyCausalLM(vocab_size=32, hidden_size=16),
        tokenizer_factory=lambda _: TinyTokenizer(vocab_size=32),
    )


def test_script_cli_help() -> None:
    completed = subprocess.run(
        ["python", "scripts/generate_golden_references.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--model" in completed.stdout


def test_mock_output_schema(tmp_path: Path) -> None:
    steps = 4
    _run_mock_generation(tmp_path, steps=steps)

    loss_path = tmp_path / "per_step_loss.json"
    grad_path = tmp_path / "per_step_grad_norm.json"
    logits_path = tmp_path / "logits_sample.pt"
    metadata_path = tmp_path / "metadata.json"

    assert loss_path.exists()
    assert grad_path.exists()
    assert logits_path.exists()
    assert metadata_path.exists()

    losses = json.loads(loss_path.read_text(encoding="utf-8"))
    assert len(losses) == steps
    assert all(isinstance(item, float) for item in losses)


def test_metadata_has_required_fields(tmp_path: Path) -> None:
    _run_mock_generation(tmp_path, steps=2)

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "seed",
        "model",
        "steps",
        "pytorch_version",
        "cuda_version",
        "gpu_name",
    }
    assert required.issubset(set(metadata.keys()))
