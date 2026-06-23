#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportDeprecated=false, reportUnusedCallResult=false
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.python.testutils.determinism import (
    deterministic_context,
    set_all_seeds,
)


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return getattr(outputs, "logits")
    raise TypeError("Model output does not contain logits")


def _extract_loss(outputs: Any) -> Optional[torch.Tensor]:
    if isinstance(outputs, dict) and "loss" in outputs:
        return outputs["loss"]
    if hasattr(outputs, "loss"):
        return getattr(outputs, "loss")
    return None


def _infer_vocab_size(model: torch.nn.Module, tokenizer: Optional[Any]) -> int:
    if tokenizer is not None:
        tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
        if tokenizer_vocab_size is not None:
            return int(tokenizer_vocab_size)
        if hasattr(tokenizer, "__len__"):
            return int(len(tokenizer))

    model_config = getattr(model, "config", None)
    if model_config is not None:
        cfg_vocab_size = getattr(model_config, "vocab_size", None)
        if cfg_vocab_size is not None:
            return int(cfg_vocab_size)

    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "num_embeddings"):
            return int(embeddings.num_embeddings)

    raise ValueError("Unable to infer vocab_size from model/tokenizer")


def _load_real_model_and_tokenizer(
    model_name: str,
) -> Tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _run_once(
    *,
    model_name: str,
    steps: int,
    seed: int,
    batch_size: int,
    seq_length: int,
    device: torch.device,
    model_factory: Optional[Callable[[str], torch.nn.Module]] = None,
    tokenizer_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    set_all_seeds(seed)

    with deterministic_context():
        if model_factory is None:
            model, tokenizer = _load_real_model_and_tokenizer(model_name)
        else:
            model = model_factory(model_name)
            tokenizer = (
                tokenizer_factory(model_name)
                if tokenizer_factory is not None
                else None
            )

        model = model.to(device)
        model.train()

        vocab_size = _infer_vocab_size(model=model, tokenizer=tokenizer)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        losses: List[float] = []
        grad_norms: List[float] = []

        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(seed)
        for _ in range(steps):
            input_ids = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, seq_length),
                dtype=torch.long,
                generator=cpu_gen,
            ).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=input_ids)
            logits = _extract_logits(outputs)
            loss = _extract_loss(outputs)
            if loss is None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    input_ids.view(-1),
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
            losses.append(float(loss.detach().item()))
            grad_norm_value = (
                float(grad_norm.detach().item())
                if hasattr(grad_norm, "detach")
                else float(grad_norm)
            )
            grad_norms.append(grad_norm_value)
            optimizer.step()

        model.eval()
        eval_gen = torch.Generator(device="cpu")
        eval_gen.manual_seed(999)
        eval_input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_length),
            dtype=torch.long,
            generator=eval_gen,
        ).to(device)

        with torch.no_grad():
            eval_outputs = model(input_ids=eval_input_ids)
            logits_sample = _extract_logits(eval_outputs).detach().cpu()

    torch_version_module = getattr(torch, "version", None)
    cuda_version = (
        getattr(torch_version_module, "cuda", None)
        if torch_version_module is not None
        else None
    )

    metadata = {
        "seed": int(seed),
        "model": model_name,
        "steps": int(steps),
        "pytorch_version": torch.__version__,
        "cuda_version": cuda_version or "N/A",
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "logits_sample": logits_sample,
        "metadata": metadata,
    }


def _persist_outputs(output_dir: Path, result: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "per_step_loss.json").write_text(
        json.dumps(result["losses"], indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_step_grad_norm.json").write_text(
        json.dumps(result["grad_norms"], indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(result["logits_sample"], output_dir / "logits_sample.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps(result["metadata"], indent=2) + "\n",
        encoding="utf-8",
    )


def generate_golden_references(
    *,
    model: str = "facebook/opt-125m",
    steps: int = 20,
    seed: int = 42,
    output_dir: str = "tests/golden/opt-125m/",
    batch_size: int = 1,
    seq_length: int = 128,
    verify: bool = False,
    model_factory: Optional[Callable[[str], torch.nn.Module]] = None,
    tokenizer_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)

    first_result = _run_once(
        model_name=model,
        steps=steps,
        seed=seed,
        batch_size=batch_size,
        seq_length=seq_length,
        device=device,
        model_factory=model_factory,
        tokenizer_factory=tokenizer_factory,
    )

    if verify:
        second_result = _run_once(
            model_name=model,
            steps=steps,
            seed=seed,
            batch_size=batch_size,
            seq_length=seq_length,
            device=device,
            model_factory=model_factory,
            tokenizer_factory=tokenizer_factory,
        )
        if not torch.equal(
            first_result["logits_sample"],
            second_result["logits_sample"],
        ):
            raise RuntimeError(
                "Determinism verification failed: logits tensors differ"
            )
        if first_result["losses"] != second_result["losses"]:
            raise RuntimeError(
                "Determinism verification failed: loss trajectories differ"
            )

    _persist_outputs(output_path, first_result)
    return first_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic golden references for training",
    )
    parser.add_argument(
        "--model",
        default="facebook/opt-125m",
        help="HuggingFace model id to use",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of gradient steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for deterministic run",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/golden/opt-125m/",
        help="Directory for generated golden artifacts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for synthetic training inputs",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length for synthetic training inputs",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run two deterministic passes and require bitwise equality",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "CUDA not available; running golden generation on CPU.",
            file=sys.stderr,
        )

    try:
        generate_golden_references(
            model=args.model,
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            verify=args.verify,
        )
    except Exception as exc:
        print(f"Golden reference generation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
