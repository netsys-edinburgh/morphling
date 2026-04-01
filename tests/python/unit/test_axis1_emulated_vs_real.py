# pyright: reportAny=false, reportDeprecated=false, reportExplicitAny=false
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnusedCallResult=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch

from tests.python.testutils import (
    TIER_STANDARD,
    TIER_STRICT,
    assert_no_nan_inf,
    compare_loss_trajectories,
    compare_tensors,
    deterministic_context,
    set_all_seeds,
)


class _LocalMatmulBackend:
    def __init__(self) -> None:
        self._queue: list[torch.Tensor] = []

    def async_dispatch_matmul(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor
    ) -> None:
        self._queue.append(torch.matmul(mat_a, mat_b.transpose(-2, -1)))

    def wait_matmul(self, _idx: int) -> torch.Tensor:
        if not self._queue:
            raise RuntimeError("wait_matmul called with empty queue")
        return self._queue.pop(0)


@pytest.fixture
def golden_refs() -> dict[str, object]:
    import json
    import pathlib

    golden_dir = pathlib.Path("tests/golden/opt-125m")
    loss_file = golden_dir / "per_step_loss.json"
    grad_file = golden_dir / "per_step_grad_norm.json"
    logits_file = golden_dir / "logits_sample.pt"
    if (
        not logits_file.exists()
        or not loss_file.exists()
        or not grad_file.exists()
    ):
        pytest.skip(
            "Golden references not found — run generate_golden_references.py first"
        )
    return {
        "logits": torch.load(str(logits_file), map_location="cpu"),
        "losses": json.loads(loss_file.read_text()),
        "grad_norms": json.loads(grad_file.read_text()),
    }


@pytest.fixture
def hooked_linear_backend() -> Generator[None, None, None]:
    import morphling
    from morphling.hooks import apply_hooks, set_greenctx
    from morphling.hooks import autograd as hooks_autograd

    orig_linear_forward = torch.nn.Linear.forward
    orig_functional_linear = torch.nn.functional.linear
    orig_tensor_matmul = torch.Tensor.__matmul__
    orig_bmm = torch.bmm
    prev_backend = getattr(hooks_autograd, "_backend", None)
    prev_enable_verification = bool(
        getattr(hooks_autograd, "_enable_verification", False)
    )

    morphling.set_backend(_LocalMatmulBackend())
    apply_hooks("linear")
    setattr(hooks_autograd, "_enable_verification", False)

    try:
        yield
    finally:
        set_greenctx(None, reset_log=True)
        setattr(hooks_autograd, "_backend", prev_backend)
        setattr(
            hooks_autograd, "_enable_verification", prev_enable_verification
        )
        torch.nn.Linear.forward = orig_linear_forward
        torch.nn.functional.linear = orig_functional_linear
        torch.Tensor.__matmul__ = orig_tensor_matmul
        torch.bmm = orig_bmm


def _load_opt_model() -> torch.nn.Module:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        pytest.skip(f"opt-125m not available: {exc}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m", local_files_only=True
        )
        _ = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", local_files_only=True
        )
    except Exception as exc:
        pytest.skip(f"opt-125m not available: {exc}")

    return model


def _run_training_steps(
    model: torch.nn.Module, num_steps: int, vocab_size: int
) -> tuple[list[float], list[float]]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    losses: list[float] = []
    grad_norms: list[float] = []

    for _ in range(num_steps):
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(1, 128),
            device="cuda",
            dtype=torch.long,
        )
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss_value = getattr(outputs, "loss", None)
        if not isinstance(loss_value, torch.Tensor):
            raise AssertionError("Model output does not contain tensor loss")
        loss_value.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
        )
        losses.append(float(loss_value.detach().item()))
        grad_norms.append(float(grad_norm.detach().item()))
        optimizer.step()

    return losses, grad_norms


@pytest.mark.smoke
def test_intentionally_wrong_logits_detected() -> None:
    set_all_seeds(42)
    with deterministic_context():
        _ = torch.manual_seed(42)
        real_logits = torch.randn(1, 10, 50272)
        corrupted = real_logits + 1.0

    result = compare_tensors(corrupted, real_logits, TIER_STRICT)
    assert not result.passed
    assert result.max_abs_diff >= 1.0


@pytest.mark.deep
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_logits_match_golden_reference(
    golden_refs: dict[str, object],
    hooked_linear_backend: None,
) -> None:
    _ = hooked_linear_backend
    set_all_seeds(42)
    with deterministic_context():
        model = _load_opt_model().to("cuda")
        _ = model.eval()

        vocab_size_obj = getattr(
            getattr(model, "config", object()), "vocab_size", None
        )
        if not isinstance(vocab_size_obj, int):
            pytest.skip("opt-125m config.vocab_size unavailable")
        _ = torch.manual_seed(42)
        input_ids = torch.randint(0, vocab_size_obj, (1, 128), device="cuda")
        with torch.no_grad():
            output = model(input_ids=input_ids)
            emulated_logits = getattr(output, "logits", None)
            if not isinstance(emulated_logits, torch.Tensor):
                raise AssertionError(
                    "Model output does not contain tensor logits"
                )
            emulated_logits = emulated_logits.cpu()

    golden_logits = golden_refs.get("logits")
    if not isinstance(golden_logits, torch.Tensor):
        pytest.skip("Golden logits missing or invalid")
    result = compare_tensors(emulated_logits, golden_logits, TIER_STRICT)
    assert result.passed


@pytest.mark.deep
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_loss_trajectory_within_tolerance(
    golden_refs: dict[str, object],
    hooked_linear_backend: None,
) -> None:
    _ = hooked_linear_backend
    set_all_seeds(42)
    with deterministic_context():
        model = _load_opt_model().to("cuda")
        vocab_size_obj = getattr(
            getattr(model, "config", object()), "vocab_size", None
        )
        if not isinstance(vocab_size_obj, int):
            pytest.skip("opt-125m config.vocab_size unavailable")

        golden_losses_obj = golden_refs.get("losses")
        if not isinstance(golden_losses_obj, list):
            pytest.skip("Golden loss trajectory missing or invalid")
        golden_losses = [float(x) for x in golden_losses_obj]
        num_steps = min(20, len(golden_losses))
        emulated_losses, _ = _run_training_steps(
            model, num_steps, vocab_size_obj
        )

    result = compare_loss_trajectories(
        emulated_losses,
        golden_losses[:num_steps],
        TIER_STANDARD,
    )
    assert result.passed


@pytest.mark.deep
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grad_norms_within_range(
    golden_refs: dict[str, object],
    hooked_linear_backend: None,
) -> None:
    _ = hooked_linear_backend
    set_all_seeds(42)
    with deterministic_context():
        model = _load_opt_model().to("cuda")
        vocab_size_obj = getattr(
            getattr(model, "config", object()), "vocab_size", None
        )
        if not isinstance(vocab_size_obj, int):
            pytest.skip("opt-125m config.vocab_size unavailable")

        golden_norms_obj = golden_refs.get("grad_norms")
        if not isinstance(golden_norms_obj, list):
            pytest.skip("Golden grad norms missing or invalid")
        golden_norms = [float(x) for x in golden_norms_obj]
        num_steps = min(20, len(golden_norms))
        _, emulated_norms = _run_training_steps(
            model, num_steps, vocab_size_obj
        )

    for idx, (emulated_norm, golden_norm) in enumerate(
        zip(emulated_norms, golden_norms[:num_steps])
    ):
        assert 0.1 * golden_norm <= emulated_norm <= 10.0 * golden_norm, (
            f"Grad norm out of range at step {idx}: "
            f"emulated={emulated_norm}, golden={golden_norm}"
        )

    assert_no_nan_inf(torch.tensor(emulated_norms), name="grad_norms")
