import ctypes
import sys

import pytest

_c = pytest.importorskip("morphling._C")
if not all(
    hasattr(_c, sym)
    for sym in ("ArcherTensorHandle", "MemoryManagerClient", "set_tensor_shm")
):
    pytest.skip(
        "morphling._C is missing ArcherTensorHandle / MemoryManagerClient; "
        "InitEmptyModel cannot construct. Tracked in #53.",
        allow_module_level=True,
    )

import torch  # noqa: E402

from morphling.runtime import InitEmptyModel  # noqa: E402

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BertModel,
        OPTForCausalLM,
    )

    _TRANSFORMERS_IMPORT_ERROR = None
except (TypeError, ImportError) as e:
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BertModel = None
    OPTForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = e


def get_loaded_so_libs(module):
    loaded_libs = set()

    module_path = module.__file__
    print(f"Module path: {module_path}")

    for _, mod in sys.modules.items():
        if mod and hasattr(mod, "__file__") and mod.__file__:
            lib_path = mod.__file__
            if lib_path.endswith(".so"):
                loaded_libs.add(lib_path)

    return loaded_libs


@pytest.mark.skipif(
    _TRANSFORMERS_IMPORT_ERROR is not None,
    reason=(
        "transformers incompatible with current Python/runtime: "
        f"{_TRANSFORMERS_IMPORT_ERROR}"
    ),
)
@pytest.mark.skipif(
    InitEmptyModel is None,
    reason="morphling runtime InitEmptyModel is unavailable in this environment",
)
def test_loaded_lib_forward_matches_original():
    assert ctypes is not None

    model_name = "facebook/opt-125m"
    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Model name: {model_name}")
    assert model_config is not None
    assert AutoModelForCausalLM is not None
    assert BertModel is not None

    initializer = InitEmptyModel(OPTForCausalLM)
    with initializer:
        model = OPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        model.eval()

    print("Emulator model loaded", flush=True)

    original_model = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    original_model.eval()
    print("Original model loaded", flush=True)

    input_text = "Hello, my dog is cute."
    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model(**input_ids, return_dict=True)
    print("Emulator forward pass done", flush=True)

    original_outputs = original_model(**input_ids, return_dict=True)
    print("Original forward pass done", flush=True)

    assert torch.allclose(outputs.logits, original_outputs.logits, atol=1e-4)

    _ = get_loaded_so_libs(torch)
