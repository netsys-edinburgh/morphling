import numpy as np
import pytest

_c = pytest.importorskip(
    "morphling._C",
    reason="morphling core extension is required for EmulationEngine",
)
if not all(
    hasattr(_c, sym)
    for sym in ("ArcherTensorHandle", "MemoryManagerClient", "set_tensor_shm")
):
    pytest.skip(
        "morphling._C is missing ArcherTensorHandle / MemoryManagerClient; "
        "EmulationEngine cannot construct. Tracked in #53.",
        allow_module_level=True,
    )

try:
    from transformers import AutoConfig, OPTForCausalLM

    _TRANSFORMERS_IMPORT_ERROR = None
except (TypeError, ImportError) as e:
    AutoConfig = None
    OPTForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = e

from morphling.common import EmulatorConfig
from morphling.runtime import EmulationEngine


@pytest.mark.skipif(
    _TRANSFORMERS_IMPORT_ERROR is not None,
    reason=(
        "transformers import is incompatible with current environment: "
        f"{_TRANSFORMERS_IMPORT_ERROR}"
    ),
)
@pytest.mark.skipif(
    EmulationEngine is None,
    reason="EmulationEngine unavailable because morphling.runtime initialization failed",
)
def test_param_offload_shapes_are_unit():
    model_name = "facebook/opt-125m"
    config = AutoConfig.from_pretrained(model_name)

    engine = EmulationEngine(config)
    engine_config = EmulatorConfig(
        gpu_memory=0.5, cpu_memory=0.5, ckpt_path="../checkpoints"
    )

    with engine.init(OPTForCausalLM, engine_config):
        model = OPTForCausalLM.from_pretrained(model_name)

    for name, param in model.named_parameters():
        assert np.prod(param.shape) == 1, (
            f"Param {name} has shape {param.shape}"
        )
