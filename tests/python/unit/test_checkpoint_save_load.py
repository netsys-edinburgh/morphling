# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportAny=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportUnusedParameter=false

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
import uuid
from unittest.mock import Mock, mock_open

import pytest


@dataclass
class _FakeStorage:
    ptr: int
    storage_size: int

    def data_ptr(self):
        return self.ptr

    def size(self):
        return self.storage_size


class _FakeTensor:
    def __init__(self, shape, *, dtype="torch.float32", stride=None):
        self.shape = tuple(shape)
        self._stride = tuple(stride or _contiguous_stride(shape))
        self.dtype = dtype
        storage_size = 1
        for dim in self.shape:
            storage_size *= dim
        self._storage = _FakeStorage(ptr=id(self), storage_size=storage_size)

    def stride(self):
        return self._stride

    def untyped_storage(self):
        return self._storage


def _contiguous_stride(shape):
    if not shape:
        return ()
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return tuple(stride)


def _fake_tensor(shape):
    return _FakeTensor(shape)


def _install_dependency_stubs(monkeypatch):
    def _noop(*args, **kwargs):
        return None

    accelerate = types.ModuleType("accelerate")

    class _DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    accelerate.dispatch_model = _noop
    accelerate.init_empty_weights = lambda *a, **k: _DummyContext()

    accelerate_utils = types.ModuleType("accelerate.utils")
    accelerate_utils.set_module_tensor_to_device = _noop

    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch_nn = types.ModuleType("torch.nn")

    class _Module:
        pass

    torch_nn.Module = _Module
    torch.nn = torch_nn

    sllm_pkg = types.ModuleType("serverless_llm_store")

    sllm_c = types.ModuleType("serverless_llm_store._C")
    sllm_c.allocate_cuda_memory = _noop
    sllm_c.get_cuda_memory_handles = _noop
    sllm_c.get_device_uuid_map = _noop
    sllm_c.restore_tensors = _noop
    sllm_c.save_tensors = lambda *a, **k: {}

    sllm_client = types.ModuleType("serverless_llm_store.client")

    class _SllmStoreClient:
        pass

    sllm_client.SllmStoreClient = _SllmStoreClient

    sllm_device_map = types.ModuleType("serverless_llm_store.device_map_utils")
    sllm_device_map.DeviceMapType = dict
    sllm_device_map._compute_device_placement_from_map = _noop
    sllm_device_map._compute_device_placement_from_map_fast = _noop
    sllm_device_map._expand_tensor_name = _noop
    sllm_device_map._transform_device_map_to_dict = _noop

    sllm_logger = types.ModuleType("serverless_llm_store.logger")
    sllm_logger.init_logger = lambda *_: types.SimpleNamespace(warning=_noop)

    sllm_utils = types.ModuleType("serverless_llm_store.utils")
    sllm_utils.calculate_device_memory = _noop
    sllm_utils.calculate_tensor_device_offsets = _noop
    sllm_utils.dtype_byte_size = _noop
    sllm_utils.get_no_split_modules = lambda *_: []
    sllm_utils.get_tied_no_split_modules = lambda *_: []
    sllm_utils.send_module_buffers_to_device = _noop

    transformers = types.ModuleType("transformers")

    class _GenerationConfig:
        _from_model_config = False

        @classmethod
        def from_model_config(cls, _):
            return cls()

        def save_pretrained(self, _):
            return None

        def __eq__(self, other):
            return isinstance(other, _GenerationConfig)

    class _AutoConfig:
        pass

    class _AutoModelForCausalLM:
        pass

    transformers.AutoConfig = _AutoConfig
    transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    transformers.GenerationConfig = _GenerationConfig

    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)
    monkeypatch.setitem(sys.modules, "serverless_llm_store", sllm_pkg)
    monkeypatch.setitem(sys.modules, "serverless_llm_store._C", sllm_c)
    monkeypatch.setitem(sys.modules, "serverless_llm_store.client", sllm_client)
    monkeypatch.setitem(
        sys.modules,
        "serverless_llm_store.device_map_utils",
        sllm_device_map,
    )
    monkeypatch.setitem(sys.modules, "serverless_llm_store.logger", sllm_logger)
    monkeypatch.setitem(sys.modules, "serverless_llm_store.utils", sllm_utils)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


@pytest.fixture
def save_and_load_module(monkeypatch):
    _install_dependency_stubs(monkeypatch)
    module_path = (
        Path(__file__).resolve().parents[3]
        / "morphling"
        / "checkpoint"
        / "save_and_load.py"
    )
    module_name = f"_save_and_load_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_save_dict_writes_tensor_index_with_mock_tensors(
    tmp_path, save_and_load_module, monkeypatch
):
    module = save_and_load_module
    captured = {}

    def _fake_save_tensors(tensor_names, tensor_data_index, model_path):
        captured["tensor_names"] = tensor_names
        captured["tensor_data_index"] = tensor_data_index
        captured["model_path"] = model_path
        return {name: (idx + 1) * 128 for idx, name in enumerate(tensor_names)}

    monkeypatch.setattr(module, "save_tensors", _fake_save_tensors)

    state_dict = {
        "layer.weight": _fake_tensor((2, 3)),
        "layer.bias": _fake_tensor((3,)),
    }
    output_dir = tmp_path / "mock_ckpt"

    module.save_dict(state_dict, str(output_dir))

    assert captured["tensor_names"] == ["layer.weight", "layer.bias"]
    assert set(captured["tensor_data_index"]) == set(state_dict)

    index_data = json.loads((output_dir / "tensor_index.json").read_text())
    assert index_data["layer.weight"][0] == 128
    assert index_data["layer.bias"][0] == 256
    assert index_data["layer.weight"][2] == [2, 3]
    assert index_data["layer.weight"][4] == "torch.float32"


def test_save_dict_creates_directory_when_path_missing(
    save_and_load_module, monkeypatch
):
    module = save_and_load_module
    model_path = "/virtual/checkpoint"
    makedirs = Mock()

    monkeypatch.setattr(module.os.path, "exists", lambda _: False)
    monkeypatch.setattr(module.os, "makedirs", makedirs)
    monkeypatch.setattr(module, "save_tensors", lambda *_, **__: {"x": 0})
    monkeypatch.setattr("builtins.open", mock_open())

    module.save_dict({"x": _fake_tensor((1,))}, model_path)

    makedirs.assert_called_once_with(model_path, exist_ok=True)


def test_save_dict_raises_for_malformed_tensor_offsets(
    tmp_path, save_and_load_module, monkeypatch
):
    module = save_and_load_module
    monkeypatch.setattr(module, "save_tensors", lambda *_, **__: {})

    with pytest.raises(KeyError):
        module.save_dict({"x": _fake_tensor((1,))}, str(tmp_path / "bad"))


def test_save_dict_propagates_file_not_found_on_index_write(
    tmp_path, save_and_load_module, monkeypatch
):
    module = save_and_load_module
    monkeypatch.setattr(module, "save_tensors", lambda *_, **__: {"x": 0})

    def _raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError("tensor_index.json missing")

    monkeypatch.setattr("builtins.open", _raise_file_not_found)

    with pytest.raises(FileNotFoundError):
        module.save_dict(
            {"x": _fake_tensor((1,))},
            str(tmp_path / "missing_target"),
        )
