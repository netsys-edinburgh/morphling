# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false

import importlib.util
import sys
import types
from pathlib import Path

import pytest


CMDLINE_PATH = (
    Path(__file__).resolve().parents[4]
    / "morphling"
    / "entrypoint"
    / "cmdline.py"
)


def _load_cmdline_module(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.float16 = "float16"
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *args, **kwargs: None

    fake_safetensors = types.ModuleType("safetensors")

    class _SafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def keys(self):
            return []

        def get_tensor(self, _key):
            return None

    fake_safetensors.safe_open = lambda *args, **kwargs: _SafeOpen()

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **kwargs: iterable

    fake_morphling = types.ModuleType("morphling")
    fake_morphling.__path__ = []

    fake_morphling_c = types.ModuleType("morphling._C")

    class _ArcherTensorHandle:
        def __init__(self, *_args, **_kwargs):
            pass

    fake_morphling_c.ArcherTensorHandle = _ArcherTensorHandle

    fake_morphling_utils = types.ModuleType("morphling.utils")
    fake_morphling_utils.get_checkpoint_paths = lambda *_args, **_kwargs: []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setitem(sys.modules, "morphling", fake_morphling)
    monkeypatch.setitem(sys.modules, "morphling._C", fake_morphling_c)
    monkeypatch.setitem(sys.modules, "morphling.utils", fake_morphling_utils)

    spec = importlib.util.spec_from_file_location(
        "cmdline_under_test", CMDLINE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_save_command(monkeypatch):
    module = _load_cmdline_module(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "morphling_cmd",
            "save",
            "--model",
            "facebook/opt-125m",
            "--output",
            "/tmp/ckpt",
            "--dtype",
            "float16",
            "--force",
        ],
    )

    args = module.parse_args()

    assert args.command == "save"
    assert args.model == "facebook/opt-125m"
    assert args.output == "/tmp/ckpt"
    assert args.dtype == "float16"
    assert args.force is True


def test_parse_args_help_text(monkeypatch, capsys):
    module = _load_cmdline_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["morphling_cmd", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        module.parse_args()

    out = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "Morphling Emulator Interface" in out
    assert "save" in out


def test_parse_args_invalid_argument(monkeypatch, capsys):
    module = _load_cmdline_module(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "morphling_cmd",
            "save",
            "--model",
            "facebook/opt-125m",
            "--output",
            "/tmp/ckpt",
            "--bad-flag",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        module.parse_args()

    err = capsys.readouterr().err
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in err
