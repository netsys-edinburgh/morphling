from __future__ import annotations

import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_morphling_runtime() -> None:
    if "morphling" not in sys.modules:
        morphling_mod = types.ModuleType("morphling")
        morphling_mod.__path__ = [str(REPO_ROOT / "morphling")]
        morphling_mod.__package__ = "morphling"
        sys.modules["morphling"] = morphling_mod

    if "morphling.runtime" not in sys.modules:
        runtime_mod = types.ModuleType("morphling.runtime")
        runtime_mod.__path__ = [str(REPO_ROOT / "morphling" / "runtime")]
        runtime_mod.__package__ = "morphling.runtime"
        sys.modules["morphling.runtime"] = runtime_mod


def _load_greenctx_symbols() -> tuple[Any, Any, Any]:
    _bootstrap_morphling_runtime()
    green_context = import_module("morphling.runtime.green_context")
    ldpc_adapter = import_module("morphling.runtime.ldpc_trace_adapter")
    return (
        getattr(green_context, "GreenContextConfig"),
        getattr(green_context, "GreenContextController"),
        getattr(ldpc_adapter, "LdpcTraceAdapter"),
    )


def _resolve_trace_path(trace_path: str) -> Path:
    path = Path(trace_path)
    if path.exists():
        return path
    if not path.is_absolute():
        repo_path = REPO_ROOT / path
        if repo_path.exists():
            return repo_path
    raise FileNotFoundError(f"Trace file not found: {trace_path}")
