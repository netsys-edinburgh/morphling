"""YAML config loader that maps to BaseConfig dataclasses."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar, cast

from baselines.core.config import (
    BaseConfig,
    ClusterConfig,
    DeviceConfig,
    DistributedConfig,
    FaultToleranceConfig,
    GreenCtxConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)
_DataclassT = TypeVar("_DataclassT")


def load_config(path: str | Path) -> BaseConfig:
    """Load a YAML config file and return a BaseConfig."""
    import yaml

    path = Path(path)
    with open(path, encoding="utf-8") as f:
        loaded_obj = cast(object, yaml.safe_load(f))
    raw: dict[str, object] = {}
    if isinstance(loaded_obj, dict):
        for key, value in cast(dict[object, object], loaded_obj).items():
            if isinstance(key, str):
                raw[key] = value
    return _parse_config(raw)


def _parse_config(raw: dict[str, object]) -> BaseConfig:
    """Parse raw YAML dict into BaseConfig."""
    # Accept both "parallelism" (YAML convention) and
    # "parallel" (dataclass attribute name from asdict).
    parallel_raw = raw.get("parallelism", raw.get("parallel", {}))
    # NOTE: Cluster config does NOT include bandwidth data.
    # All bandwidth must come from iperf3 profiling. NO FALLBACKS.
    model_cfg = _build_dataclass(ModelConfig, raw.get("model", {}))
    training_cfg = _build_dataclass(TrainingConfig, raw.get("training", {}))

    # Sync micro_batch_size into ModelConfig so strategies
    # and schedulers can access it without a separate
    # TrainingConfig reference.
    if model_cfg.micro_batch_size <= 1 and training_cfg.micro_batch_size > 1:
        model_cfg.micro_batch_size = training_cfg.micro_batch_size

    return BaseConfig(
        device=_build_dataclass(DeviceConfig, raw.get("device", {})),
        distributed=_build_dataclass(DistributedConfig, raw.get("distributed", {})),
        model=model_cfg,
        training=training_cfg,
        parallel=_build_dataclass(ParallelConfig, parallel_raw),
        fault_tolerance=_build_dataclass(
            FaultToleranceConfig,
            raw.get("fault_tolerance", {}),
        ),
        cluster=ClusterConfig(),  # Empty - bandwidth from profiling only
        greenctx=_build_dataclass(
            GreenCtxConfig,
            raw.get("greenctx", {}),
        ),
    )

def _build_dataclass(
    cls: type[_DataclassT],
    data: object,
) -> _DataclassT:
    """Build a dataclass from a dict, ignoring unknown keys.

    Supports YAML-to-dataclass field aliases (e.g. ``n_kv_heads``
    in YAML maps to ``num_kv_heads`` in ModelConfig).
    """
    if not isinstance(data, dict):
        return cls()
    dataclass_fields = getattr(cls, "__dataclass_fields__", {})
    valid_fields = set(dataclass_fields.keys())
    # YAML → dataclass field aliases
    _ALIASES: dict[str, str] = {
        "n_kv_heads": "num_kv_heads",
    }
    filtered: dict[str, object] = {}
    for key, value in cast(dict[object, object], data).items():
        if not isinstance(key, str):
            continue
        field_name = _ALIASES.get(key, key)
        if field_name in valid_fields:
            filtered[field_name] = value
    return cls(**filtered)


def save_config(config: BaseConfig, path: str | Path) -> None:
    """Save a BaseConfig to a YAML file."""
    import dataclasses
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _sanitize_for_yaml(dataclasses.asdict(config))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)


def _sanitize_for_yaml(obj: object) -> object:
    """Convert tuples to lists so yaml.safe_dump can serialize them."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(item) for item in obj]
    return obj


__all__ = ["load_config", "save_config"]
