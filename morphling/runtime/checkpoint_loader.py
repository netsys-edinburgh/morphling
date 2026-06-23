# pyright: reportReturnType=false, reportMissingTypeArgument=false

import gc
import os
from typing import List

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

from morphling.utils import get_checkpoint_paths


def discover_checkpoints(model_name_or_path: str) -> List[str]:
    if os.path.exists(model_name_or_path):
        return get_checkpoint_paths(model_name_or_path)

    model_path = snapshot_download(
        model_name_or_path,
        cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
        ignore_patterns=["flax*", "tf*"],
    )
    if model_path is None:
        raise RuntimeError(
            f"The `snapshot_download` function could not find the checkpoint {model_name_or_path}. "
            f"Please provide a valid checkpoint."
        )

    return get_checkpoint_paths(model_path)


def offload_state_dict(ckpt_files, archer_tensor_handle, dtype) -> dict:
    param_meta_map = {}
    param_id = 0

    for ckpt in tqdm(ckpt_files, desc="Loading checkpoint files", smoothing=0):
        state_dict = {}
        if "safetensors" in ckpt:
            with safe_open(ckpt, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            state_dict = torch.load(ckpt)

        for k, v in state_dict.items():
            state_dict[k] = v.to(dtype).to("cpu")

        for param_name, param in state_dict.items():
            if param_name not in param_meta_map:
                file_offset = archer_tensor_handle.offload_tensor(
                    param, param_id
                )

                param_meta_map[param_name] = {
                    "id": param_id,
                    "size": param.numel() * param.element_size(),
                    "file_offset": file_offset,
                    "shape": tuple(param.shape),
                    "stride": tuple(param.stride()),
                    "dtype": str(param.dtype),
                }
                param_id += 1

        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    return param_meta_map
