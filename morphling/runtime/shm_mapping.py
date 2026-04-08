# pyright: reportMissingTypeArgument=false, reportOptionalCall=false

import torch

set_tensor_shm = None


def map_params_to_shm(model, param_shm_map: dict, client):
    for name, param in model.named_parameters(recurse=True):
        if name not in param_shm_map:
            print(f"param {name} not found in param_shm_map")
            continue
        shm_name, shm_size = param_shm_map[name]
        tensor = torch.empty(param.data.shape, dtype=param.data.dtype)
        set_tensor_shm(tensor, shm_name, shm_size)
        param.data = tensor
        assert ~(
            torch.isclose(param.data, torch.zeros_like(param.data)).all()
            == True
        ), f"param {name} is zero {param}"

    return model
