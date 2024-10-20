# python run_devices.py --num_devices 4

import asyncio
import os
import subprocess

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from morphling import set_backend
from morphling.backend import AutoBackend
from morphling.entrypoint import DeviceConfigArguments, ModelConfigArguments
from morphling.hooks import apply_hooks

if __name__ == "__main__":
    parser = HfArgumentParser((DeviceConfigArguments, ModelConfigArguments))
    device_args, model_args = parser.parse_args_into_dataclasses()
    print(device_args, model_args)

    # read the output of the bash script
    output = subprocess.run(
        ["bash", "scripts/env_init.sh"], stdout=subprocess.PIPE
    )
    print(output.stdout)

    device_processes = []
    for i in range(device_args.num_devices):
        command = [
            "morphling_device",
            "--flops",
            str(device_args.device_flops[i]),
            "--memory",
            str(device_args.device_mem[i]),
            "--ul_bw",
            str(device_args.ul_bw[i]),
            "--dl_bw",
            str(device_args.dl_bw[i]),
            "--ul_lat",
            str(device_args.ul_lat[i]),
            "--dl_lat",
            str(device_args.dl_lat[i]),
            "--emulation",
        ]

        env = os.environ.copy()
        env["MORPHLING_PIN_SIZE"] = str(device_args.device_mem[i])

        p = subprocess.Popen(command, env=env)
        device_processes.append(p)

    # start model from here
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    async def run_backend():
        loop = asyncio.get_event_loop()
        backend = AutoBackend.from_name(
            model_args.backend, loop, block_size=model_args.block_size
        )
        await backend.connect()
        set_backend(backend)

    asyncio.run(run_backend())

    input_text = "Hello, my dog is cute."
    input_ids = tokenizer(input_text, return_tensors="pt")
    print(input_ids)

    apply_hooks("linear")

    outputs = model(**input_ids)

    print(outputs)


# morphling_device_config --num_devices 4 --output build/device_config.json

# morphling_device
