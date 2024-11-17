# python run_devices.py --num_devices 4

import asyncio
import os
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

import morphling

# from morphling import set_backend
from morphling.backend import AutoBackend
from morphling.entrypoint import DeviceConfigArguments, ModelConfigArguments
from morphling.hooks import apply_hooks


# if SIGINT is received, kill all the devices
def signal_handler(sig, frame):
    for p in device_processes:
        p.kill()
    exit(0)


import signal

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    parser = HfArgumentParser((DeviceConfigArguments, ModelConfigArguments))
    device_args, model_args = parser.parse_args_into_dataclasses()
    print(device_args, model_args, flush=True)

    # read the output of the bash script
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    output = subprocess.run(
        ["bash", f"{this_file_path}/env_init.sh"], stdout=subprocess.PIPE
    )
    print("env_init", output.stdout)

    device_processes = []
    print("Running devices", device_args.num_devices)
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
            "--backend",
            model_args.backend,
        ]

        env = os.environ.copy()
        env["MORPHLING_PIN_SIZE"] = str(device_args.device_mem[i])
        env["SPDLOG_LEVEL"] = os.environ.get("SPDLOG_LEVEL", "info")
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1"

        print("Running device", command)

        p = subprocess.Popen(command, env=env)
        device_processes.append(p)

    # start model from here
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    print("Model loaded", model)

    if model_args.backend == "rabbitmq":
        loop = asyncio.get_event_loop()
        backend = AutoBackend.from_name(
            model_args.backend, loop, block_size=model_args.block_size
        )
        loop.run_until_complete(backend.connect())

    elif model_args.backend == "amqp":
        backend = AutoBackend.from_name(
            model_args.backend, "localhost", model_args.block_size
        )

    # backend = AutoBackend.from_name("amqp", "localhost", model_args.block_size)
    morphling.hooks.autograd._backend = backend

    print("Backend connected")

    # random text for seqlen > 128
    input_text = "".join("Hello, my dog is cute. He is a good ") * 128
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    print("input_ids", input_ids, flush=True)

    ref_model = model
    ref_input_ids = input_ids
    ref_outputs = ref_model(**input_ids).logits
    # ref_outputs = [out.cpu() for out in ref_outputs if isinstance(out, torch.Tensor)]

    apply_hooks("linear")

    model = model.to("cpu")
    input_ids = input_ids.to("cpu")
    outputs = model(**input_ids).logits
    # print(outputs)
    # outputs = [out for out in outputs if isinstance(out, torch.Tensor)]
    # print(outputs[0])

    # all elements needs to be close
    for ref, out in zip(ref_outputs, outputs):
        print("ref", ref)
        print("out", out)
        assert torch.allclose(
            ref, out, atol=1e-6
        ), f"Outputs are not close!, max diff: {torch.max(torch.abs(ref - out))}"

    print("All outputs are close!")


# morphling_device_config --num_devices 4 --output build/device_config.json

# morphling_device
