"""
Run server and device emulator processes with a configured backend.

Usage:
  python3 scripts/run_devices.py --num_devices 4 --model_name facebook/opt-125m \
    --backend proxy --seq_length 128 --batch_size 1 --cfg config/proxy/svr.ini
  python3 scripts/run_devices.py --num_devices 2 --model_name facebook/opt-125m \
    --backend proxy --enable-hooks
"""

import os
import subprocess
import time
from typing import Any, cast

import torch
from transformers import HfArgumentParser

import morphling

# from morphling import set_backend
from morphling.entrypoint import DeviceConfigArguments, ModelConfigArguments
from morphling.hooks import apply_hooks
from scripts._runtime_common import (
    load_model_and_tokenizer,
    prepare_inputs,
    start_backend,
    wait_for_connections,
)

torch.autograd.set_detect_anomaly(True)  # type: ignore[attr-defined]

# # if SIGINT is received, kill all the devices
# def signal_handler(sig, frame):
#     for p in device_processes:
#         p.kill()
#     exit(0)


# import signal

# signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # Detect enable-hooks flag before using HfArgumentParser because some
    # versions of HfArgumentParser can raise if unknown args remain.
    import sys

    _enable_hooks_flag_names = ("--enable-hooks", "--enable_hooks", "--hooks")
    local_enable_hooks = False
    # Build argv without our local flags so HfArgumentParser won't complain
    orig_argv = sys.argv
    filtered_argv = [orig_argv[0]]
    for a in orig_argv[1:]:
        if a in _enable_hooks_flag_names:
            local_enable_hooks = True
        else:
            filtered_argv.append(a)

    # Temporarily replace sys.argv for HfArgumentParser
    sys.argv = filtered_argv

    parser = HfArgumentParser(
        cast(Any, (DeviceConfigArguments, ModelConfigArguments))
    )
    device_args, model_args = parser.parse_args_into_dataclasses()

    # Restore original argv
    sys.argv = orig_argv
    print(device_args, model_args, flush=True)

    os.environ["NUM_DEVICES"] = str(device_args.num_devices)
    num_gpus = torch.cuda.device_count()
    this_file_path = os.path.dirname(os.path.realpath(__file__))

    subprocess.run(
        ["pkill", "-f", "morphling_device"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    # time.sleep(15)
    # start model from here
    model, tokenizer = load_model_and_tokenizer(
        model_args.model_name, dtype=torch.float32
    )

    print("Model loaded", model)

    backend = start_backend(
        backend_name=model_args.backend,
        block_size=model_args.block_size,
        cfg_path=model_args.cfg,
    )

    morphling.hooks.autograd._backend = backend

    time.sleep(5)

    device_processes = []
    print("Running devices", device_args.num_devices)
    for i in range(device_args.num_devices):
        # command = [
        #     "morphling_device",
        #     "--id",
        #     str(i),
        #     "--flops",
        #     str(device_args.device_flops[i]),
        #     "--memory",
        #     str(device_args.device_mem[i]),
        #     "--ul_bw",
        #     str(device_args.ul_bw[i]),
        #     "--dl_bw",
        #     str(device_args.dl_bw[i]),
        #     "--ul_lat",
        #     str(device_args.ul_lat[i]),
        #     "--dl_lat",
        #     str(device_args.dl_lat[i]),
        #     "--emulation",
        #     "--backend",
        #     model_args.backend,
        #     "&",
        # ]

        # env = os.environ.copy()
        # env["MORPHLING_PIN_SIZE"] = str(device_args.device_mem[i])
        # env["SPDLOG_LEVEL"] = os.environ.get("SPDLOG_LEVEL", "info")
        # env["TORCH_SHOW_CPP_STACKTRACES"] = "1"

        command = [
            "CUDA_VISIBLE_DEVICES=" + str(i % num_gpus),
            "bash",
            f"{this_file_path}/run_device.sh",
            str(i),
            str(device_args.device_flops[i]),
            str(device_args.device_mem[i]),
            str(device_args.ul_bw[i]),
            str(device_args.dl_bw[i]),
            str(device_args.ul_lat[i]),
            str(device_args.dl_lat[i]),
            model_args.backend,
            getattr(
                model_args, "proxy_host", ""
            ),  # Pass proxy_host as the 10th parameter (optional)
        ]
        # print("Running device", command)
        os.system(" ".join(command))

        # print("Running device", command)

        # subprocess.Popen(command, env=env)

        # # create new process rather than subprocess
        # os.system(" ".join(command))
        # # device_processes.append(p)

    # Wait for devices to connect for proxy backend
    if model_args.backend == "proxy":
        print("Waiting for devices to connect to proxy server...")
        connection_count = wait_for_connections(
            backend, min_devices=device_args.num_devices, timeout=120
        )
        if connection_count < device_args.num_devices:
            print(
                f"Timeout waiting for devices to connect. Connected: {connection_count}/{device_args.num_devices}"
            )

    time.sleep(5)

    # random text for seqlen > 128
    inputs = prepare_inputs(
        tokenizer,
        batch_size=model_args.batch_size,
        seq_length=model_args.seq_length,
    )

    print("inputs", inputs, flush=True)

    # ref_model = model.to("cuda:0")
    # ref_input_ids = input_ids.to("cuda:0")
    # ref_outputs = ref_model(
    #     **input_ids,
    #     return_dict=True,
    #     output_hidden_states=True,
    #     output_attentions=True,
    # )
    # ref_logits = ref_outputs.logits
    # ref_hidden_states = ref_outputs.hidden_states
    # ref_attentions = ref_outputs.attentions
    # # ref_outputs = [out.cpu() for out in ref_outputs if isinstance(out, torch.Tensor)]

    if local_enable_hooks:
        print("✓ Distributed computation mode: apply_hooks('linear') ENABLED")
        apply_hooks("linear")
    else:
        print("✗ Local computation mode: apply_hooks('linear') DISABLED")

    model = model.to("cpu")
    inputs = inputs.to("cpu")
    start = time.time()
    outputs = model(
        **inputs,
        return_dict=True,
        output_hidden_states=True,
        output_attentions=True,
    )
    end = time.time()
    print("Forward time", end - start)
    out_logits = outputs.logits
    out_hidden_states = outputs.hidden_states
    out_attentions = outputs.attentions
    print("out_logits", out_logits)

    # Save logits to pt file
    os.makedirs("logits_comparison", exist_ok=True)
    suffix = "with_hooks" if local_enable_hooks else "without_hooks"
    logits_path = os.path.join("logits_comparison", f"logits_{suffix}.pt")
    torch.save(out_logits.cpu().detach(), logits_path)
    print(f"✓ Saved logits to {logits_path}")

    labels = inputs["input_ids"]
    loss = torch.nn.functional.cross_entropy(
        out_logits.view(-1, out_logits.size(-1)), labels.view(-1)
    )
    print("Loss", loss)
    start = time.time()
    loss.backward()
    end = time.time()
    print("Backward time", end - start)

    # print("ref_hidden_states", ref_hidden_states)

    # for i, (ref, out) in enumerate(zip(ref_hidden_states, out_hidden_states)):
    #     print("ref", ref)
    #     print("out", out)
    #     assert torch.allclose(
    #         ref.to("cpu"), out.to("cpu"), atol=1e-6
    #     ), f"Attention are not close!, max diff: {torch.max(torch.abs(ref.to('cpu') - out.to('cpu')))}"
    #     print(f"Attention {i} is close!")

    # for i, (ref, out) in enumerate(zip(ref_hidden_states, out_hidden_states)):
    #     assert torch.allclose(
    #         ref.to("cpu"), out.to("cpu"), atol=1e-6
    #     ), f"hidden_states are not close!, max diff: {torch.max(torch.abs(ref.to("cpu") - out.to("cpu")))}"
    #     print(f"Hidden state {i} is close!")

    # assert torch.allclose(
    #     ref_logits.to("cpu"), out_logits.to("cpu"), atol=1e-6
    # ), f"Logits are not close!, max diff: {torch.max(torch.abs(ref_logits.to("cpu") - out_logits.to("cpu")))}"

    # # all elements needs to be close
    # for ref, out in zip(ref_outputs, outputs):
    #     print("ref", ref)
    #     print("out", out)
    #     assert torch.allclose(
    #         ref, out, atol=1e-6
    #     ), f"Outputs are not close!, max diff: {torch.max(torch.abs(ref - out))}"

    # print("All outputs are close!")


# morphling_device_config --num_devices 4 --output build/device_config.json

# morphling_device
