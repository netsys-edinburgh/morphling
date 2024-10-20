# python run_devices.py --num_devices 4

import os
import subprocess

from transformers import HfArgumentParser

from morphling.entrypoint import DeviceConfigArguments

if __name__ == "__main__":
    parser = HfArgumentParser(DeviceConfigArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)

    # read the output of the bash script
    output = subprocess.run(
        ["bash", "scripts/env_init.sh"], stdout=subprocess.PIPE
    )
    print(output.stdout)

    device_processes = []
    for i in range(args.num_devices):
        command = [
            "morphling_device",
            "--flops",
            str(args.device_flops[i]),
            "--memory",
            str(args.device_mem[i]),
            "--ul_bw",
            str(args.ul_bw[i]),
            "--dl_bw",
            str(args.dl_bw[i]),
            "--ul_lat",
            str(args.ul_lat[i]),
            "--dl_lat",
            str(args.dl_lat[i]),
            "--emulation",
        ]

        env = os.environ.copy()
        env["MORPHLING_PIN_SIZE"] = str(args.device_mem[i])

        p = subprocess.Popen(command, env=env)
        device_processes.append(p)

    # start model from here

# morphling_device_config --num_devices 4 --output build/device_config.json

# morphling_device
