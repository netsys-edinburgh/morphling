"""Emulator server entrypoint for Morphling.

Launches the C++ morphling_server with configured checkpoint path,
GPU memory size, CPU pinned memory size, and network listen address.
"""

import os
import subprocess
import sys

from transformers import HfArgumentParser

import morphling
from morphling.common import EmulatorConfig

KB = 1024
MB = 1024 * KB
GB = 1024 * MB


def main():
    parser = HfArgumentParser((EmulatorConfig,))
    args = parser.parse_args()

    print(args)

    print(morphling.__path__[0])

    server_executable = os.path.join(morphling.__path__[0], "morphling_server")
    checkpoint_path = args.ckpt_path
    listen_address = f"{args.listen_ip}:{args.listen_port}"

    env = os.environ.copy()
    env["MORPHLING_SERVER_ADDRESS"] = listen_address
    env["MORPHLING_GPU_SIZE"] = str(args.gpu_memory * GB)
    env["MORPHLING_PIN_SIZE"] = str(args.cpu_memory * GB)

    if args.debug:
        env["SPDLOG_LEVEL"] = "DEBUG"

    sys.exit(
        subprocess.call(
            [
                server_executable,
                "--listen",
                listen_address,
                "--path",
                checkpoint_path,
            ],
            env=env,
        )
    )


if __name__ == "__main__":
    main()
