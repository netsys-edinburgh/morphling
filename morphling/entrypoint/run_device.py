import os
import subprocess
import time
from argparse import REMAINDER, ArgumentParser

from morphling.common import bytes2human, human2bytes


def main():
    parser = ArgumentParser(description="Morphling Emulator Interface")

    parser.add_argument(
        "--flops",
        type=str,
        help="The number of FLOPs of the device, support customary units",
    )
    parser.add_argument(
        "--memory",
        type=str,
        help="The memory size in bytes of the device, support customary units",
    )

    parser.add_argument(
        "--ul_bw",
        type=str,
        help="The uplink bandwidth in bytes of the device, support customary units",
    )

    parser.add_argument(
        "--dl_bw",
        type=str,
        help="The downlink bandwidth in bytes of the device, support customary units",
    )

    parser.add_argument(
        "--ul_lat",
        type=float,
        help="The uplink latency of the device",
    )

    parser.add_argument(
        "--dl_lat",
        type=float,
        help="The downlink latency of the device",
    )

    # positional
    parser.add_argument(
        "user_script",
        type=str,
        help="The full path to the single GPU user "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "user script",
    )

    # rest from the user program
    parser.add_argument("user_script_args", nargs=REMAINDER)

    args = parser.parse_args()
    print(args)

    # human to bytes
    args.flops = human2bytes(args.flops)
    args.memory = human2bytes(args.memory)
    args.ul_bw = human2bytes(args.ul_bw)
    args.dl_bw = human2bytes(args.dl_bw)

    # create env variables
    env = os.environ.copy()
    env["MORPHLING_FLOPS"] = str(args.flops)
    env["MORPHLING_MEMORY"] = str(args.memory)
    env["MORPHLING_UL_BW"] = str(args.ul_bw)
    env["MORPHLING_DL_BW"] = str(args.dl_bw)
    env["MORPHLING_UL_LAT"] = str(args.ul_lat)
    env["MORPHLING_DL_LAT"] = str(args.dl_lat)

    # run the user script with the env
    cmd = ["python", args.user_script] + args.user_script_args

    print(f"Running user script: {cmd} with env: {env}")
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
