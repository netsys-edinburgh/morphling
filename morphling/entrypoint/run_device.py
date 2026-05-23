"""Device runner entrypoint for Morphling emulator.

Provides CLI interface for launching virtual or physical device instances
with configurable FLOPs, memory, network bandwidth, and latency parameters.
Supports multiple backends including RabbitMQ, AMQP, MQTT, and proxy mode.
"""

import asyncio
import os
import subprocess
import threading
import time
import uuid
from argparse import REMAINDER, ArgumentParser

from morphling.common import bytes2human, human2bytes


def main():
    parser = ArgumentParser(description="Morphling Emulator Interface")
    parser.add_argument(
        "--id",
        type=int,
        help="The device id",
    )
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

    parser.add_argument(
        "--proxy_host",
        type=str,
        default="",
        help="The host and port of the proxy server (e.g., 155.98.37.203:39000), overrides config file",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="rabbitmq",
        help="The backend to use for the device",
        choices=["rabbitmq", "amqp", "mqtt", "proxy"],  # more to be added later
    )
    parser.add_argument(
        "--emulation",
        action="store_true",
        help="Enable emulation mode",
    )
    parser.add_argument(
        "--cfg",
        default="",
        help="The path to the config file",
    )

    # # positional
    # parser.add_argument(
    #     "user_script",
    #     type=str,
    #     help="The full path to the single GPU user "
    #     "program/script to be launched in parallel, "
    #     "followed by all the arguments for the "
    #     "user script",
    # )

    # # rest from the user program
    # parser.add_argument("user_script_args", nargs=REMAINDER)

    args = parser.parse_args()
    # print(args, flush=True)

    # human to bytes
    args.flops = human2bytes(args.flops)
    args.memory = human2bytes(args.memory)
    args.ul_bw = human2bytes(args.ul_bw)
    args.dl_bw = human2bytes(args.dl_bw)

    os.environ["MORPHLING_PIN_SIZE"] = str(args.memory)

    device_info = {
        "id": args.id,
        "flops": args.flops,
        "memory": args.memory,
        "ul_bw": args.ul_bw,
        "dl_bw": args.dl_bw,
        "ul_lat": int(args.ul_lat * 1e6),
        "dl_lat": int(args.dl_lat * 1e6),
        "logical_time": 0,
    }

    # FIXME(#45): server-side trust model for device-reported FLOPS/bandwidth/latency.

    if args.emulation:
        # enable interception of torch.mm
        print("Enabling interception of torch.mm")
        import torch

        import morphling._C
    from morphling.backend import AutoWorker

    if args.backend == "rabbitmq":

        async def main():
            loop = asyncio.get_event_loop()
            worker = AutoWorker.from_name(args.backend, device_info, loop)
            await worker.connect()
            await worker.start_consuming()

        asyncio.run(main())
    elif args.backend == "amqp":
        worker = AutoWorker.from_name(args.backend, "localhost", 32)
        worker.handle_req()

    elif args.backend == "mqtt":
        worker = AutoWorker.from_name(args.backend, str(args.id))
        worker.start()
        while True:
            time.sleep(1)

    elif args.backend == "proxy":
        os.environ["MORPHLING_FLOPS"] = str(device_info["flops"])
        os.environ["MORPHLING_MEMORY"] = str(device_info["memory"])
        os.environ["MORPHLING_UL_BW"] = str(device_info["ul_bw"])
        os.environ["MORPHLING_DL_BW"] = str(device_info["dl_bw"])
        os.environ["MORPHLING_UL_LAT"] = str(device_info["ul_lat"])
        os.environ["MORPHLING_DL_LAT"] = str(device_info["dl_lat"])

        if args.proxy_host:
            try:
                host, port = args.proxy_host.split(":")
                os.environ["MORPHLING_PROXY_HOST"] = host
                os.environ["MORPHLING_PROXY_PORT"] = port
                print(f"Using proxy server: {host}:{port}", flush=True)
            except ValueError:
                print(
                    f"Error: Invalid proxy_host format '{args.proxy_host}'. Expected format: host:port",
                    flush=True,
                )
                return

        worker = AutoWorker.from_name(args.backend)
        worker.initialize(args.cfg, args.id)
        worker.start()
        while True:
            time.sleep(1)

    # # create env variables
    # env = os.environ.copy()
    # env["MORPHLING_FLOPS"] = str(args.flops)
    # env["MORPHLING_MEMORY"] = str(args.memory)
    # env["MORPHLING_UL_BW"] = str(args.ul_bw)
    # env["MORPHLING_DL_BW"] = str(args.dl_bw)
    # env["MORPHLING_UL_LAT"] = str(args.ul_lat)
    # env["MORPHLING_DL_LAT"] = str(args.dl_lat)

    # # run the user script with the env
    # cmd = ["python", args.user_script] + args.user_script_args

    # print(f"Running user script: {cmd} with env: {env}")
    # subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
