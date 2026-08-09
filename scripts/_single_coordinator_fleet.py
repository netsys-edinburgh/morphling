"""Local emulated-device fleet lifecycle for coordinator experiments."""

from __future__ import annotations

import os
import subprocess

import torch

_device_processes: list[subprocess.Popen[bytes]] = []


def cleanup_fake_fleet() -> None:
    for process in _device_processes:
        if process.poll() is not None:
            continue
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
    _device_processes.clear()


def spawn_fake_fleet(
    num_devices: int, backend_name: str, proxy_host: str
) -> None:
    cleanup_fake_fleet()
    num_gpus = max(1, torch.cuda.device_count())
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    device_cfg = os.environ.get("CFG_PATH") or os.path.join(
        root, "config", "proxy", "cli.ini"
    )
    log_dir = os.environ.get("MORPHLING_DEV_LOG_DIR", "")
    for index in range(num_devices):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(index % num_gpus)
        command = [
            "morphling_device",
            "--id", str(index),
            "--flops", "100T",
            "--memory", "8G",
            "--ul_bw", "10G",
            "--dl_bw", "10G",
            "--ul_lat", "0",
            "--dl_lat", "0",
            "--backend", backend_name,
            "--cfg", device_cfg,
        ]
        if proxy_host:
            command += ["--proxy_host", proxy_host]
        output = (
            open(os.path.join(log_dir, f"dev_{index}.log"), "wb")
            if log_dir
            else open(os.devnull, "wb")
        )
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
            _device_processes.append(process)
        finally:
            output.close()
