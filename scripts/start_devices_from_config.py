#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class LaunchArgs:
    config: str
    backend: str
    proxy_host: str
    cfg: str
    log_dir: str
    max_devices: int | None
    print_pids: bool


def parse_args() -> LaunchArgs:
    parser = argparse.ArgumentParser(
        description="Start Morphling device processes from fleet config"
    )
    _ = parser.add_argument("--config", required=True, type=str)
    _ = parser.add_argument(
        "--backend",
        default="proxy",
        choices=["proxy"],
    )
    _ = parser.add_argument("--proxy-host", default="127.0.0.1:39000", type=str)
    _ = parser.add_argument("--cfg", default="config/proxy/cli.ini", type=str)
    _ = parser.add_argument("--log-dir", default="logs", type=str)
    _ = parser.add_argument("--max-devices", default=None, type=int)
    _ = parser.add_argument(
        "--print-pids",
        action="store_true",
        help="Print launched PIDs as JSON list",
    )
    ns = parser.parse_args()
    max_devices = (
        int(ns.max_devices)
        if isinstance(ns.max_devices, int) and ns.max_devices >= 0
        else None
    )
    return LaunchArgs(
        config=str(ns.config),
        backend=str(ns.backend),
        proxy_host=str(ns.proxy_host),
        cfg=str(ns.cfg),
        log_dir=str(ns.log_dir),
        max_devices=max_devices,
        print_pids=bool(ns.print_pids),
    )


def _load_fleet(path: Path) -> list[dict[str, int | float | str]]:
    payload_obj = json.loads(path.read_text(encoding="utf-8"))
    payload = payload_obj if isinstance(payload_obj, list) else None
    if payload is None:
        raise ValueError(f"Fleet JSON must be a list: {path}")
    fleet: list[dict[str, int | float | str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        normalized = {str(k): v for k, v in item.items()}
        fleet.append(normalized)
    return fleet


def _pick_rank(entry: dict[str, int | float | str], index: int) -> int:
    if "rank" in entry:
        return int(entry["rank"])
    if "id" in entry:
        return int(entry["id"])
    return index


def _spawn_device(
    entry: dict[str, int | float | str],
    index: int,
    backend: str,
    proxy_host: str,
    cfg: str,
    log_dir: Path,
) -> tuple[subprocess.Popen[bytes], TextIO]:
    rank = _pick_rank(entry, index)
    cmd = [
        "morphling_device",
        "--id",
        str(rank),
        "--flops",
        str(entry["flops"]),
        "--memory",
        str(entry["memory"]),
        "--ul_bw",
        str(entry["ul_bw"]),
        "--dl_bw",
        str(entry["dl_bw"]),
        "--ul_lat",
        str(entry["ul_lat"]),
        "--dl_lat",
        str(entry["dl_lat"]),
        "--backend",
        backend,
    ]
    if backend == "proxy":
        cmd += ["--proxy_host", proxy_host, "--cfg", cfg]

    log_path = log_dir / f"device_{rank}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("SPDLOG_LEVEL", os.environ.get("SPDLOG_LEVEL", "info"))
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc, log_f


def main() -> None:
    args = parse_args()
    fleet_path = Path(args.config)
    log_dir = Path(args.log_dir)
    backend = args.backend
    proxy_host = args.proxy_host
    cfg = args.cfg
    max_devices = args.max_devices
    print_pids = args.print_pids
    log_dir.mkdir(parents=True, exist_ok=True)

    fleet = _load_fleet(fleet_path)
    if max_devices is not None:
        fleet = fleet[: max(0, max_devices)]
    if not fleet:
        raise RuntimeError("No devices to launch")

    processes: list[subprocess.Popen[bytes]] = []
    log_files: list[TextIO] = []
    stopping = False

    def _shutdown(signum: int, _frame: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print(f"Received signal {signum}; stopping {len(processes)} devices...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5.0
        for proc in processes:
            if proc.poll() is None:
                remaining = max(0.0, deadline - time.time())
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for log_f in log_files:
            try:
                _ = log_f.close()
            except Exception:
                pass
        sys.exit(0)

    _ = signal.signal(signal.SIGINT, _shutdown)
    _ = signal.signal(signal.SIGTERM, _shutdown)

    for idx, entry in enumerate(fleet):
        required = {"flops", "memory", "ul_bw", "dl_bw", "ul_lat", "dl_lat"}
        missing = sorted(required - set(entry.keys()))
        if missing:
            raise KeyError(
                f"Device index {idx} missing required fields: {', '.join(missing)}"
            )
        proc, log_f = _spawn_device(
            entry=entry,
            index=idx,
            backend=backend,
            proxy_host=proxy_host,
            cfg=cfg,
            log_dir=log_dir,
        )
        processes.append(proc)
        log_files.append(log_f)

    pids = [proc.pid for proc in processes]
    print(f"Launched {len(processes)} devices")
    print(f"PIDs: {pids}")
    if print_pids:
        print(json.dumps(pids))

    while True:
        time.sleep(1)
        for proc in processes:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Device process exited unexpectedly (pid={proc.pid}, rc={proc.returncode})"
                )


if __name__ == "__main__":
    main()
