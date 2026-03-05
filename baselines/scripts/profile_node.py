#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _read_mem_total_mb() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024
    except OSError:
        pass
    return 0


def _split_peer(peer: str) -> tuple[str, int]:
    peer = peer.strip()
    if not peer:
        return "", 5201
    if ":" not in peer:
        return peer, 5201

    host, port_str = peer.rsplit(":", 1)
    try:
        return host, int(port_str)
    except ValueError:
        return host, 5201


def get_hardware_info(skip_gpu: bool = False) -> dict[str, Any]:
    hostname = socket.gethostname()

    try:
        arch = (
            subprocess.check_output(["uname", "-m"], text=True)
            .strip()
        )
    except Exception:
        arch = "unknown"

    try:
        cpu_cores = int(
            subprocess.check_output(["nproc"], text=True).strip()
        )
    except Exception:
        cpu_cores = 0

    info: dict[str, Any] = {
        "hostname": hostname,
        "architecture": arch,
        "cpu_cores": cpu_cores,
        "system_memory_mb": _read_mem_total_mb(),
        "gpu_available": False,
    }

    if skip_gpu:
        return info

    if not torch.cuda.is_available():
        return info

    info["gpu_available"] = True
    info["gpu_count"] = torch.cuda.device_count()

    props = torch.cuda.get_device_properties(0)
    info["gpu_name"] = props.name
    info["gpu_memory_mb"] = props.total_memory // (1024 * 1024)
    info["compute_capability"] = f"{props.major}.{props.minor}"

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            info["nvidia_driver"] = result.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return info


class ProfilerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h2 = self.ln2(x)
        return x + self.ff(h2)


def profile_layer(
    block: ProfilerBlock,
    x: torch.Tensor,
    use_cuda: bool,
    warmup: int = 5,
    steps: int = 20,
) -> dict[str, float]:
    block.train()
    latencies: list[float] = []

    for _ in range(warmup):
        out = block(x)
        loss = out.mean()
        loss.backward()
        block.zero_grad(set_to_none=True)
        if use_cuda:
            torch.cuda.synchronize()

    for _ in range(steps):
        t0 = time.perf_counter()
        out = block(x)
        loss = out.mean()
        loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        block.zero_grad(set_to_none=True)

    values = np.array(latencies, dtype=np.float64)
    mean_ms = float(values.mean())

    tokens = float(x.shape[0] * x.shape[1])
    tokens_per_sec = 0.0
    if mean_ms > 0:
        tokens_per_sec = tokens / (mean_ms / 1000.0)

    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": float(np.percentile(values, 50)),
        "latency_ms_p90": float(np.percentile(values, 90)),
        "latency_ms_std": float(values.std()),
        "tokens_per_sec": tokens_per_sec,
        "steps": float(steps),
    }


def _wait_for_port(
    host: str,
    port: int,
    timeout: float = 10.0,
    poll: float = 1.0,
) -> bool:
    """Block until *host:port* accepts a TCP
    connection, or *timeout* seconds elapse."""
    import socket as _socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = _socket.create_connection(
                (host, port), timeout=2
            )
            s.close()
            return True
        except OSError:
            time.sleep(poll)
    return False


def measure_tcp_bandwidth_iperf(
    peer: str,
    duration: int = 5,
    retries: int = 3,
    retry_delay: float = 3.0,
) -> dict[str, Any]:
    host, port = _split_peer(peer)
    if not host:
        return {
            "ok": False,
            "method": "iperf3",
            "peer": peer,
            "error": "empty peer",
        }

    cmd = [
        "iperf3",
        "-c",
        host,
        "-p",
        str(port),
        "-J",
        "-t",
        str(duration),
        "-P",
        "1",
        "--connect-timeout",
        "5000",
    ]

    last_error = "unknown"
    for attempt in range(retries):
        if attempt > 0:
            # Flat retry delay — keep it short since
            # --connect-timeout already handles hangs.
            print(
                f"  iperf3 {peer}: retry "
                f"{attempt + 1}/{retries} "
                f"in {retry_delay:.0f}s "
                f"(last: {last_error[:60]})"
            )
            time.sleep(retry_delay)

        # NOTE: do NOT call _wait_for_port() here — it
        # opens a TCP connection that consumes the
        # --one-off iperf3 server, causing the real
        # iperf3 client to hit a dead socket.  The
        # --connect-timeout 5000 flag handles unreachable
        # servers.

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 30,
                check=False,
            )
        except FileNotFoundError:
            return {
                "ok": False,
                "method": "iperf3",
                "peer": peer,
                "error": "iperf3 not installed",
            }
        except Exception as exc:
            last_error = str(exc)
            continue

        if proc.returncode != 0:
            last_error = (
                proc.stderr.strip()
                or proc.stdout.strip()
                or "iperf3 failed"
            )
            continue

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            last_error = f"bad iperf3 json: {exc}"
            continue

        end = data.get("end", {})
        recv = end.get("sum_received", {})
        sent = end.get("sum_sent", {})
        bps = recv.get("bits_per_second")
        if bps is None:
            bps = sent.get("bits_per_second")

        if bps is None:
            last_error = "missing bits_per_second"
            continue

        return {
            "ok": True,
            "method": "iperf3",
            "peer": peer,
            "bandwidth_mbps": float(bps) / 1e6,
            "attempts": attempt + 1,
        }

    return {
        "ok": False,
        "method": "iperf3",
        "peer": peer,
        "error": last_error,
        "attempts": retries,
    }


def measure_tcp_bandwidth_socket(
    peer: str,
    payload_mb: int = 64,
    timeout: int = 8,
) -> dict[str, Any]:
    host, port = _split_peer(peer)
    if not host:
        return {
            "ok": False,
            "method": "socket",
            "peer": peer,
            "error": "empty peer",
        }

    payload = b"x" * 65536
    total_bytes = payload_mb * 1024 * 1024
    sent = 0
    start = 0.0
    sock: socket.socket | None = None

    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(timeout)
        start = time.perf_counter()

        while sent < total_bytes:
            remain = total_bytes - sent
            chunk = payload if remain >= len(payload) else payload[:remain]
            wrote = sock.send(chunk)
            if wrote <= 0:
                raise RuntimeError("socket write returned 0")
            sent += wrote

        elapsed = time.perf_counter() - start
        if elapsed <= 0:
            raise RuntimeError("non-positive elapsed time")

        mbps = (sent * 8) / (elapsed * 1e6)
        return {
            "ok": True,
            "method": "socket",
            "peer": peer,
            "bandwidth_mbps": float(mbps),
            "bytes_sent": sent,
            "elapsed_sec": float(elapsed),
        }
    except Exception as exc:
        return {
            "ok": False,
            "method": "socket",
            "peer": peer,
            "error": str(exc),
        }
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def measure_network_bandwidth(
    peer: str,
    strict_iperf: bool = False,
) -> dict[str, Any]:
    iperf_result = measure_tcp_bandwidth_iperf(peer)
    if iperf_result.get("ok"):
        return iperf_result

    if strict_iperf:
        # No fallback — iperf3 must succeed
        return {
            "ok": False,
            "method": "iperf3",
            "peer": peer,
            "error": iperf_result.get(
                "error", "iperf3 failed"
            ),
            "attempts": iperf_result.get(
                "attempts", 0
            ),
        }

    socket_result = measure_tcp_bandwidth_socket(peer)
    if socket_result.get("ok"):
        socket_result["fallback_from"] = iperf_result.get("error", "unknown")
        return socket_result

    return {
        "ok": False,
        "method": "none",
        "peer": peer,
        "error": {
            "iperf3": iperf_result.get("error", "unknown"),
            "socket": socket_result.get("error", "unknown"),
        },
    }


def profile_all_layers(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    batch_sizes = [
        int(token.strip())
        for token in args.batch_sizes.split(",")
        if token.strip()
    ]

    use_cuda = device.type == "cuda"
    results: dict[str, Any] = {}

    for batch_size in batch_sizes:
        block = ProfilerBlock(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
        ).to(device)

        x = torch.randn(
            batch_size,
            args.seq_len,
            args.embed_dim,
            device=device,
        )

        metrics = profile_layer(
            block=block,
            x=x,
            use_cuda=use_cuda,
        )

        total_latency_ms = metrics["latency_ms_mean"] * args.num_layers
        model_tokens = float(batch_size * args.seq_len)
        model_tokens_per_sec = 0.0
        if total_latency_ms > 0:
            model_tokens_per_sec = model_tokens / (total_latency_ms / 1000.0)

        metrics["estimated_total_latency_ms"] = float(total_latency_ms)
        metrics["estimated_model_tokens_per_sec"] = float(model_tokens_per_sec)
        metrics["num_layers"] = float(args.num_layers)

        results[str(batch_size)] = metrics

        del block
        del x
        if use_cuda:
            torch.cuda.empty_cache()

    return results


def profile_network_peers(
    peers: str,
    strict_iperf: bool = False,
) -> dict[str, Any]:
    peer_list = [
        peer.strip()
        for peer in peers.split(",")
        if peer.strip()
    ]
    if not peer_list:
        return {}

    results: dict[str, Any] = {}
    for peer in peer_list:
        results[peer] = measure_network_bandwidth(
            peer, strict_iperf=strict_iperf,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile a BASELINES node")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--strict-iperf",
        action="store_true",
        default=False,
        help=(
            "Require iperf3 for all network "
            "measurements (no socket fallback)"
        ),
    )
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--network-peers", type=str, default="")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network (iperf) profiling",
    )
    parser.add_argument(
        "--network-only",
        action="store_true",
        help="Run only network profiling and merge into existing profile",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BASELINES NODE PROFILER")
    print("=" * 60)

    torch.manual_seed(0)

    use_cuda = not args.skip_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"rank={args.rank} hostname={socket.gethostname()} device={device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname()
    output_path = output_dir / f"profile_{hostname}_rank{args.rank}.json"

    # --network-only: just measure network and merge
    # into existing profile JSON.
    if args.network_only:
        print("Network-only mode: measuring peers...")
        network_profile = profile_network_peers(
            args.network_peers,
            strict_iperf=args.strict_iperf,
        )
        payload: dict[str, Any] = {}
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        payload["network"] = network_profile
        payload["network_timestamp_unix"] = time.time()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        ok = sum(
            1 for v in network_profile.values()
            if v.get("ok")
        )
        print(
            f"Network profiling done: {ok}/"
            f"{len(network_profile)} peers OK"
        )
        print(f"Updated profile: {output_path}")
        return 0

    hardware = get_hardware_info(skip_gpu=args.skip_gpu)
    layer_profile = profile_all_layers(args=args, device=device)

    network_profile: dict[str, Any] = {}
    if not args.skip_network:
        network_profile = profile_network_peers(
            args.network_peers,
            strict_iperf=args.strict_iperf,
        )

    payload = {
        "rank": args.rank,
        "hostname": hostname,
        "hardware": hardware,
        "model": {
            "num_layers": args.num_layers,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "seq_len": args.seq_len,
        },
        "layer_profile": layer_profile,
        "network": network_profile,
        "timestamp_unix": time.time(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved profile: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
