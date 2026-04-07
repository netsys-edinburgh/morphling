"""CLI entrypoint for hybrid emulation with virtual time dilation."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import threading
import time
import uuid
from multiprocessing.managers import DictProxy
from typing import Callable, Literal, Optional, cast

from morphling.emulation.barrier import StructuralBarrier
from morphling.emulation.comm_shm import ShmAllReduceCommFn
from morphling.emulation.dilation import DeviceDilation, DilationMatrix
from morphling.emulation.recorder import EmulationRecorder
from morphling.emulation.step_runner import (
    BSPStepRunner,
    EmulationConfig,
    EmulationStepResult,
)
from morphling.simulator.output import (
    export_csv,
    export_json,
    format_comparison_table,
)


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def _make_sleep_compute_fn(duration_s: float) -> Callable[[], None]:
    def fn() -> None:
        time.sleep(duration_s)

    return fn


def _make_loopback_comm_fn(duration_s: float) -> Callable[[], None]:
    def fn() -> None:
        time.sleep(duration_s)

    return fn


def _run_device(
    rank: int,
    runner: BSPStepRunner,
    num_steps: int,
    compute_fn: Callable[[], None],
    comm_fn: Callable[[], None],
    results_out: dict[int, list[EmulationStepResult]],
) -> None:
    device_results: list[EmulationStepResult] = []
    for step_idx in range(num_steps):
        step_result = runner.run_step(step_idx, compute_fn, comm_fn)
        device_results.append(step_result)
    results_out[rank] = device_results


def _run_device_process(
    rank: int,
    config: EmulationConfig,
    num_steps: int,
    sleep_compute_s: float,
    comm_fn_name: str,
    sleep_comm_s: float,
    tensor_size: int,
    barrier_host: str,
    barrier_run_id: str,
    results_out: DictProxy[int, list[EmulationStepResult]],
) -> None:
    compute_fn = _make_sleep_compute_fn(sleep_compute_s)
    barrier = StructuralBarrier(
        config.num_devices,
        backend="rabbitmq",
        host=barrier_host,
        run_id=barrier_run_id,
    )
    runner = BSPStepRunner(rank=rank, config=config, barrier=barrier)
    if comm_fn_name == "loopback":
        comm_fn = _make_loopback_comm_fn(sleep_comm_s)
    else:
        comm_fn = ShmAllReduceCommFn(
            rank=rank,
            world_size=config.num_devices,
            tensor_size=tensor_size,
            barrier=barrier,
        )

    device_results: list[EmulationStepResult] = []
    for step_idx in range(num_steps):
        step_result = runner.run_step(step_idx, compute_fn, comm_fn)
        device_results.append(step_result)
    results_out[rank] = device_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid emulation with virtual time dilation"
    )
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--alpha",
        type=str,
        default=None,
        help="Per-device compute dilation factors (comma-separated)",
    )
    parser.add_argument(
        "--beta",
        type=str,
        default=None,
        help="Per-device comm dilation factors (comma-separated)",
    )
    parser.add_argument(
        "--overlap",
        choices=["none", "full"],
        default="none",
    )
    parser.add_argument(
        "--compute-fn",
        choices=["sleep"],
        default="sleep",
    )
    parser.add_argument(
        "--comm-fn",
        choices=["loopback", "shm"],
        default="loopback",
    )
    parser.add_argument(
        "--barrier",
        choices=["threading", "rabbitmq"],
        default="threading",
    )
    parser.add_argument(
        "--rabbitmq-host",
        type=str,
        default="amqp://localhost/",
    )
    parser.add_argument("--sleep-compute-s", type=float, default=0.01)
    parser.add_argument("--sleep-comm-s", type=float, default=0.005)
    parser.add_argument("--tensor-size", type=int, default=1024)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    num_dev = cast(int, args.num_devices)
    num_steps = cast(int, args.num_steps)
    overlap_mode = cast(Literal["none", "full"], args.overlap)
    barrier_backend = cast(str, args.barrier)
    comm_fn_name = cast(str, args.comm_fn)
    sleep_compute_s = cast(float, args.sleep_compute_s)
    sleep_comm_s = cast(float, args.sleep_comm_s)
    tensor_size = cast(int, args.tensor_size)
    rabbitmq_host = cast(str, args.rabbitmq_host)
    output_json = cast(Optional[str], args.output_json)
    output_csv = cast(Optional[str], args.output_csv)

    if args.alpha is not None:
        alphas = _parse_float_list(cast(str, args.alpha))
        if len(alphas) != num_dev:
            print(
                f"Error: --alpha has {len(alphas)} values but --num-devices is {num_dev}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        alphas = [1.0] * num_dev

    if args.beta is not None:
        betas = _parse_float_list(cast(str, args.beta))
        if len(betas) != num_dev:
            print(
                f"Error: --beta has {len(betas)} values but --num-devices is {num_dev}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        betas = [1.0] * num_dev

    dilation = DilationMatrix(
        devices=[
            DeviceDilation(rank=r, alpha_compute=alphas[r], beta_comm=betas[r])
            for r in range(num_dev)
        ]
    )
    dilation.validate()

    config = EmulationConfig(
        dilation=dilation,
        num_steps=num_steps,
        num_devices=num_dev,
        overlap_mode=overlap_mode,
    )

    results_by_rank: dict[int, list[EmulationStepResult]]

    if barrier_backend == "threading":
        barrier = StructuralBarrier(num_dev, backend="threading")

        runners: dict[int, BSPStepRunner] = {}
        for r in range(num_dev):
            runners[r] = BSPStepRunner(rank=r, config=config, barrier=barrier)

        compute_fns: dict[int, Callable[[], None]] = {}
        comm_fns: dict[int, Callable[[], None]] = {}
        for r in range(num_dev):
            compute_fns[r] = _make_sleep_compute_fn(sleep_compute_s)
            if comm_fn_name == "loopback":
                comm_fns[r] = _make_loopback_comm_fn(sleep_comm_s)
            else:
                comm_fns[r] = ShmAllReduceCommFn(
                    rank=r,
                    world_size=num_dev,
                    tensor_size=tensor_size,
                    barrier=barrier,
                )

        results_by_rank = {}
        threads: list[threading.Thread] = []
        for r in range(num_dev):
            t = threading.Thread(
                target=_run_device,
                args=(
                    r,
                    runners[r],
                    num_steps,
                    compute_fns[r],
                    comm_fns[r],
                    results_by_rank,
                ),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    else:
        manager = multiprocessing.Manager()
        shared_results = cast(
            DictProxy[int, list[EmulationStepResult]],
            manager.dict(),
        )
        processes: list[multiprocessing.Process] = []
        barrier_run_id = str(uuid.uuid4())

        for r in range(num_dev):
            p = multiprocessing.Process(
                target=_run_device_process,
                args=(
                    r,
                    config,
                    num_steps,
                    sleep_compute_s,
                    comm_fn_name,
                    sleep_comm_s,
                    tensor_size,
                    rabbitmq_host,
                    barrier_run_id,
                    shared_results,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            print(
                f"Error: worker process failed with exit codes {failed}",
                file=sys.stderr,
            )
            sys.exit(1)

        results_by_rank = {int(k): v for k, v in shared_results.items()}
        manager.shutdown()

    recorder = EmulationRecorder(topology_name="emulation")
    for step_idx in range(num_steps):
        merged = EmulationStepResult(step_idx=step_idx)
        for r in range(num_dev):
            device_step = results_by_rank[r][step_idx]
            merged.wall_compute_s.update(device_step.wall_compute_s)
            merged.wall_comm_s.update(device_step.wall_comm_s)
            merged.virtual_compute_s.update(device_step.virtual_compute_s)
            merged.virtual_compute_fwd_s.update(
                device_step.virtual_compute_fwd_s
            )
            merged.virtual_compute_bwd_s.update(
                device_step.virtual_compute_bwd_s
            )
            merged.virtual_comm_s.update(device_step.virtual_comm_s)
            merged.virtual_total_s.update(device_step.virtual_total_s)
        merged.step_virtual_time_s = max(merged.virtual_total_s.values())
        for r in range(num_dev):
            device_step = results_by_rank[r][step_idx]
            for phase, overhead in device_step.barrier_overhead_s.items():
                merged.barrier_overhead_s[phase] = max(
                    merged.barrier_overhead_s.get(phase, 0.0), overhead
                )
        recorder.record_step(merged)

    sim_result = recorder.finalize()
    table = format_comparison_table({"emulation": sim_result})
    print(table)

    wall_summary = recorder.wall_time_summary()
    print(f"\nWall time: {wall_summary['total_wall_time_s']:.3f}s")
    print(f"Barrier overhead: {wall_summary['barrier_overhead_pct']:.1f}%")

    if output_json:
        export_json({"emulation": sim_result}, output_json)
        print(f"JSON exported to {output_json}")

    if output_csv:
        export_csv({"emulation": sim_result}, output_csv)
        print(f"CSV exported to {output_csv}")


if __name__ == "__main__":
    main()
