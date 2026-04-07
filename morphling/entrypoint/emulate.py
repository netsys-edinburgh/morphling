"""CLI entrypoint for hybrid emulation with virtual time dilation."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Callable, Dict, List

from morphling.emulation.barrier import StructuralBarrier
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


def _parse_float_list(s: str) -> List[float]:
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
    results_out: Dict[int, List[EmulationStepResult]],
) -> None:
    device_results: List[EmulationStepResult] = []
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
        choices=["loopback"],
        default="loopback",
    )
    parser.add_argument("--sleep-compute-s", type=float, default=0.01)
    parser.add_argument("--sleep-comm-s", type=float, default=0.005)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    num_dev = args.num_devices

    if args.alpha is not None:
        alphas = _parse_float_list(args.alpha)
        if len(alphas) != num_dev:
            print(
                f"Error: --alpha has {len(alphas)} values but --num-devices is {num_dev}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        alphas = [1.0] * num_dev

    if args.beta is not None:
        betas = _parse_float_list(args.beta)
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
        num_steps=args.num_steps,
        num_devices=num_dev,
        overlap_mode=args.overlap,
    )

    barrier = StructuralBarrier(num_dev)

    runners: Dict[int, BSPStepRunner] = {}
    for r in range(num_dev):
        runners[r] = BSPStepRunner(rank=r, config=config, barrier=barrier)

    compute_fns: Dict[int, Callable[[], None]] = {}
    comm_fns: Dict[int, Callable[[], None]] = {}
    for r in range(num_dev):
        compute_fns[r] = _make_sleep_compute_fn(args.sleep_compute_s)
        comm_fns[r] = _make_loopback_comm_fn(args.sleep_comm_s)

    results_by_rank: Dict[int, List[EmulationStepResult]] = {}
    threads: List[threading.Thread] = []
    for r in range(num_dev):
        t = threading.Thread(
            target=_run_device,
            args=(
                r,
                runners[r],
                args.num_steps,
                compute_fns[r],
                comm_fns[r],
                results_by_rank,
            ),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    recorder = EmulationRecorder(topology_name="emulation")
    for step_idx in range(args.num_steps):
        merged = EmulationStepResult(step_idx=step_idx)
        for r in range(num_dev):
            device_step = results_by_rank[r][step_idx]
            merged.wall_compute_s.update(device_step.wall_compute_s)
            merged.wall_comm_s.update(device_step.wall_comm_s)
            merged.virtual_compute_s.update(device_step.virtual_compute_s)
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

    if args.output_json:
        export_json({"emulation": sim_result}, args.output_json)
        print(f"JSON exported to {args.output_json}")

    if args.output_csv:
        export_csv({"emulation": sim_result}, args.output_csv)
        print(f"CSV exported to {args.output_csv}")


if __name__ == "__main__":
    main()
