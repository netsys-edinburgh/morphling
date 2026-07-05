# Per-GEMM CUDA Green Context Switching

Morphling can optionally switch CUDA green contexts per linear GEMM using a
trace-driven controller and autograd hooks.

- Before each linear GEMM dispatch in forward and backward, the controller is
  asked to activate a partition with `activate_for_time(elapsed_us)`.
- `elapsed_us` is a wall-clock timestamp in microseconds taken from the first
  GEMM after the green context is set. Timestamps are relative to this epoch.
- After each GEMM completes, a structured log entry is recorded.

## Configuration

Configure a `GreenContextController` in wall-clock mode with a trace file:

```python
from morphling.runtime.green_context import (
    GreenContextConfig,
    GreenContextController,
)

cfg = GreenContextConfig(
    enabled=True,
    trace_path="tests/data/greenctx/second_level_wall.trace",
    clock_mode="wall",
)
controller = GreenContextController.from_config(device_id=0, cfg=cfg)
```

- `enabled=True` turns on green context management; otherwise an off backend
  is used and per-GEMM switching is disabled.
- `clock_mode="wall"` is required so `activate_for_time(elapsed_us)` uses
  wall-clock microseconds instead of step-based indices.
- `trace_path` must point to a trace that the selected backend can load.

## Hook usage and GEMM log format

Install linear hooks with the controller:

```python
from morphling.hooks import apply_hooks, get_gemm_log

apply_hooks("linear", greenctx=controller)
```

This routes linear GEMMs through a custom autograd function that:

- Activates the configured green context before each GEMM in
  forward, `backward_grad_input`, and `backward_grad_weight`.
- Deactivates the context after each GEMM.
- Appends a dictionary to an in-memory GEMM log for each operation.

Each log entry has the fields:

- `gemm_idx`: integer index of the GEMM within the current run.
- `phase`: one of `"forward"`, `"backward_grad_input"`, `"backward_grad_weight"`.
- `start_us`: wall-clock timestamp (microseconds) when `activate_for_time` was
  called.
- `end_us`: wall-clock timestamp (microseconds) after the GEMM completes.
- `duration_us`: `end_us - start_us`.
- `sm_count`: number of SMs requested for this GEMM, or `None` if the backend
  does not report a count.
- `greenctx_enabled`: boolean indicating whether a green context controller was
  configured when the entry was recorded.
- `m`: GEMM M dimension for this operation.
- `n`: GEMM N dimension for this operation.
- `k`: GEMM K dimension for this operation.

## Standalone controller example

The controller can be driven directly — no autograd hooks, no dispatch
backend — which makes it runnable as-is inside the Docker image on any
green-context-capable GPU (driver 12.5+):

```python
import torch

from morphling.runtime.green_context import (
    GreenContextConfig,
    GreenContextController,
)

cfg = GreenContextConfig(
    enabled=True,
    trace_path="tests/data/greenctx/second_level.trace",
    clock_mode="step",
)
controller = GreenContextController.from_config(device_id=0, cfg=cfg)

model = torch.nn.Linear(16, 16).cuda()
x = torch.randn(8, 16, device="cuda")

for step in range(3):
    with controller.step_scope(step) as streams:
        with torch.cuda.stream(streams.comp):
            y = model(x)
    torch.cuda.synchronize()

print("green-context switches:", controller.switch_count())
controller.close()
```

## Per-GEMM logging in the full runtime

The autograd-hook path (`apply_hooks("linear", greenctx=controller)` followed
by `get_gemm_log()`) reroutes every linear GEMM through the proxy **dispatch
backend** at `morphling.hooks.autograd._backend`. That backend is started only
by the training runtime, so per-GEMM green-context logging is exercised through
`scripts/run_devices.py` (which starts the proxy backend and connects devices)
rather than as a standalone snippet — calling `apply_hooks("linear", ...)` and
then running a bare `torch.nn.Linear` with no backend set raises
`AttributeError: 'NoneType' object has no attribute 'async_dispatch_matmul'`.

## Cleanup ordering (critical)

Green contexts share lifetime with the CUDA runtime. Mis-ordered teardown
SIGSEGVs at process exit. The required order is:

1. `cuCtxSetCurrent(ctx)` **before** destroying any resource bound to that
   context (cuBLAS handles, streams, device memory).
2. Free pooled / cached CUDA memory **before** destroying the contexts that
   own it.
3. After destroying all green contexts, call `cudaSetDevice(gpu_id)` to
   restore the primary context — without this, CUDA runtime cleanup at process
   exit will SIGSEGV.
4. Worker threads must `Stop()` (join) before destroying CUDA resources.

This is implemented inside `morphling.runtime.green_context`; downstream users
do not need to manage it manually as long as they let `GreenContextController`
go out of scope before module teardown.
