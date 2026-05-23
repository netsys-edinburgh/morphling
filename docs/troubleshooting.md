# Troubleshooting

## Version mismatch or missing libraries

Errors like:

```text
undefined symbol: sk_pop_free_ex (e.g., OpenSSL/gRPC mismatch)
ImportError: librttr_core.so.*: cannot open shared object file (missing RTTR)
ImportError: libmosquitto.so.*: cannot open shared object file (missing Mosquitto)
```

usually mean libraries were built in a different environment than the one
used at runtime, or the dynamic linker cannot find them.

**Fix:** rebuild within the current environment to match installed libraries.

```bash
pip install --no-build-isolation --force-reinstall -e .
```

If certain libraries (e.g., RTTR, Mosquitto) are in a non-standard location,
set `LD_LIBRARY_PATH`:

```bash
# Example for RTTR
export LD_LIBRARY_PATH="/path/to/rttr/install/lib:$LD_LIBRARY_PATH"

# Example for local build artefacts (e.g., C++ .so files)
export LD_LIBRARY_PATH="/path/to/emulator/build/lib.linux-x86_64-cpython-310/morphling:$LD_LIBRARY_PATH"
```

## CUDA driver too old for green contexts

Green contexts require CUDA driver **12.5 or newer**. On older drivers the
runtime falls back to "off" mode and per-GEMM SM partitioning is disabled.
You will see a log line at startup similar to:

```text
[green-ctx] driver does not support cuGreenCtxCreate; partitioning disabled
```

To enable: upgrade the NVIDIA driver and rebuild the Docker image so the
container sees the new driver capabilities.

## SM step sizes vary by compute capability

`cuGreenCtxCreate` enforces architecture-specific SM step sizes:

| Compute capability | SM partition step |
|---|---|
| 8.x (Ampere)       | 2 SMs |
| 9.0+ (Hopper)      | 8 SMs |

A trace requesting a non-multiple SM count will be rounded to the next valid
multiple, which can change observed performance. Verify trace SM counts
against your target GPU.

## Cleanup-time SIGSEGV at process exit

Symptoms: tests pass, then Python exits with `Segmentation fault (core
dumped)`. Almost always caused by:

- CUDA pools (pinned memory, AlignedBufferPool, GPU caches) being destroyed
  *after* the green context that owned them.
- A worker thread still running when CUDA resources are released.

See `docs/green-context.md` § "Cleanup ordering" for the canonical fix.

## Docker image is stale after code change

Per [`CLAUDE.md`](../CLAUDE.md) §3, every Python or C++ change requires an
image rebuild:

```bash
make docker-build
make docker-test
```

If tests fail unexpectedly after a code change, the first hypothesis is
"image is stale". Rebuild before debugging anything else.

## Submodules out of sync

If a fresh clone fails to build with missing files under `external/`:

```bash
git submodule update --init --recursive
```
