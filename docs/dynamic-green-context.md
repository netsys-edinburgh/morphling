# Dynamic Green-Context SM Preemption (Proxy)

The proxy GPU path can react to a **live, cross-process notification** of how
many SMs it may use *while a GEMM is running*. When the granted SM budget
changes, the in-flight GEMM is chunked and the remaining output columns are
computed on the matching pre-created green context — completed columns are kept
(**resume**, not restart).

The feature is **off by default**. When off, the proxy runs a single plain
`cublasSgemm` at the partition's full SM count, exactly as before but via plain
cuBLAS rather than the retired cuBLASXt path.

## How it works

- Each `XtGemmWorker` pre-creates one green-context slot at every valid SM
  granularity within its partition (CC 8.x: step 2 SMs; CC 9.0+: step 8 SMs)
  and can switch between them cheaply (`SwitchContext`). This already existed.
- An external **producer** publishes a per-worker target SM count into a POSIX
  shared-memory segment. The worker reads its own target at each GEMM **chunk
  boundary** and, if it changed, synchronizes the current stream and switches to
  the green context whose SM count is the largest available value `<=` the
  target (**snap-down** — the proxy never exceeds the granted budget).
- The GEMM runs via [`XtGemmWorker::RunChunkedGemm`](../csrc/scheduler/gpu_worker.cpp)
  on a stream the proxy owns (plain `cublasSgemm`). Full A and B are staged to
  device once; each chunk computes a contiguous range of output columns and
  copies that range back to the host. Device buffers stay valid across a green-
  context switch because green contexts on the same device share the device
  address space (verified on CC 8.6).

All target reads, context switches, and GEMM launches happen on the **single
worker thread**, so no locking of the active context is needed.

## Shared-memory layout

The producer owns creation; the proxy opens the segment read-only. Layout is a
64-byte header followed by an array of 64-byte, cache-line-aligned slots
(defined in [`sm_notify_source.h`](../csrc/scheduler/sm_notify_source.h)):

```c
magic   = 0x534D4743  // 'SMGC'
version = 1

struct SmNotifyHeader { uint32 magic; uint32 version; uint32 num_slots; uint32 reserved[13]; };   // 64 B
struct SmNotifySlot   { uint32 seq; int32 gpu_id; int32 partition_idx; int32 target_sms; uint64 generation; };  // 64 B
```

## Producer contract

1. `shm_open(O_CREAT|O_RDWR)` the configured name, `ftruncate` to
   `sizeof(header) + num_slots * sizeof(slot)`, `mmap` read-write.
2. Fill the header (`magic`, `version`, `num_slots`) and each slot's identity
   (`gpu_id`, `partition_idx`).
3. Publish a new `target_sms` for a slot under a **seqlock**: bump `seq` to odd,
   write `target_sms` (and `generation`), bump `seq` to even. The proxy retries
   its read while `seq` is odd or changes, so it never observes a torn value.

One slot per `(gpu_id, partition_idx)`. The proxy runs one partition per GPU by
default, so `partition_idx = 0`.

## Configuration

Set via the proxy INI file (section `[dynamic_greenctx]`) or the corresponding
environment variable. Precedence: INI > env > default.

| INI key (`[dynamic_greenctx]`) | Env var | Default | Meaning |
|---|---|---|---|
| `enable` | `MORPHLING_DYNGC_ENABLE` | `0` | Master switch. Off => plain-cuBLAS single-shot at partition max. |
| `shm_name` | `MORPHLING_DYNGC_SHM_NAME` | `/morphling_sm_ctl` | POSIX shm segment name. |
| `chunk_target_us` | `MORPHLING_DYNGC_CHUNK_TARGET_US` | `500` | Adaptive target wall-time per chunk. |
| `min_chunk_cols` | `MORPHLING_DYNGC_MIN_CHUNK_COLS` | `64` | Lower clamp on adaptive chunk width. |
| `max_chunk_cols` | `MORPHLING_DYNGC_MAX_CHUNK_COLS` | `0` | Upper clamp; `0` = up to remaining columns. |
| `min_gemm_chunk_threshold` | `MORPHLING_DYNGC_MIN_GEMM_THRESHOLD` | `0` | GEMMs below this FLOP count run single-chunk. |
| `require_shm` | `MORPHLING_DYNGC_REQUIRE_SHM` | `1` | When enabled, hard-fail worker init if the segment is missing/invalid. |
| `min_dwell_chunks` | `MORPHLING_DYNGC_MIN_DWELL` | `0` | Minimum chunks between switches (anti-thrash). |

`enable=0` keeps the proxy fully functional with no producer present.
`require_shm=1` (the default when enabled) means a missing/invalid segment
fails startup with a clear error instead of silently degrading; set
`require_shm=0` for best-effort (run at partition max if the segment is absent).

## Adaptive chunk sizing

Each chunk is timed with CUDA events; an EWMA of columns-per-microsecond drives
the next chunk width toward `chunk_target_us`, clamped to
`[min_chunk_cols, max_chunk_cols-or-remaining]`. Because a smaller SM partition
computes more slowly, its chunks shrink to keep the checkpoint cadence roughly
constant — so reactivity is independent of GEMM shape and active SM count.

## Notes

- **cuBLASXt retired.** Plain cuBLAS is now the sole GPU GEMM backend for the
  proxy; `RunXtGemm`/`cublasXt` were removed. Results are verified within FP32
  tolerance against a torch reference (the resume-across-switch path included),
  so the default path is numerically equivalent to the old cuBLASXt path.
- **Shared address space.** The resume design relies on device buffers staying
  valid across `SwitchContext`. This holds for green contexts on the same
  device and is exercised by `test_chunked_gemm_switch` (mid-GEMM switch with a
  correct result). If a future device violates it, the fallback is to
  reallocate + re-stage device operands immediately after a switch.
- **Exactly-once response.** Resume is an internal loop within one task, never a
  re-enqueue, so each request produces exactly one response across any number of
  mid-GEMM switches.
