# Performance Log Formats

Morphling's proxy server and devices each emit a **separate performance
log** via `DevicePartitionTracker::InitSeparatePerfLog`
([`csrc/backend/device_tracker.cpp`](../csrc/backend/device_tracker.cpp)).
These logs capture per-GEMM virtual-time events, network throughput, and
(optionally) measured-vs-reported device profiles. This document is the
canonical reference for those on-disk formats.

## Where the logs live

- Server: `logs/perf_server.log`
- Device *N*: `logs/perf_device_<N>.log`

The directory is relative to the process working directory (the runtime
passes `./logs`). The underlying `base::LogFile` rolls files at 512 MiB
and appends a `.<date>-<host>-<pid>.log` suffix, so on disk a single
logical log may appear as e.g. `perf_server.log.20260605-111207.host.118.log`.
Every file opens with comment headers (lines beginning with `#`)
describing each record format.

## `gemm_id`

`gemm_id` is a **global GEMM operation counter** assigned at dispatch
time. In [`csrc/backend/proxy_svr.cpp`](../csrc/backend/proxy_svr.cpp) it is
sourced from the atomic `gemm_id_count_` (`proxy_svr.h`), stamped onto
every `MatrixPartition` in `DispatchMatMulAsync` (`partition->gemm_id =
gemm_id_count_`), and incremented after each dispatch. It appears in both
the VTIME and Throughput records so events can be grouped by the GEMM that
produced them; it starts at 0 and increases by one per `DispatchMatMulAsync`.

## VTIME format

```
VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,vt_duration_us
```

| Field            | Meaning                                                    |
|------------------|------------------------------------------------------------|
| `VTIME`          | Record type tag                                            |
| `timestamp_us`   | System (wall-clock) timestamp, microseconds                |
| `device_id`      | Device ID (0, 1, 2, ...)                                    |
| `gemm_id`        | Global GEMM operation ID (see above)                       |
| `phase`          | `COMPUTE`, `RECEIVE`, or `SEND`                            |
| `event`          | `START` or `END`                                           |
| `vt_start_us`    | Virtual-time start, microseconds                           |
| `vt_end_us`      | Virtual-time end, microseconds                             |
| `vt_duration_us` | Virtual-time duration, microseconds (`vt_end - vt_start`)  |

Emitted by `DevicePartitionTracker::LogVirtualTimeEvent`. Example:

```
VTIME,1765380077213829,2,0,SEND,END,10098658,10100432,1774
VTIME,1765380077218046,2,0,COMPUTE,START,3024656,3024656,0
VTIME,1765380077218874,2,0,COMPUTE,END,3024656,3025486,830
```

## Throughput format

```
timestamp_us,device_id,gemm_id,direction,bytes,throughput_b_s,epoch_start_us,epoch_end_us,packet_duration_us
```

| Field                | Meaning                                          |
|----------------------|--------------------------------------------------|
| `timestamp_us`       | System (wall-clock) timestamp, microseconds      |
| `device_id`          | Device ID                                        |
| `gemm_id`            | Global GEMM operation ID                         |
| `direction`          | `UPLOAD` or `DOWNLOAD`                            |
| `bytes`              | Bytes transferred                                |
| `throughput_b_s`     | Throughput, bytes per second                     |
| `epoch_start_us`     | Transfer start, microseconds                     |
| `epoch_end_us`       | Transfer end, microseconds                       |
| `packet_duration_us` | Packet duration, microseconds                    |

Emitted by `DevicePartitionTracker::LogThroughputToFile`. Example:

```
1765380077219527,2,0,DOWNLOAD,131153,70022.96,1765380077219482,1765380077219482,0
1765380077335074,2,1,DOWNLOAD,131154,131878.83,1765380077219482,1765380077335065,115583
```

## PROFILE_DELTA format (optional, issue #60)

Only written when server-side device measurement is enabled (see
[`docs/deployment.md`](deployment.md#server-measured-device-profile-issue-55)).
Each row records measured-vs-reported device profile fields; it is
observability-only and drives no scheduling decision.

```
PROFILE_DELTA,timestamp_us,device_id,uuid,
  flops_reported,flops_measured,flops_verified,flops_ratio,
  ul_bw_reported,ul_bw_measured,ul_bw_ratio,
  dl_bw_reported,dl_bw_measured,dl_bw_ratio,
  ul_lat_reported_us,dl_lat_reported_us,measured_lat_ns
```

- `*_ratio = measured / reported`, or `-1` when the reported field is `0`.
- Latency has **no** ratio column: reported latency is in microseconds
  while `measured_lat_ns` is in nanoseconds. Normalize at analysis time.

Emitted by `DevicePartitionTracker::LogProfileDelta` /
`FormatProfileDeltaRow`
([`csrc/backend/profile_delta_format.h`](../csrc/backend/profile_delta_format.h)).

## Merging logs

[`scripts/merge_perf_logs.py`](../scripts/merge_perf_logs.py) merges all
`perf_*.log` files in a directory into a single timestamp-sorted log,
preserving the `#` format headers:

```bash
python3 scripts/merge_perf_logs.py logs/ logs/perf_merged.log
```

It prints VTIME / Throughput / header counts and writes the sorted output.
The pipeline (header preservation, `gemm_id` field positions, timestamp
ordering) is covered by
[`tests/python/unit/test_merge_perf_logs.py`](../tests/python/unit/test_merge_perf_logs.py).

After merging, downstream sync analysis uses
[`scripts/sync_virtual_time.py`](../scripts/sync_virtual_time.py) and
[`scripts/analyze_sync.py`](../scripts/analyze_sync.py); see
[`docs/EARLIEST_vs_LATEST.md`](EARLIEST_vs_LATEST.md).

## Source of truth

| Record         | Header written by              | Row written by                              |
|----------------|--------------------------------|---------------------------------------------|
| VTIME          | `InitSeparatePerfLog`          | `LogVirtualTimeEvent`                        |
| Throughput     | `InitSeparatePerfLog`          | `LogThroughputToFile`                        |
| PROFILE_DELTA  | `InitSeparatePerfLog`          | `LogProfileDelta` / `FormatProfileDeltaRow`  |

All in [`csrc/backend/device_tracker.cpp`](../csrc/backend/device_tracker.cpp)
(plus `profile_delta_format.h`). If you change a wire format, update the
header string, the row emitter, the merge script, and the table above
together.
