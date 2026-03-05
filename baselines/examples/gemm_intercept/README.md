# GEMM Intercept (LD_PRELOAD)

`libgemm_intercept.so` is an LD_PRELOAD shim that intercepts cuBLAS/cuBLASLt GEMM calls and logs per-call timing as CSV.

## Build

```bash
cd baselines/examples/gemm_intercept
make
```

Artifact:

```text
libgemm_intercept.so
```

## Usage

### Prerequisites

The interceptor must be able to find the real cuBLAS functions via `dlsym(RTLD_NEXT, ...)`. This requires the real cuBLAS libraries to be in the LD_PRELOAD chain **after** the interceptor:

```bash
export GEMM_LOG_PATH=/tmp/gemm_calls.csv

# Correct order: interceptor FIRST, then real libs
export LD_PRELOAD="/path/to/libgemm_intercept.so:/usr/local/cuda/lib64/libcublas.so.12:/usr/local/cuda/lib64/libcublasLt.so.12"

# Run your workload
python3 baselines/examples/llama_sst_single_gpu.py --max-iters 100
```

**Why this order matters:**
- The interceptor uses `dlsym(RTLD_NEXT, ...)` to get the real function pointer
- `RTLD_NEXT` searches the *next* object in the search order
- If the real cuBLAS library isn't loaded yet, `RTLD_NEXT` returns NULL
- By preloading cuBLAS after the interceptor, we ensure it's available

### Using the Launcher Script

The launcher script (`run_llama_sst.sh`) handles this automatically:

```bash
# Mode: gemm (interception only)
./run_llama_sst.sh gemm --max-iters 100

# Mode: both (green context + interception)
./run_llama_sst.sh both --max-iters 100
```

## Intercepted Functions

| Function | Precision | Notes |
|----------|-----------|-------|
| `cublasSgemm_v2` | FP32 | Standard single-precision GEMM |
| `cublasDgemm_v2` | FP64 | Double-precision GEMM |
| `cublasHgemm` | FP16 | Half-precision GEMM |
| `cublasGemmEx` | Mixed | Generic GEMM with compute type (most common in PyTorch AMP) |
| `cublasLtMatmul` | Mixed | cuBLASLt matrix multiplication (used by modern PyTorch) |

## CSV Format

Header:

```text
function_name,start_ns,end_ns,duration_ns,m,n,k
```

Each row records:

- **function_name**: cuBLAS function name
- **start_ns**: Start timestamp from `CLOCK_MONOTONIC` (nanoseconds)
- **end_ns**: End timestamp from `CLOCK_MONOTONIC` (nanoseconds)
- **duration_ns**: `end_ns - start_ns`
- **m, n, k**: GEMM dimensions (M×N output, K inner dimension)

For `cublasLtMatmul`, dimensions are derived from matrix layout `ROWS`/`COLS` attributes via `cublasLtMatrixLayoutGetAttribute`.

## Example Output

```csv
function_name,start_ns,end_ns,duration_ns,m,n,k
cublasGemmEx,1234567890123456,1234567890123506,50,512,1024,512
cublasGemmEx,1234567890123600,1234567890123650,50,512,512,512
cublasLtMatmul,1234567890123800,1234567890123855,55,1024,2048,512
```

## Thread Safety

The interceptor uses `pthread_mutex` to serialize access to the log file. All GEMM calls from any thread are logged safely.

## Performance Impact

The interceptor adds ~50-100ns overhead per GEMM call (mutex lock + timestamp + fprintf). For typical DL training with thousands of GEMMs per step, this is negligible (<0.1% overhead).

## Troubleshooting

### Empty log file

Check that:
1. `GEMM_LOG_PATH` environment variable is set
2. `LD_PRELOAD` includes both the interceptor AND the real cuBLAS libraries
3. The process has write permissions to the log path

### "undefined symbol" errors

Ensure you're running on a CUDA 12.x system with cuBLAS installed:

```bash
ls /usr/local/cuda/lib64/libcublas.so.12
ls /usr/local/cuda/lib64/libcublasLt.so.12
```

### dlsym returns NULL

This means the real cuBLAS library isn't in the LD_PRELOAD chain or isn't loaded. Fix:

```bash
# Add the real libs to LD_PRELOAD
export LD_PRELOAD="${INTERCEPTOR}:${CUBLAS}:${CUBLASLT}"
```

## Integration with Violation Analysis

The GEMM log is the primary input for SM partition violation analysis. See the [LLaMA SST Guide](../LLAMA_SST_GUIDE.md) for the full workflow.

Quick example:

```bash
# 1. Train with GEMM interception
./run_llama_sst.sh both --max-iters 500

# 2. Analyze violations
./run_llama_sst.sh analyze
```

## References

- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [dlsym(3) - Linux Manual](https://man7.org/linux/man-pages/man3/dlsym.3.html)
- [LD_PRELOAD tricks](https://jvns.ca/blog/2014/09/06/how-to-read-an-executable/)
