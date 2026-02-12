# Zero-Copy Persistent Memory Test Suite

Unit tests and benchmarks for the zero-copy memory pool and serialization
components: `AlignedBufferPool`, `SerializationBuffer`, `ScatterGatherBuffer`,
`MatrixPartition`, and `CudaPinnedMemoryPool`.

## Prerequisites

- Docker with the project image built:

```bash
docker build -t device-emulator .
```

## Quick Start

### 1. Launch a container

```bash
# Without GPU (runs all tests except CUDA pinned pool)
docker run -it --name zerocopy-tests device-emulator bash

# With GPU (runs all tests including CUDA pinned pool)
docker run -it --gpus all --name zerocopy-tests device-emulator bash
```

### 2. Configure and build

```bash
cd /app
mkdir -p build_tests && cd build_tests

cmake -DBUILD_TESTS=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DMORPHLING_PYTHON_EXECUTABLE=/opt/conda/bin/python \
      -G Ninja ..

# Build all tests
ninja test_aligned_buffer_pool \
      test_serialization_buffer \
      test_scatter_gather_buffer \
      test_matrix_partition

# Build CUDA test (requires GPU)
ninja test_cuda_pinned_pool

# Build benchmarks
ninja bench_pool_allocation bench_serialization
```

### 3. Run tests

```bash
# Run all tests via ctest
cd /app/build_tests
ctest --test-dir tests/cpp/zerocopy --output-on-failure

# Or run individually
./tests/cpp/zerocopy/test_aligned_buffer_pool
./tests/cpp/zerocopy/test_serialization_buffer
./tests/cpp/zerocopy/test_scatter_gather_buffer
./tests/cpp/zerocopy/test_matrix_partition
./tests/cpp/zerocopy/test_cuda_pinned_pool   # needs --gpus all
```

### 4. Run benchmarks

```bash
./tests/cpp/zerocopy/bench_pool_allocation
./tests/cpp/zerocopy/bench_serialization
```

## One-liner (non-interactive)

```bash
docker run --rm device-emulator bash -c "
  cd /app && mkdir -p build_tests && cd build_tests &&
  cmake -DBUILD_TESTS=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DMORPHLING_PYTHON_EXECUTABLE=/opt/conda/bin/python -G Ninja .. &&
  ninja test_aligned_buffer_pool test_serialization_buffer \
        test_scatter_gather_buffer test_matrix_partition &&
  ./tests/cpp/zerocopy/test_aligned_buffer_pool &&
  ./tests/cpp/zerocopy/test_serialization_buffer &&
  ./tests/cpp/zerocopy/test_scatter_gather_buffer &&
  ./tests/cpp/zerocopy/test_matrix_partition
"
```

## Test Coverage

| Test Binary | Class Under Test | Tests |
| --- | --- | --- |
| `test_aligned_buffer_pool` | `AlignedBufferPool` | Bucket sizing, page alignment, buffer reuse, pool-full eviction, multi-bucket independence, null release, thread safety |
| `test_serialization_buffer` | `SerializationBuffer` | Read/write roundtrips (uint32/uint64/int64/bytes, native & network order), seek/canread/validate, move semantics, pool-based free |
| `test_scatter_gather_buffer` | `ScatterGatherBuffer` | Reference vs owned segments, total size, destructor pool release, move semantics with cleanup |
| `test_matrix_partition` | `MatrixPartition` | Proto serialize/deserialize roundtrip, zero-copy size parity, flatten-deserialize roundtrip, tensor pointer equality, utility methods |
| `test_cuda_pinned_pool` | `CudaPinnedMemoryPool` | Acquire/release/reuse, bucket sizing, pool-full, thread safety (requires GPU) |

## Benchmark Coverage

| Benchmark Binary | What It Measures |
| --- | --- |
| `bench_pool_allocation` | Pool acquire/release (warm ~20ns) vs raw posix_memalign+mlock (~10us), multi-thread contention (1-16 threads) |
| `bench_serialization` | `SerializeProto` vs `SerializeZeroCopy` throughput (1KB-16MB), deserialization, WriteBytes throughput |

## Build with Sanitizers

```bash
# AddressSanitizer
cmake -DBUILD_TESTS=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DMORPHLING_PYTHON_EXECUTABLE=/opt/conda/bin/python \
      -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
      -G Ninja ..

# ThreadSanitizer
cmake -DBUILD_TESTS=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DMORPHLING_PYTHON_EXECUTABLE=/opt/conda/bin/python \
      -DCMAKE_CXX_FLAGS="-fsanitize=thread" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
      -G Ninja ..
```
