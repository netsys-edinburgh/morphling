# C++ Tests

This directory contains standalone C++ tests and benchmarks organized by type and component.

## Test Organization

```
tests/cpp/
├── unit/           # Unit tests (Google Test)
│   ├── cuda/      # CUDA/cuBLAS tests
│   ├── memory/    # Memory operation tests
│   ├── ml/        # ML framework tests
│   ├── network/   # Network/messaging tests
│   ├── worker/    # Worker tests
│   └── zerocopy/  # Zero-copy buffer tests
│
├── bench/          # Benchmarks (Google Benchmark)
│   ├── cuda/      # CUDA benchmarks
│   └── zerocopy/  # Zero-copy benchmarks
│
└── integration/   # Integration tests (placeholder)
```

## Running Tests

Tests are automatically built in the Docker image. Use the test runner script:

```bash
# Run all tests
./tests/run_cpp_tests.sh

# Run specific categories
./tests/run_cpp_tests.sh unit      # Unit tests only
./tests/run_cpp_tests.sh cuda     # CUDA tests
./tests/run_cpp_tests.sh worker   # Worker tests
```

## Running Tests in Docker

### Option A: exec into existing container

```bash
docker exec -it <container_id> bash -lc "cd /app && ./tests/run_cpp_tests.sh"
```

### Option B: one-shot run

```bash
docker run --rm --gpus=all --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v "$(pwd)":/app -w /app device-emulator:latest \
  bash -lc "./tests/run_cpp_tests.sh"
```

### Run individual tests

```bash
docker run --rm --gpus=all device-emulator:latest \
  /app/tests/cpp/build/test_worker_base
```

## Available Tests

The following tests are built by default:

- `test_worker_base` - WorkerBase tests (unit/worker/)
- `test_cublas_hostalloc_direct` - CUDA host allocation test
- `test_cublas_hostregister_posix` - CUDA host register test
- `test_cublas_error15_repro` - cuBLAS error reproduction test

Additional tests require specific CMake flags (see Building Tests below).

## Building Tests (if needed)

If you need to rebuild tests manually:

```bash
# Build all tests with all features
cmake -S tests/cpp -B tests/cpp/build \
    -DENABLE_CUDA_TESTS=ON \
    -DENABLE_XTGEMM_TESTS=ON \
    -DENABLE_ZEROCOPY_TESTS=ON
cmake --build tests/cpp/build -j
```

## Notes

- All tests require building in Docker with proper GPU support
- Tests are built automatically in the Docker image
- Some tests (XtGemm, Zerocopy) require additional dependencies and are disabled by default
