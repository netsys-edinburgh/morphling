# C++ Tests

This directory contains standalone C++ tests and benchmarks. Two CUDA/cuBLAS
tests validate whether host-allocated buffers can be used directly with
cuBLAS, and compare mapped vs. non-mapped host registration.

## CUDA/cuBLAS Tests

### 1) `test_cublas_hostalloc_direct`

Goal: Compare cuBLAS behavior when using:

- `cudaHostAllocDefault` (host pointer directly)
- `cudaHostAllocMapped` + `cudaHostGetDevicePointer` (mapped device pointer)

### 2) `test_cublas_hostregister_posix`

Goal: Compare cuBLAS behavior when using:

- `posix_memalign` (+ `mlock`) host pointers directly
- `cudaHostRegisterDefault` host pointers directly
- `cudaHostRegisterMapped` + `cudaHostGetDevicePointer`

Both tests print CUDA pointer attributes and the cuBLAS status so you can see
which pointer modes are accepted.

## Build and Run in Docker

These commands assume the Docker image is already built with the provided
`Dockerfile`, and you have a running container with GPU access.

### Option A: exec into existing container

```bash
sudo docker exec -it <container_id> bash -lc "\
  cmake -S tests/cpp -B tests/cpp/build && \
  cmake --build tests/cpp/build -j && \
  ./tests/cpp/build/test_cublas_hostalloc_direct && \
  ./tests/cpp/build/test_cublas_hostregister_posix"
```

### Option B: one-shot run

```bash
sudo docker run --rm --gpus=all --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v "$(pwd)":/app -w /app device-emulator:latest \
  bash -lc "\
    cmake -S tests/cpp -B tests/cpp/build && \
    cmake --build tests/cpp/build -j && \
    ./tests/cpp/build/test_cublas_hostalloc_direct && \
    ./tests/cpp/build/test_cublas_hostregister_posix"
```

## Notes

- Zerocopy tests are optional and controlled via:

```bash
cmake -S tests/cpp -B tests/cpp/build -DENABLE_ZEROCOPY_TESTS=ON
```

By default, zerocopy tests are off so CUDA tests can build independently
without `generate_proto_files`.

- CUDA tests are optional and default on. If CUDA toolkit libraries are
missing, CMake will skip them with a warning. You can force-disable via:

```bash
cmake -S tests/cpp -B tests/cpp/build -DENABLE_CUDA_TESTS=OFF
```
