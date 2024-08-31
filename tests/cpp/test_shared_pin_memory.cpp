// test shm to pinned memory using cudahost register

#include <cuda_runtime_api.h>
#include <sys/shm.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#define MEM_SIZE 1024ULL * 1024 * 1024 * 4

int main() {
  // use shm to allocate memory
  int shm_id = shmget(IPC_PRIVATE, MEM_SIZE, IPC_CREAT | 0666);
  if (shm_id == -1) {
    std::cerr << "shmget failed" << std::endl;
    return 1;
  }

  void* shm_ptr = shmat(shm_id, nullptr, 0);
  if (shm_ptr == (void*)-1) {
    std::cerr << "shmat failed" << std::endl;
    return 1;
  }

  // use cudaHostRegister to pin memory
  cudaError_t err = cudaHostRegister(shm_ptr, MEM_SIZE, cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  void* pin_ptr;
  err = cudaHostAlloc(&pin_ptr, MEM_SIZE, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  // alloc cuda memory
  void* cuda_ptr;
  cudaSetDevice(1);
  err = cudaMalloc(&cuda_ptr, MEM_SIZE);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  // test memory copy speed using pinned memory
  std::vector<float> src(MEM_SIZE / sizeof(float));
  for (size_t i = 0; i < src.size(); i++) {
    src[i] = i;
  }
  cudaMemset(cuda_ptr, 0, MEM_SIZE);
  auto start = std::chrono::high_resolution_clock::now();
  err = cudaMemcpy(cuda_ptr, src.data(), MEM_SIZE, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  // convert to GB/s
  std::cout << "Bandwidth: " << MEM_SIZE / 1024 / 1024 / 1024 / elapsed.count()
            << " GB/s\n";

  // speed test of pin_ptr
  cudaMemset(cuda_ptr, 0, MEM_SIZE);
  start = std::chrono::high_resolution_clock::now();
  err = cudaMemcpy(cuda_ptr, pin_ptr, MEM_SIZE, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  // convert to GB/s
  std::cout << "Bandwidth: " << MEM_SIZE / 1024 / 1024 / 1024 / elapsed.count()
            << " GB/s\n";
  

  // free cuda memory
  err = cudaFree(cuda_ptr);
  if (err != cudaSuccess) {
    std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  // use cudaHostUnregister to unpin memory
  err = cudaHostUnregister(shm_ptr);
  if (err != cudaSuccess) {
    std::cerr << "cudaHostUnregister failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  // detach shm
  if (shmdt(shm_ptr) == -1) {
    std::cerr << "shmdt failed" << std::endl;
    return 1;
  }

  // remove shm
  if (shmctl(shm_id, IPC_RMID, nullptr) == -1) {
    std::cerr << "shmctl failed" << std::endl;
    return 1;
  }

  return 0;
}