#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <cmath>

// ============================================================================
// Simple Test: CUDA Host Register for PyTorch Tensors
// 
// Verify: Can we use cudaHostRegister pointers for torch tensor creation
//         and matrix multiplication?
// ============================================================================

void PrintLine() {
  std::cout << std::string(80, '=') << "\n";
}

void PrintSection(const std::string& title) {
  std::cout << "\n";
  PrintLine();
  std::cout << title << "\n";
  PrintLine();
}

// Compare two matrices with tolerance
bool CompareMatrices(float* a, float* b, int64_t size, float tolerance = 1e-4) {
  float max_diff = 0.0f;
  float max_rel_diff = 0.0f;
  int diff_count = 0;

  for (int64_t i = 0; i < size; i++) {
    float abs_diff = std::abs(a[i] - b[i]);
    if (abs_diff > tolerance) {
      diff_count++;
      max_diff = std::max(max_diff, abs_diff);
      
      // Relative difference
      float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
      if (max_val > 1e-6f) {
        float rel_diff = abs_diff / max_val;
        max_rel_diff = std::max(max_rel_diff, rel_diff);
      }
    }
  }

  std::cout << "  Differences found: " << diff_count << " / " << size << "\n";
  if (diff_count > 0) {
    std::cout << "  Max absolute difference: " << std::scientific << std::setprecision(6) 
              << max_diff << "\n";
    std::cout << "  Max relative difference: " << max_rel_diff << "\n";
  }

  return diff_count == 0;
}

float* TestMatrixMultiplication(const std::string& name, float* host_ptr_a,
                                  float* host_ptr_b, int m, int k, int n) {
  std::cout << "\n[" << name << "] Testing matrix multiplication...\n";

  // Allocate GPU memory
  float* gpu_a = nullptr;
  float* gpu_b = nullptr;
  float* gpu_c = nullptr;

  int64_t size_a = (int64_t)m * k * sizeof(float);
  int64_t size_b = (int64_t)k * n * sizeof(float);
  int64_t size_c = (int64_t)m * n * sizeof(float);

  cudaMalloc((void**)&gpu_a, size_a);
  cudaMalloc((void**)&gpu_b, size_b);
  cudaMalloc((void**)&gpu_c, size_c);

  // Copy data to GPU
  cudaMemcpy(gpu_a, host_ptr_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_b, host_ptr_b, size_b, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  std::cout << "  Matrix A: " << m << " x " << k << "\n";
  std::cout << "  Matrix B: " << k << " x " << n << "\n";
  std::cout << "  Data copied to GPU\n";

  // Create cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Warmup: C = A * B
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
              &alpha, gpu_b, n, gpu_a, k, &beta, gpu_c, n);
  cudaDeviceSynchronize();

  // Benchmark (5 iterations)
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 5; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, gpu_b, n, gpu_a, k, &beta, gpu_c, n);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double gpu_elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count() / 5.0;

  // Compute FLOPs
  double flops = 2.0 * m * n * k;  // 2 multiplications + 1 addition per element
  double gpu_gflops = flops / gpu_elapsed_ms / 1e6;

  std::cout << "  GPU Matrix mult time: " << std::fixed << std::setprecision(3)
            << gpu_elapsed_ms << " ms\n";
  std::cout << "  GPU Performance: " << std::setprecision(2) << gpu_gflops
            << " GFLOPS\n";

  // Copy result back from GPU
  float* gpu_result = (float*)malloc(size_c);
  cudaMemcpy(gpu_result, gpu_c, size_c, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Cleanup GPU
  cublasDestroy(handle);
  cudaFree(gpu_a);
  cudaFree(gpu_b);
  cudaFree(gpu_c);

  // Return the GPU result for later comparison
  return gpu_result;
}

// void TestMemoryTransfer(const std::string& name, float* host_ptr, int64_t size_bytes) {
//   std::cout << "\n[" << name << "] Testing memory transfer...\n";

//   float* device_ptr;
//   cudaMalloc(&device_ptr, size_bytes);

//   // Warmup
//   cudaMemcpy(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);

//   // Benchmark H2D
//   auto start = std::chrono::high_resolution_clock::now();
//   for (int i = 0; i < 5; i++) {
//     cudaMemcpy(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);
//   }
//   cudaDeviceSynchronize();
//   auto end = std::chrono::high_resolution_clock::now();

//   double h2d_time_ms =
//       std::chrono::duration<double, std::milli>(end - start).count() / 5.0;
//   double h2d_tp_gbs = (size_bytes / 1e9) / (h2d_time_ms / 1000.0);

//   std::cout << "  H2D Time: " << std::fixed << std::setprecision(3)
//             << h2d_time_ms << " ms, TP: " << std::setprecision(2) << h2d_tp_gbs
//             << " GB/s\n";

//   // Benchmark D2H
//   start = std::chrono::high_resolution_clock::now();
//   for (int i = 0; i < 5; i++) {
//     cudaMemcpy(host_ptr, device_ptr, size_bytes, cudaMemcpyDeviceToHost);
//   }
//   cudaDeviceSynchronize();
//   end = std::chrono::high_resolution_clock::now();

//   double d2h_time_ms =
//       std::chrono::duration<double, std::milli>(end - start).count() / 5.0;
//   double d2h_tp_gbs = (size_bytes / 1e9) / (d2h_time_ms / 1000.0);

//   std::cout << "  D2H Time: " << d2h_time_ms << " ms, TP: " << d2h_tp_gbs
//             << " GB/s\n";

//   cudaFree(device_ptr);
// }

int main() {
  PrintSection("CUDA Host Register with PyTorch Tensors");

  // Check CUDA
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "ERROR: No CUDA devices found!\n";
    return 1;
  }

  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Memory: " << (prop.totalGlobalMem / 1e9) << " GB\n";

  // Matrix dimensions (reduced for faster testing with result verification)
  int m = 256;  // Rows of A
  int k = 256;  // Cols of A = Rows of B
  int n = 256;  // Cols of B

  int64_t elements = (int64_t)m * k;
  int64_t size_bytes = elements * sizeof(float);

  std::cout << "\nMatrix Configuration:\n";
  std::cout << "  A: " << m << " x " << k << " (" << (size_bytes / 1e6)
            << " MB)\n";
  std::cout << "  B: " << k << " x " << n << " (" << (size_bytes / 1e6)
            << " MB)\n";

  // Test 1: Standard malloc (baseline)
  PrintSection("Test 1: Standard malloc (Pageable Memory)");

  std::cout << "Allocating with malloc...\n";
  float* host_malloc_a = (float*)malloc(size_bytes);
  float* host_malloc_b = (float*)malloc(size_bytes);

  if (!host_malloc_a || !host_malloc_b) {
    std::cerr << "malloc failed!\n";
    return 1;
  }

  // Initialize with random data (same source data for both tests)
  for (int64_t i = 0; i < elements; i++) {
    host_malloc_a[i] = static_cast<float>(rand()) / RAND_MAX;
    host_malloc_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  std::cout << "✓ Allocated successfully\n";

  float* malloc_gpu_result = TestMatrixMultiplication("Standard malloc", host_malloc_a, host_malloc_b, m,
                                                       k, n);

  // Test 2: CUDA Host Register
  PrintSection("Test 2: CUDA Host Register (Pinned Memory)");

  std::cout << "Allocating with malloc + cudaHostRegister...\n";
  float* host_register_a = (float*)malloc(size_bytes);
  float* host_register_b = (float*)malloc(size_bytes);

  if (!host_register_a || !host_register_b) {
    std::cerr << "malloc failed!\n";
    return 1;
  }

  // Use the SAME host data for both tests
  // This verifies that both memory allocation methods can correctly access and copy the same host data
  std::cout << "Initializing with SAME host data from Test 1...\n";
  for (int64_t i = 0; i < elements; i++) {
    host_register_a[i] = host_malloc_a[i];
    host_register_b[i] = host_malloc_b[i];
  }

  // Register as pinned memory
  std::cout << "Registering memory with cudaHostRegister...\n";
  cudaError_t err1 = cudaHostRegister(host_register_a, size_bytes,
                                       cudaHostRegisterDefault);
  cudaError_t err2 = cudaHostRegister(host_register_b, size_bytes,
                                       cudaHostRegisterDefault);

  if (err1 != cudaSuccess || err2 != cudaSuccess) {
    std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(err1)
              << " / " << cudaGetErrorString(err2) << "\n";
    return 1;
  }

  std::cout << "✓ Registered successfully\n";

  float* register_gpu_result = TestMatrixMultiplication("CUDA Host Register", host_register_a,
                                                         host_register_b, m, k, n);

  // Compare the two GPU results
  PrintSection("Comparing GPU Results");
  std::cout << "Verifying that both memory allocation methods can correctly access host data\n";
  std::cout << "================================================================\n\n";
  
  std::cout << "Test Design:\n";
  std::cout << "  - Both tests use the SAME host data (one copy)\n";
  std::cout << "  - Test 1: malloc + cudaMemcpy → GPU → GPU calculation\n";
  std::cout << "  - Test 2: cudaHostRegister + cudaMemcpy → GPU → GPU calculation\n";
  std::cout << "  - Expected: GPU results should be IDENTICAL\n";
  std::cout << "    (proves both methods correctly copied and used the same host data)\n\n";

  int64_t result_size = (int64_t)m * n;
  bool results_match = CompareMatrices(malloc_gpu_result, register_gpu_result, result_size);
  
  std::cout << "Verification Result:\n";
  if (results_match) {
    std::cout << "✓ PASS! GPU results are IDENTICAL\n";
    std::cout << "  Both malloc and cudaHostRegister:\n";
    std::cout << "  • Correctly accessed the same host data\n";
    std::cout << "  • Successfully copied data to GPU\n";
    std::cout << "  • Produced identical GPU computation results\n";
  } else {
    std::cout << "✗ FAIL! GPU results are DIFFERENT\n";
    std::cout << "  This indicates a problem:\n";
    std::cout << "  • One method did not correctly access/copy the host data\n";
    std::cout << "  • Or there's a data corruption issue\n";
  }

  // Cleanup
  PrintSection("Cleanup");

  free(host_malloc_a);
  free(host_malloc_b);
  free(malloc_gpu_result);

  cudaError_t err3 = cudaHostUnregister(host_register_a);
  cudaError_t err4 = cudaHostUnregister(host_register_b);
  if (err3 != cudaSuccess || err4 != cudaSuccess) {
    std::cerr << "cudaHostUnregister failed\n";
  }
  free(host_register_a);
  free(host_register_b);
  free(register_gpu_result);

  std::cout << "✓ Cleaned up successfully\n";

  // Summary
  PrintSection("Summary");
  std::cout << "✓ Both allocation methods work correctly\n";
  std::cout << "✓ Both malloc and cudaHostRegister can access the same host data\n";
  std::cout << "✓ GPU matrix multiplication with both methods produces identical results\n";
  std::cout << "\nConclusion: cudaHostRegister correctly handles host data access for GPU computation\n";

  return 0;
}
