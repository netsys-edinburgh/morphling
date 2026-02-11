cmake_minimum_required(VERSION 3.22)
project(DeviceEmulatorTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Google Test & Google Benchmark via FetchContent
# ============================================================================
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)

# Prevent benchmark from building its own tests
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest googlebenchmark)

# ==========================================================================
# CUDA / cuBLAS tests
# ==========================================================================
option(ENABLE_CUDA_TESTS "Build CUDA/cuBLAS tests" ON)
if(ENABLE_CUDA_TESTS)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    add_executable(test_cublas_hostalloc_direct
      test_cublas_hostalloc_direct.cpp
    )
    target_link_libraries(test_cublas_hostalloc_direct
      CUDA::cudart
      CUDA::cublas
    )

    add_executable(test_cublas_hostregister_posix
      test_cublas_hostregister_posix.cpp
    )
    target_link_libraries(test_cublas_hostregister_posix
      CUDA::cudart
      CUDA::cublas
    )

    add_executable(test_cublas_error15_repro
      test_cublas_error15_repro.cpp
    )
    target_link_libraries(test_cublas_error15_repro
      GTest::gtest
      GTest::gtest_main
      CUDA::cudart
      CUDA::cublas
    )
    add_test(NAME test_cublas_error15_repro COMMAND test_cublas_error15_repro)
  else()
    message(WARNING "CUDAToolkit not found; CUDA tests will be skipped.")
  endif()
endif()

# ============================================================================
# Zerocopy test suite (optional)
# ============================================================================
option(ENABLE_ZEROCOPY_TESTS "Build zerocopy tests" OFF)
if(ENABLE_ZEROCOPY_TESTS)
  add_subdirectory(zerocopy)
endif()
