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
# CUDA / cuBLAS tests (unit/cuda/)
# ==========================================================================
option(ENABLE_CUDA_TESTS "Build CUDA/cuBLAS tests" ON)
if(ENABLE_CUDA_TESTS)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    add_executable(test_cublas_hostalloc_direct
      unit/cuda/test_cublas_hostalloc_direct.cpp
    )
    target_link_libraries(test_cublas_hostalloc_direct
      CUDA::cudart
      CUDA::cublas
    )

    add_executable(test_cublas_hostregister_posix
      unit/cuda/test_cublas_hostregister_posix.cpp
    )
    target_link_libraries(test_cublas_hostregister_posix
      CUDA::cudart
      CUDA::cublas
    )

    add_executable(test_cublas_error15_repro
      unit/cuda/test_cublas_error15_repro.cpp
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

# ==========================================================================
# XtGemm Worker tests (require CUDA green contexts, compute capability 9.0+)
# ============================================================================
option(ENABLE_XTGEMM_TESTS "Build XtGemm worker tests" OFF)
if(ENABLE_XTGEMM_TESTS)
  find_package(CUDAToolkit QUIET)
  find_package(Threads REQUIRED)

  if(CUDAToolkit_FOUND AND TARGET CUDA::cuda_driver)
    set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

    # Torch: add cmake prefix path
    execute_process(
      COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_PREFIX_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PREFIX_PATH})
    find_package(Torch REQUIRED)

    # Base logging sources
    file(GLOB BASE_LOGGING_SRC ${PROJECT_ROOT}/csrc/base/*.cc)

    # Source dependencies for XtGemm tests
    set(XTGEMM_TEST_DEPS
      ${PROJECT_ROOT}/csrc/scheduler/gpu_worker.cpp
      ${PROJECT_ROOT}/csrc/memory/caching_allocator.cpp
      ${PROJECT_ROOT}/csrc/utils/cuda_utils.cpp
      ${PROJECT_ROOT}/csrc/utils/logger.cpp
      ${PROJECT_ROOT}/csrc/common/generator.cc
      ${PROJECT_ROOT}/csrc/timer/virtual_clock.cc
      ${BASE_LOGGING_SRC}
    )

    set(XTGEMM_INCLUDE_DIRS
      ${PROJECT_ROOT}/csrc
      ${TORCH_INCLUDE_DIRS}
    )

    set(XTGEMM_COMMON_LIBS
      CUDA::cublas
      CUDA::cudart
      CUDA::cuda_driver
      ${TORCH_LIBRARIES}
      uuid
      Threads::Threads
    )

    # Correctness tests (unit/worker/)
    add_executable(test_xtgemm_worker
      unit/worker/test_xtgemm_worker.cpp
      ${XTGEMM_TEST_DEPS}
    )
    target_include_directories(test_xtgemm_worker PRIVATE
      ${XTGEMM_INCLUDE_DIRS}
    )
    target_link_libraries(test_xtgemm_worker PRIVATE
      GTest::gtest
      GTest::gtest_main
      ${XTGEMM_COMMON_LIBS}
    )
    add_test(NAME test_xtgemm_worker COMMAND test_xtgemm_worker)

    # Performance benchmarks (bench/cuda/)
    add_executable(bench_xtgemm_worker
      bench/cuda/bench_xtgemm_worker.cpp
      ${XTGEMM_TEST_DEPS}
    )
    target_include_directories(bench_xtgemm_worker PRIVATE
      ${XTGEMM_INCLUDE_DIRS}
    )
    target_link_libraries(bench_xtgemm_worker PRIVATE
      benchmark::benchmark
      benchmark::benchmark_main
      ${XTGEMM_COMMON_LIBS}
    )

    # Green Context benchmarks (bench/cuda/)
    add_executable(bench_green_ctx
      bench/cuda/bench_green_ctx.cpp
      ${XTGEMM_TEST_DEPS}
    )
    target_include_directories(bench_green_ctx PRIVATE
      ${XTGEMM_INCLUDE_DIRS}
    )
    target_link_libraries(bench_green_ctx PRIVATE
      benchmark::benchmark
      benchmark::benchmark_main
      ${XTGEMM_COMMON_LIBS}
    )
  else()
    message(WARNING "CUDAToolkit or CUDA driver API not found; XtGemm tests skipped.")
  endif()
endif()

# ==========================================================================
# WorkerBase / WorkerPool tests (unit/worker/)
# ============================================================================
find_package(Threads REQUIRED)

set(PROJECT_ROOT_WB ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(test_worker_base
  unit/worker/test_worker_base.cpp
)
target_include_directories(test_worker_base PRIVATE
  ${PROJECT_ROOT_WB}/csrc
)
target_link_libraries(test_worker_base PRIVATE
  GTest::gtest
  GTest::gtest_main
  Threads::Threads
)
add_test(NAME test_worker_base COMMAND test_worker_base)

# ==========================================================================
# Zerocopy test suite (optional)
# ============================================================================
option(ENABLE_ZEROCOPY_TESTS "Build zerocopy tests" OFF)
if(ENABLE_ZEROCOPY_TESTS)
  find_package(Threads REQUIRED)

  # Use vendored protobuf (same as main build) to avoid version mismatch
  # between system protoc and system libprotobuf-dev
  set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  # Vendored protobuf 3.13 requires cmake_minimum_required < 3.5 compat
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  add_subdirectory(${PROJECT_ROOT}/external/protobuf/cmake
                   ${CMAKE_BINARY_DIR}/protobuf)
  unset(CMAKE_POLICY_VERSION_MINIMUM)
  set(Protobuf_PROTOC_EXECUTABLE $<TARGET_FILE:protoc>)
  set(Protobuf_INCLUDE_DIRS "${PROJECT_ROOT}/external/protobuf/src")
  set(Protobuf_LIBRARIES protobuf::libprotobuf)
  include(FindProtobuf)

  # Torch: add its cmake prefix path so find_package works
  execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_PREFIX_PATH})
  find_package(Torch REQUIRED)

  # Generate proto C++ sources
  set(PROTO_SRC_DIR "${PROJECT_ROOT}/proto")
  set(PROTO_FILES
    "${PROTO_SRC_DIR}/morphling.proto"
    "${PROTO_SRC_DIR}/global_api.proto"
  )
  protobuf_generate(
    LANGUAGE cpp
    OUT_VAR PROTO_SRCS
    PROTOS ${PROTO_FILES}
    PROTOC_OUT_DIR "${CMAKE_BINARY_DIR}"
    IMPORT_DIRS "${PROTO_SRC_DIR}"
  )
  add_custom_target(generate_proto_files ALL DEPENDS ${PROTO_SRCS})

  add_subdirectory(unit/zerocopy)
  add_subdirectory(bench/zerocopy)
endif()
