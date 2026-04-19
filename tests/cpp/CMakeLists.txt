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
# XtGemm Worker tests (require CUDA driver 12.5+)
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
    file(GLOB BASE_LOGGING_SRC ${PROJECT_ROOT}/external/muduo_base/*.cc)

    # Source dependencies for XtGemm tests
    set(XTGEMM_TEST_DEPS
      ${PROJECT_ROOT}/csrc/scheduler/gpu_worker.cpp
      ${PROJECT_ROOT}/csrc/memory/caching_allocator.cpp
      ${PROJECT_ROOT}/csrc/core/cuda_utils.cpp
      ${PROJECT_ROOT}/csrc/core/logger.cpp
      ${PROJECT_ROOT}/csrc/core/generator.cpp
      ${PROJECT_ROOT}/csrc/timer/virtual_clock.cpp
      ${BASE_LOGGING_SRC}
    )

    set(XTGEMM_INCLUDE_DIRS
      ${PROJECT_ROOT}/csrc
      ${PROJECT_ROOT}/external
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
      ${CMAKE_CURRENT_SOURCE_DIR}/bench/cuda
    )
    target_link_libraries(bench_green_ctx PRIVATE
      benchmark::benchmark
      benchmark::benchmark_main
      ${XTGEMM_COMMON_LIBS}
      CUDA::cublas
      CUDA::cuda_driver
    )

    # Trace-switch benchmark (bench/cuda/)
    add_executable(bench_trace_switch
      bench/cuda/bench_trace_switch.cpp
      ${XTGEMM_TEST_DEPS}
    )
    target_include_directories(bench_trace_switch PRIVATE
      ${XTGEMM_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/bench/cuda
    )
    target_link_libraries(bench_trace_switch PRIVATE
      benchmark::benchmark
      benchmark::benchmark_main
      ${XTGEMM_COMMON_LIBS}
      CUDA::cublas
      CUDA::cuda_driver
     )

    # Pool dispatch benchmarks (GPU vs CPU vs Hybrid) (bench/cuda/)
    set(POOL_DISPATCH_DEPS
      ${XTGEMM_TEST_DEPS}
      ${PROJECT_ROOT}/csrc/scheduler/cpu_worker.cpp
    )

    add_executable(bench_pool_dispatch
      bench/cuda/bench_pool_dispatch.cpp
      ${POOL_DISPATCH_DEPS}
    )
    target_include_directories(bench_pool_dispatch PRIVATE
      ${XTGEMM_INCLUDE_DIRS}
    )
    target_link_libraries(bench_pool_dispatch PRIVATE
      benchmark::benchmark
      benchmark::benchmark_main
      ${XTGEMM_COMMON_LIBS}
      openblas
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

# ============================================================================
# DispatchGate / DeviceTracker tests (unit/backend + integration)
# ============================================================================
set(PROJECT_ROOT_DG ${CMAKE_CURRENT_SOURCE_DIR}/../..)

file(GLOB DG_BASE_LOGGING_SRC ${PROJECT_ROOT_DG}/external/muduo_base/*.cc)

add_executable(test_dispatch_gate
  unit/backend/test_dispatch_gate.cpp
  ${PROJECT_ROOT_DG}/csrc/backend/dispatch_gate.cpp
  ${PROJECT_ROOT_DG}/csrc/core/logger.cpp
  ${DG_BASE_LOGGING_SRC}
)
target_include_directories(test_dispatch_gate PRIVATE
  ${PROJECT_ROOT_DG}/csrc
  ${PROJECT_ROOT_DG}/external
)
target_link_libraries(test_dispatch_gate PRIVATE
  GTest::gtest_main
  GTest::gtest
  Threads::Threads
  uuid
)
add_test(NAME test_dispatch_gate COMMAND test_dispatch_gate)

add_executable(test_barrier_integration
  integration/test_barrier_integration.cpp
  ${PROJECT_ROOT_DG}/csrc/backend/dispatch_gate.cpp
  ${DG_BASE_LOGGING_SRC}
)
target_include_directories(test_barrier_integration PRIVATE
  ${PROJECT_ROOT_DG}/csrc
  ${PROJECT_ROOT_DG}/external
)
target_link_libraries(test_barrier_integration PRIVATE
  GTest::gtest
  GTest::gtest_main
  Threads::Threads
  uuid
)
add_test(NAME test_barrier_integration COMMAND test_barrier_integration)

# ==========================================================================
# ProxyEnvCfg transport_mode tests (unit/config/)
# ==========================================================================
set(PROJECT_ROOT_CFG ${CMAKE_CURRENT_SOURCE_DIR}/../..)

file(GLOB CFG_BASE_LOGGING_SRC ${PROJECT_ROOT_CFG}/external/muduo_base/*.cc)

add_executable(test_transport_mode
  unit/config/test_transport_mode.cpp
  ${PROJECT_ROOT_CFG}/csrc/core/env_cfg.cpp
  ${CFG_BASE_LOGGING_SRC}
)
target_include_directories(test_transport_mode PRIVATE
  ${PROJECT_ROOT_CFG}/csrc
  ${PROJECT_ROOT_CFG}/external
  ${CMAKE_CURRENT_SOURCE_DIR}/stubs
)
if(CUDAToolkit_FOUND)
  target_include_directories(test_transport_mode PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
  )
endif()
target_link_libraries(test_transport_mode PRIVATE
  GTest::gtest
  GTest::gtest_main
  Threads::Threads
)
add_test(NAME test_transport_mode COMMAND test_transport_mode)

# ==========================================================================
# ShmAttachmentManager tests (unit/memory/)
# ==========================================================================
set(PROJECT_ROOT_MEM ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(test_shm_attachment_manager
  unit/memory/test_shm_attachment_manager.cpp
  ${PROJECT_ROOT_MEM}/csrc/memory/shm_attachment_manager.cpp
  ${PROJECT_ROOT_MEM}/csrc/memory/shared_memory.cpp
  ${PROJECT_ROOT_MEM}/csrc/core/logger.cpp
  ${PROJECT_ROOT_MEM}/external/muduo_base/log_stream.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/logging.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/log_file.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/date.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/timestamp.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/process_info.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/thread.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/thread_pool.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/condition.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/countdown_latch.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/file_util.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/exception.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/ini_config.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/slice.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/my_uuid.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/async_logging.cc
  ${PROJECT_ROOT_MEM}/external/muduo_base/timezone.cc
)
target_include_directories(test_shm_attachment_manager PRIVATE
  ${PROJECT_ROOT_MEM}/csrc
  ${PROJECT_ROOT_MEM}/external
)
target_link_libraries(test_shm_attachment_manager PRIVATE
  GTest::gtest
  GTest::gtest_main
  Threads::Threads
  uuid
)
add_test(NAME test_shm_attachment_manager COMMAND test_shm_attachment_manager)

# ==========================================================================
# GreenTraceParser tests (unit/worker/) — no CUDA required
# ============================================================================
set(PROJECT_ROOT_GTP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

file(GLOB GTP_BASE_LOGGING_SRC ${PROJECT_ROOT_GTP}/external/muduo_base/*.cc)

add_executable(test_green_trace_parser
  unit/worker/test_green_trace_parser.cpp
  ${PROJECT_ROOT_GTP}/csrc/scheduler/green_trace_parser.cpp
  ${PROJECT_ROOT_GTP}/csrc/core/logger.cpp
  ${GTP_BASE_LOGGING_SRC}
)
target_include_directories(test_green_trace_parser PRIVATE
  ${PROJECT_ROOT_GTP}/csrc
  ${PROJECT_ROOT_GTP}/external
)
target_link_libraries(test_green_trace_parser PRIVATE
  GTest::gtest
  GTest::gtest_main
  Threads::Threads
  uuid
)
add_test(NAME test_green_trace_parser COMMAND test_green_trace_parser)

# ==========================================================================
# GreenContextRuntime tests (unit/worker/) — requires CUDA driver 12.5+
# ============================================================================
option(ENABLE_GREEN_CTX_TESTS "Build green context runtime tests" OFF)
if(ENABLE_GREEN_CTX_TESTS)
  find_package(CUDAToolkit QUIET)
  find_package(Threads REQUIRED)

  if(CUDAToolkit_FOUND AND TARGET CUDA::cuda_driver)
    set(PROJECT_ROOT_GCR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
    file(GLOB GCR_BASE_LOGGING_SRC ${PROJECT_ROOT_GCR}/external/muduo_base/*.cc)

    set(GCR_SOURCES
      ${PROJECT_ROOT_GCR}/csrc/scheduler/green_context_runtime.cpp
      ${PROJECT_ROOT_GCR}/csrc/scheduler/green_trace_parser.cpp
      ${PROJECT_ROOT_GCR}/csrc/core/logger.cpp
      ${PROJECT_ROOT_GCR}/csrc/core/cuda_utils.cpp
      ${GCR_BASE_LOGGING_SRC}
    )

    add_executable(test_green_context_runtime
      unit/worker/test_green_context_runtime.cpp
      ${GCR_SOURCES}
    )
    target_include_directories(test_green_context_runtime PRIVATE
      ${PROJECT_ROOT_GCR}/csrc
      ${PROJECT_ROOT_GCR}/external
    )
    target_link_libraries(test_green_context_runtime PRIVATE
      GTest::gtest
      GTest::gtest_main
      CUDA::cudart
      CUDA::cuda_driver
      CUDA::cublas
      Threads::Threads
      uuid
    )
    add_test(NAME test_green_context_runtime COMMAND test_green_context_runtime)
  else()
    message(WARNING "CUDAToolkit not found; green context runtime tests skipped.")
  endif()
endif()

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

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/zerocopy/CMakeLists.txt")
    add_subdirectory(unit/zerocopy)
  else()
    message(WARNING "unit/zerocopy/CMakeLists.txt not found; zerocopy unit tests skipped.")
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/zerocopy/CMakeLists.txt")
    add_subdirectory(bench/zerocopy)
  else()
    message(WARNING "bench/zerocopy/CMakeLists.txt not found; zerocopy benchmarks skipped.")
  endif()
endif()
