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
  else()
    message(WARNING "CUDAToolkit not found; CUDA tests will be skipped.")
  endif()
endif()

# ============================================================================
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

  add_subdirectory(zerocopy)
endif()
