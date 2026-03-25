cmake_minimum_required(VERSION 3.20)

# Set the project name
project(emulator)

# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/CMakeLists.txt
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# ============================================================================
# Compiler Optimization Flags
# ============================================================================
# Performance optimization flags for memory-intensive operations
# These flags are applied to all targets in the project
set(OPTIMIZATION_FLAGS "-O3 -march=native -mtune=native -funroll-loops -mavx -mavx2")

# Apply optimization flags to C and CXX
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPTIMIZATION_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS} -g3 -Wall")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

option(BUILD_TESTS "Build tests" OFF)

# Disable other tests
set(BUILD_TESTING OFF)
set(ENABLE_TESTS OFF)

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

set(BUILD_UNIT_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_SHARED_LIBS ON)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/protobuf/cmake)
set(Protobuf_PROTOC_EXECUTABLE $<TARGET_FILE:protoc>)
set(Protobuf_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/external/protobuf/src")
set(Protobuf_LIBRARIES protobuf::libprotobuf)
include(FindProtobuf)

add_subdirectory(external/rabbitmq-c)
add_subdirectory(external/redis-plus-plus)

if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX_PATH $ENV{CONDA_PREFIX})
  message(STATUS "Conda environment path: ${CONDA_PREFIX_PATH}")
else()
  message(WARNING "CONDA_PREFIX is not set. Make sure your Conda environment is activated.")
endif()

# find hiredis
# find_package(hiredis REQUIRED)

# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11" "3.12")

if(MORPHLING_PYTHON_EXECUTABLE)
  find_python_from_executable(${MORPHLING_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

# find_package(Protobuf REQUIRED HINTS ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/torch/share/cmake/Caffe2/public)
find_package(Python COMPONENTS Development REQUIRED)
# find_package(gRPC CONFIG REQUIRED)
# find_package(gflags REQUIRED)

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

#
# Normally `torch.utils.cpp_extension.CUDAExtension` would add
# `libtorch_python.so` for linking against an extension. Torch's cmake
# configuration does not include this library (presumably since the cmake
# config is used for standalone C++ binaries that link against torch).
# The `libtorch_python.so` library defines some of the glue code between
# torch/python via pybind and is required by MORPHLING extensions for this
# reason. So, add it by manually with `find_library` using torch's
# installed library path.
#
find_library(torch_python_LIBRARY torch_python PATHS
  "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "torch_python_LIBRARY: ${torch_python_LIBRARY}")

# Find the protoc executable from Protobuf
set(_PROTOBUF_PROTOC "${Protobuf_PROTOC_EXECUTABLE}")

# Proto files
set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_FILES
  "${PROTO_SRC_DIR}/morphling.proto"
  "${PROTO_SRC_DIR}/global_api.proto"
)

protobuf_generate(
  LANGUAGE cpp
  OUT_VAR PROTO_SRCS
  PROTOS ${PROTO_FILES}
  PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  IMPORT_DIRS "${PROTO_SRC_DIR}"
)
protobuf_generate(
  LANGUAGE python
  OUT_VAR PROTO_PY
  PROTOS ${PROTO_FILES}
  PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  IMPORT_DIRS "${PROTO_SRC_DIR}"
)

message(STATUS "_PROTOBUF_PROTOC: ${_PROTOBUF_PROTOC}")

message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")

add_custom_target(
  generate_proto_files ALL
  DEPENDS ${PROTO_SRCS} ${PROTO_PY}
)

# Include generated *.pb.h files
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/external
  ${Python3_INCLUDE_DIRS}
  ${absl_SOURCE_DIR}
  ${Protobuf_INCLUDE_DIRS}
  /usr/local/include
)

file(GLOB INTERCEPT_SRC "csrc/intercept/*.cpp")
file(GLOB MEMORY_SRC "csrc/memory/*.cpp")
file(GLOB CORE_SRC "csrc/core/*.cpp")
file(GLOB CHECKPOINT_SRC "csrc/checkpoint/*.cpp")
file(GLOB SCHEDULER_SRC "csrc/scheduler/*.cpp")
file(GLOB PYTHON_SRC "csrc/python/*.cpp")
file(GLOB BACKEND_SRC "csrc/backend/*.cpp")
file(GLOB NETWORK_SRC "csrc/network/*.cpp")
file(GLOB BASE_SRC "external/muduo_base/*.cc")
file(GLOB TIMER_SRC "csrc/timer/*.cpp")

message(STATUS "Protobuf_LIBRARIES: ${Protobuf_LIBRARIES}")



# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

set(MORPHLING_GPU_LANG "cuda")

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `MORPHLING_GPU_ARCHES`.
#
override_gpu_arches(MORPHLING_GPU_ARCHES
  ${MORPHLING_GPU_LANG}
  "${${MORPHLING_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Query torch for additional GPU compilation flags for the given
# `MORPHLING_GPU_LANG`.
# The final set of arches is stored in `MORPHLING_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(MORPHLING_GPU_FLAGS ${MORPHLING_GPU_LANG})

#
# _C extension
#
set(MORPHLING_EXT_SRC
  ${CORE_SRC}
  ${BASE_SRC}
  ${TIMER_SRC}
  ${INTERCEPT_SRC}
  ${PROTO_SRCS}
  "csrc/checkpoint/archer_tensor_handle.cpp"
  "csrc/memory/shared_memory.cpp"
  "csrc/python/checkpoint_py.cpp"
)
set(MORPHLING_EXT_LIB
  # ${TORCH_LIBRARIES}
  torch ${torch_python_LIBRARY}
  uuid Threads::Threads fmt
  ${Protobuf_LIBRARIES}
  rabbitmq
  CUDA::cublas
)

message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
define_gpu_extension_target(
  _C
  DESTINATION morphling
  LANGUAGE ${MORPHLING_GPU_LANG}
  SOURCES ${MORPHLING_EXT_SRC}
  COMPILE_FLAGS ${MORPHLING_GPU_FLAGS}
  ARCHITECTURES ${MORPHLING_GPU_ARCHES}
  LIBRARIES ${MORPHLING_EXT_LIB}
  # INCLUDE_DIRECTORIES ${MORPHLING_EXT_INCLUDE_DIRS}
  WITH_SOABI
)


set(MORPHLING_MSG_SRC
  ${BACKEND_SRC}
  ${NETWORK_SRC}
  ${BASE_SRC}
  ${TIMER_SRC}
  ${CORE_SRC}
  ${PROTO_SRCS}
  "csrc/scheduler/amqp_dispatcher.cpp"
  "csrc/scheduler/amqp_worker.cpp"
  "csrc/scheduler/cpu_worker.cpp"
  "csrc/scheduler/gpu_worker.cpp"
  "csrc/python/client_py.cpp"
  "csrc/memory/caching_allocator.cpp"
  "csrc/memory/shared_memory.cpp"
)
set(MORPHLING_EXT_LIB
  torch ${torch_python_LIBRARY}
  uuid Threads::Threads fmt
  rabbitmq mosquitto
  ${Protobuf_LIBRARIES}
  redis++
  hiredis
  event
  openblas
  CUDA::cublas
  CUDA::cuda_driver
)

define_gpu_extension_target(
  _Msg
  DESTINATION morphling
  LANGUAGE ${MORPHLING_GPU_LANG}
  SOURCES ${MORPHLING_MSG_SRC}
  COMPILE_FLAGS ${MORPHLING_GPU_FLAGS}
  ARCHITECTURES ${MORPHLING_GPU_ARCHES}
  LIBRARIES ${MORPHLING_EXT_LIB}
  # INCLUDE_DIRECTORIES ${MORPHLING_EXT_INCLUDE_DIRS}
  WITH_SOABI
)

#
# _GreenCtx extension (green context runtime for dynamic SM partitioning)
#
set(MORPHLING_GREENCTX_SRC
  ${BASE_SRC}
  "csrc/scheduler/green_context_runtime.cpp"
  "csrc/scheduler/green_trace_parser.cpp"
  "csrc/python/green_ctx_py.cpp"
  "csrc/core/logger.cpp"
  "csrc/core/cuda_utils.cpp"
)
set(MORPHLING_GREENCTX_LIB
  torch ${torch_python_LIBRARY}
  uuid Threads::Threads fmt
  CUDA::cuda_driver
)

define_gpu_extension_target(
  _GreenCtx
  DESTINATION morphling
  LANGUAGE ${MORPHLING_GPU_LANG}
  SOURCES ${MORPHLING_GREENCTX_SRC}
  COMPILE_FLAGS ${MORPHLING_GPU_FLAGS}
  ARCHITECTURES ${MORPHLING_GPU_ARCHES}
  LIBRARIES ${MORPHLING_GREENCTX_LIB}
  WITH_SOABI
)

# Enable testing if the option is ON
message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
