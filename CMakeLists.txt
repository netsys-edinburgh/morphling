cmake_minimum_required(VERSION 3.20)

# Set the project name
project(emulator)

# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/CMakeLists.txt
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0) # needed for torch backward compatibility

# Set c++ 17
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build tests" OFF)

# Disable other tests
set(BUILD_TESTING OFF)
set(ENABLE_TESTS OFF)

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

# # Specify the directory where the UMPIRE project will be cloned and built
# set(EXTERNAL_DIR ${CMAKE_BINARY_DIR}/external)
# set(UMPIRE_REPO_URL https://github.com/LLNL/Umpire.git)
# set(UMPIRE_REPO_TAG v2024.07.0)

# clone_and_build_external_project(${EXTERNAL_DIR}
#   umpire
#   ${UMPIRE_REPO_URL}
#   ${UMPIRE_REPO_TAG}
#   -DENABLE_TESTS=OFF -DENABLE_CUDA=ON -DENABLE_OPENMP=ON -DENABLE_MPI=ON
#   -DUMPIRE_ENABLE_IPC_SHARED_MEMORY=ON -DUMPIRE_ENABLE_ASAN=ON -DUMPIRE_ENABLE_NUMA=ON -DUMPIRE_ENABLE_LOGGING=ON
#   -DBLT_CXX_STD=c++17 -DCMAKE_CUDA_STANDARD=17 -DUMPIRE_ENABLE_GMOCK=OFF -DUMPIRE_ENABLE_FORTRAN=OFF
# )
# set(UMPIRE_INCLUDE_DIRS ${EXTERNAL_DIR}/umpire/src ${EXTERNAL_DIR}/umpire/build/include)
# set(UMPIRE_LIBRARY_DIRS ${EXTERNAL_DIR}/umpire/build/lib)

# RTTR Support Flags
set(BUILD_UNIT_TESTS OFF)
set(BUILD_EXAMPLES OFF)

# # BLT Support Flags
# set(ENABLE_CUDA ON)
# set(ENABLE_OPENMP ON)
# set(ENABLE_MPI ON)
# set(ENABLE_HIP OFF)

# # Umpire Support Flags
# set(UMPIRE_ENABLE_IPC_SHARED_MEMORY ON)
# set(UMPIRE_ENABLE_ASAN ON)
# set(UMPIRE_ENABLE_NUMA ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG v2.2.2
)

# FetchContent_Declare(
# glog
# GIT_REPOSITORY https://github.com/google/glog.git
# GIT_TAG v0.6.0
# )
FetchContent_Declare(
  rttr
  GIT_REPOSITORY https://github.com/rttrorg/rttr.git
  GIT_TAG v0.9.6
)
FetchContent_Declare(
  rapidjson
  GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
  GIT_TAG v1.1.0
)
FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG v1.48.4
)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.14.1
)
FetchContent_MakeAvailable(rttr rapidjson grpc spdlog gflags)

set(gRPC_INSTALL ON) # Install gRPC for TARGET_FILE

if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX_PATH $ENV{CONDA_PREFIX})
  message(STATUS "Conda environment path: ${CONDA_PREFIX_PATH}")
else()
  message(WARNING "CONDA_PREFIX is not set. Make sure your Conda environment is activated.")
endif()

# Retrieve the Python version
# message(STATUS "Python version: ${PYTHON_VERSION_STRING}")
find_package(PythonInterp REQUIRED)

# Retrieve the Python major and minor version
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\..*" _ ${PYTHON_VERSION_STRING})
set(Python_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
message(STATUS "Python version: ${Python_VERSION}")

# set(Python3_EXECUTABLE ${CONDA_PREFIX_PATH}/bin/python3)
# set(Python_INCLUDE_DIRS ${CONDA_PREFIX_PATH}/include)
# set(Python3_LIBRARIES ${CONDA_PREFIX_PATH}/lib)
set(CONDA_INCLUDE_DIRS ${CONDA_PREFIX_PATH}/include)
set(CONDA_LINK_DIRS ${CONDA_PREFIX_PATH}/lib)

# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

if(MORPHLING_PYTHON_EXECUTABLE)
  find_python_from_executable(${MORPHLING_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

# find_package(Protobuf REQUIRED HINTS ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/torch/share/cmake/Caffe2/public)
# find_package(Python COMPONENTS Development REQUIRED)
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

# Find the protoc executable
# find_program(_PROTOBUF_PROTOC protoc)
# find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

# Protobuf and gRPC libraries
# set(_PROTOBUF_LIBPROTOBUF libprotobuf)
# set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
# set(_GRPC_GRPCPP grpc++)
# set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

# Proto file
get_filename_component(hw_proto "proto/morphling.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/morphling.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/morphling.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/morphling.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/morphling.grpc.pb.h")

message(STATUS "_PROTOBUF_PROTOC: ${_PROTOBUF_PROTOC}")
message(STATUS "_GRPC_CPP_PLUGIN_EXECUTABLE: ${_GRPC_CPP_PLUGIN_EXECUTABLE}")

message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")

# # Path to .proto files
# set(PROTO_FILES ${CMAKE_SOURCE_DIR}/proto/morphling.proto)

# # Specify where the generated files will be located
# set(PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated)

# # Ensure the generated directory exists
# file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

# # Generate the protobuf (.pb.cc and .pb.h) and gRPC files (.grpc.pb.cc and .grpc.pb.h)
# foreach(proto_file ${PROTO_FILES})
# get_filename_component(proto_filename ${proto_file} NAME_WE)

# add_custom_command(
# OUTPUT ${PROTO_GEN_DIR}/${proto_filename}.pb.cc ${PROTO_GEN_DIR}/${proto_filename}.pb.h
# ${PROTO_GEN_DIR}/${proto_filename}.grpc.pb.cc ${PROTO_GEN_DIR}/${proto_filename}.grpc.pb.h
# COMMAND ${PROTOC_EXECUTABLE}
# ARGS --proto_path=${CMAKE_SOURCE_DIR}/proto
# --cpp_out=${PROTO_GEN_DIR}
# --grpc_out=${PROTO_GEN_DIR}
# --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
# ${proto_file}
# DEPENDS ${proto_file}
# COMMENT "Generating C++ code from ${proto_file}"
# )

# # Add the generated files to the list of sources
# list(APPEND PROTO_SRCS ${PROTO_GEN_DIR}/${proto_filename}.pb.cc)
# list(APPEND PROTO_SRCS ${PROTO_GEN_DIR}/${proto_filename}.grpc.pb.cc)
# endforeach()
add_custom_target(
  generate_proto_files ALL
  DEPENDS "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
)

add_custom_command(
  OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
  --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
  -I "${hw_proto_path}"
  --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
  "${hw_proto}"
  DEPENDS "${hw_proto}"
  VERBATIM
)

# Include generated *.pb.h files
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}

  ${CONDA_PREFIX_PATH}/include
  ${rttr_SOURCE_DIR}/src
  ${rttr_BINARY_DIR}/src
  ${absl_SOURCE_DIR}
  ${grpc_SOURCE_DIR}/include
  ${protobuf_SOURCE_DIR}/src
  ${spdlog_SOURCE_DIR}/include
)

link_directories(
  ${CONDA_PREFIX_PATH}/lib
)

file(GLOB INTERCEPT_SRC "csrc/intercept/*.cpp")
file(GLOB MEMORY_SRC "csrc/memory/*.cpp")
file(GLOB UTILS_SRC "csrc/utils/*.cpp")
file(GLOB CHECKPOINT_SRC "csrc/checkpoint/*.cpp")
file(GLOB SCHEDULER_SRC "csrc/scheduler/*.cpp")
file(GLOB PYTHON_SRC "csrc/python/*.cpp")

# add_library(morphling_interceptor SHARED ${INTERCEPT_SRC} ${UTILS_SRC} ${hw_proto_srcs} ${hw_grpc_srcs} "csrc/memory/caching_allocator.cpp")
# target_link_libraries(morphling_interceptor PUBLIC
#   CUDA::cudart Threads::Threads
#   libspdlog.so grpc++ grpc++_reflection
# )
# target_include_directories(morphling_interceptor PUBLIC "csrc" ${CUDA_INCLUDE_DIRS})

# add_library(morphling STATIC ${UTILS_SRC} ${CHECKPOINT_SRC})
# target_link_libraries(morphling PUBLIC
#   CUDA::cudart Threads::Threads
#   libspdlog.so ${TORCH_LIBRARIES}
# )
# target_include_directories(morphling PUBLIC "csrc" ${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

message(STATUS "rttr_SOURCE_DIR: ${rttr_SOURCE_DIR}")
message(STATUS "Protobuf_LIBRARIES: ${Protobuf_LIBRARIES}")

add_executable(morphling_server "csrc/server.cpp" ${hw_proto_srcs} ${hw_grpc_srcs}
  ${SCHEDULER_SRC} ${MEMORY_SRC} ${UTILS_SRC} ${INTERCEPT_SRC} ${CHECKPOINT_SRC}
)
add_dependencies(morphling_server generate_proto_files)
target_link_libraries(morphling_server PRIVATE
  gflags::gflags fmt
  ${_PROTOBUF_LIBPROTOBUF}
  grpc++ grpc++_reflection
  libspdlog.so
  rttr_core uuid
  CUDA::cublas CUDA::cudart
  libpython${Python_VERSION}.so
  ${TORCH_LIBRARIES}
)
target_include_directories(morphling_server PRIVATE "csrc" ${TORCH_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# add_executable(morphling_worker_server "csrc/worker_server.cpp" ${hw_proto_srcs} ${hw_grpc_srcs})
# add_dependencies(morphling_worker_server generate_proto_files)
# target_link_libraries(morphling_worker_server PRIVATE
#   gflags::gflags fmt
#   ${_PROTOBUF_LIBPROTOBUF}
#   grpc++ grpc++_reflection
#   libspdlog.so
#   uuid
# )
# target_include_directories(morphling_worker_server PRIVATE "csrc" ${TORCH_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})



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
  ${CHECKPOINT_SRC}
  ${MEMORY_SRC}
  ${UTILS_SRC}
  ${INTERCEPT_SRC}
  ${hw_proto_srcs}
  ${hw_grpc_srcs}
  "csrc/python/checkpoint_py.cpp"
)
set(MORPHLING_EXT_LIB
  torch ${torch_python_LIBRARY}
  uuid Threads::Threads fmt
  libspdlog.so rttr_core
  grpc++ grpc++_reflection
)
define_gpu_extension_target(
  _C
  DESTINATION morphling
  LANGUAGE ${MORPHLING_GPU_LANG}
  SOURCES ${MORPHLING_EXT_SRC}
  COMPILE_FLAGS ${MORPHLING_GPU_FLAGS}
  ARCHITECTURES ${MORPHLING_GPU_ARCHES}
  LIBRARIES ${MORPHLING_EXT_LIB}
  WITH_SOABI
)

# set(MORPHLING_INTERCEPTOR_SRC
# ${INTERCEPT_SRC}
# ${MEMORY_SRC}
# ${UTILS_SRC}
# ${hw_proto_srcs}
# ${hw_grpc_srcs}
# "csrc/python/intercept_py.cpp"
# )
# set(MORPHLING_INTERCEPTOR_LIB
# mkl_rt grpc++ grpc++_reflection rttr_core
# libspdlog.so uuid
# )
# define_gpu_extension_target(
# _intercept
# DESTINATION morphling
# LANGUAGE ${MORPHLING_GPU_LANG}
# SOURCES ${MORPHLING_INTERCEPTOR_SRC}
# COMPILE_FLAGS ${MORPHLING_GPU_FLAGS}
# ARCHITECTURES ${MORPHLING_GPU_ARCHES}
# LIBRARIES ${MORPHLING_INTERCEPTOR_LIB}
# WITH_SOABI
# )

# set(MORPHLING_ALLOCATOR_SRC
# ${MEMORY_SRC}
# ${UTILS_SRC}
# )
# set(MORPHLING_ALLOCATOR_LIB
# uuid Threads::Threads
# libspdlog.so
# )
# define_gpu_extension_target(
# morphling_allocator
# DESTINATION morphling
# LANGUAGE ${MORPHLING_GPU_LANG}
# SOURCES ${MORPHLING_ALLOCATOR_SRC}
# COMPILE_FLAGS ${MORPHLING_GPU_FLAGS}
# ARCHITECTURES ${MORPHLING_GPU_ARCHES}
# LIBRARIES ${MORPHLING_ALLOCATOR_LIB}
# WITH_SOABI
# )

# Enable testing if the option is ON
message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()