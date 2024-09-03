cmake_minimum_required(VERSION 3.20)

# Set the project name
project(morphling)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Set c++ 17
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build tests" ON)

# Disable other tests
set(BUILD_TESTING OFF)
set(ENABLE_TESTS OFF)

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

# Specify the directory where the UMPIRE project will be cloned and built
set(EXTERNAL_DIR ${CMAKE_BINARY_DIR}/external)
set(UMPIRE_REPO_URL https://github.com/LLNL/Umpire.git)
set(UMPIRE_REPO_TAG v2024.07.0)

clone_and_build_external_project(${EXTERNAL_DIR}
  umpire
  ${UMPIRE_REPO_URL}
  ${UMPIRE_REPO_TAG}
  -DENABLE_TESTS=OFF -DENABLE_CUDA=ON -DENABLE_OPENMP=ON -DENABLE_MPI=ON
  -DUMPIRE_ENABLE_IPC_SHARED_MEMORY=ON -DUMPIRE_ENABLE_ASAN=ON -DUMPIRE_ENABLE_NUMA=ON -DUMPIRE_ENABLE_LOGGING=ON
  -DBLT_CXX_STD=c++17 -DCMAKE_CUDA_STANDARD=17 -DUMPIRE_ENABLE_GMOCK=OFF -DUMPIRE_ENABLE_FORTRAN=OFF
)
set(UMPIRE_INCLUDE_DIRS ${EXTERNAL_DIR}/umpire/src ${EXTERNAL_DIR}/umpire/build/include)
set(UMPIRE_LIBRARY_DIRS ${EXTERNAL_DIR}/umpire/build/lib)

# BLT Support Flags
set(ENABLE_CUDA ON)
set(ENABLE_OPENMP ON)
set(ENABLE_MPI ON)
set(ENABLE_HIP OFF)

# Umpire Support Flags
set(UMPIRE_ENABLE_IPC_SHARED_MEMORY ON)
set(UMPIRE_ENABLE_ASAN ON)
set(UMPIRE_ENABLE_NUMA ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG v0.7.1
)
FetchContent_MakeAvailable(glog)

if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX_PATH $ENV{CONDA_PREFIX})
  message(STATUS "Conda environment path: ${CONDA_PREFIX_PATH}")
else()
  message(WARNING "CONDA_PREFIX is not set. Make sure your Conda environment is activated.")
endif()

set(PYTHON_EXECUTABLE ${CONDA_PREFIX_PATH}/bin/python)

# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

if(PYTHON_EXECUTABLE)
  find_python_from_executable(${PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

find_package(PythonInterp REQUIRED)

# Retrieve the Python version
message(STATUS "Python version: ${PYTHON_VERSION_STRING}")

# Retrieve the Python major and minor version
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\..*" _ ${PYTHON_VERSION_STRING})
set(Python_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
message(STATUS "Python version: ${Python_VERSION}")

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

# file(GLOB SOURCES "csrc/morphling/*.cpp")
# add_library(morphling SHARED ${SOURCES})
# target_link_libraries(morphling PUBLIC CUDA::cudart Threads::Threads glog::glog)
# target_include_directories(morphling PUBLIC "csrc/morphling" ${CUDA_INCLUDE_DIRS})

# add_executable(morphling_server "csrc/server.cpp" ${hw_proto_srcs} ${hw_grpc_srcs})
# target_link_libraries(morphling_server PRIVATE morphling grpc++ grpc++_reflection libprotobuf gflags::gflags glog::glog)
# target_include_directories(morphling_server PRIVATE "csrc/morphling")

# # Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/CMakeLists.txt
# include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

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
file(GLOB CKPT_SOURCES "csrc/checkpoint/*.cpp")
set(MORPHLING_EXT_SRC
  ${CKPT_SOURCES}
  "csrc/utils/logger.cpp"
  "csrc/utils/cuda_utils.cpp"
)

define_gpu_extension_target(
  _C
  DESTINATION morphling
  LANGUAGE ${MORPHLING_GPU_LANG}
  SOURCES ${MORPHLING_EXT_SRC}
  COMPILE_FLAGS ${MORPHLING_GPU_FLAGS}
  ARCHITECTURES ${MORPHLING_GPU_ARCHES}
  WITH_SOABI
)

# Enable testing if the option is ON
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()