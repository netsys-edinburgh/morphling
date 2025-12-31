# include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/tests.cmake)

# # Manually specify the Python executable
# set(Python_EXECUTABLE ${CONDA_PREFIX_PATH}/bin/python)
# find_package(PythonInterp REQUIRED)
# # Retrieve the Python version
# message(STATUS "Python version: ${PYTHON_VERSION_STRING}")

# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# add_definitions(-DTorch_DIR=${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/torch/share/cmake/Torch/)

# message(STATUS "-DTorch_DIR: ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/torch/share/cmake/")

# # Add the test files, create one executable for each test file, start with "test_*"
# find_package(Threads REQUIRED)
# find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

# find_package(MKL REQUIRED)
find_package(CUDAToolkit REQUIRED)
# find_package(MPI QUIET)

set(CMAKE_SKIP_RPATH ON)

# find all files in the directory match with "test_*"
file(GLOB TEST_SOURCES "test_*.cpp")

# Define source file globs for compilation
file(GLOB NETWORK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/network/*.cc")
file(GLOB BASE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/base/*.cc")
file(GLOB TIMER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/timer/*.cc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -std=c++17 -ggdb3 -fno-omit-frame-pointer")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

include_directories(
    ${EXTERNAL_DIR}/umpire/src/tpl/umpire/camp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${EXTERNAL_DIR}/umpire/src/tpl
    ${EXTERNAL_DIR}/umpire/build/src/tpl/umpire/camp/include
)

link_directories(
    ${CONDA_PREFIX_PATH}/lib
    ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/torch/lib
)

message(STATUS "torch_python_LIBRARY: ${torch_python_LIBRARY}")

# ============================================================================
# Simple memory copy performance test - no heavy dependencies
# ============================================================================
add_executable(test_pin_thread_perf test_pin_thread_perf.cpp)
set_target_properties(test_pin_thread_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME test_pin_thread_perf COMMAND test_pin_thread_perf)