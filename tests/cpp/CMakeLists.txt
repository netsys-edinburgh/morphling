# gtest make all files in the directory available to the test

project(tests)

# ignore the warnings of cmake
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)

# Retrieve the Conda environment path from the environment variable
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX_PATH $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment path: ${CONDA_PREFIX_PATH}")
else()
    message(WARNING "CONDA_PREFIX is not set. Make sure your Conda environment is activated.")
endif()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DTorch_DIR=${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/torch/share/cmake/Torch/)

# Manually specify the Python executable
set(Python_EXECUTABLE ${CONDA_PREFIX_PATH}/bin/python)
find_package(PythonInterp REQUIRED)
# Retrieve the Python version
message(STATUS "Python version: ${PYTHON_VERSION_STRING}")

# Retrieve the Python major and minor version
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\..*" _ ${PYTHON_VERSION_STRING})
set(Python_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
message(STATUS "Python version: ${Python_VERSION}")

# set(Py_VERSION ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
# message(STATUS "Python version: ${Py_VERSION}")

# Add the test files, create one executable for each test file, start with "test_*"
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

find_package(Torch REQUIRED)

# find all files in the directory match with "test_*"
file(GLOB TEST_SOURCES "test_*.cpp")
file(GLOB PROJECT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/memory/*.cpp")

# print CMAKE_ROOT_DIR, CMAKE_CURRENT_SOURCE_DIR
message(STATUS "CMAKE_ROOT_DIR: ${CMAKE_ROOT_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
# Retrieve the Python version
message(STATUS "Python version: ${Python_VERSION}")

# cpp flag add -m option to enable avx2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -D__AVX256__=1 -fopenmp -std=c++17")
# -mavx2 -march=znver3

# set(CMAKE_CUDA_COMPILER "g++-9")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE} ${PROJECT_SOURCES})
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TORCH_INCLUDE_DIRS}
        ${CONDA_PREFIX_PATH}/include
        ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/nvidia/cublas/include/
        ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/nvidia/cuda_runtime/include/
        ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/triton/backends/nvidia/include/
        /usr/local/cuda/include/
    )
    target_link_libraries(${TEST_NAME}
        pthread gtest gtest_main numa
        ${CONDA_PREFIX_PATH}/lib/libmkl_intel_lp64.so
        ${CONDA_PREFIX_PATH}/lib/libmkl_sequential.so
        ${CONDA_PREFIX_PATH}/lib/libmkl_core.so
        ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/nvidia/cublas/lib/libcublas.so.12
        ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
        ${TORCH_LIBRARIES}
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach(TEST_SOURCE)
