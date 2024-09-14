# # Retrieve the Conda environment path from the environment variable
# if(DEFINED ENV{CONDA_PREFIX})
# set(CONDA_PREFIX_PATH $ENV{CONDA_PREFIX})
# message(STATUS "Conda environment path: ${CONDA_PREFIX_PATH}")
# else()
# message(WARNING "CONDA_PREFIX is not set. Make sure your Conda environment is activated.")
# endif()

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
find_package(MPI REQUIRED)

set(CMAKE_SKIP_RPATH ON)

# find all files in the directory match with "test_*"
file(GLOB TEST_SOURCES "test_*.cpp")

# # print CMAKE_ROOT_DIR, CMAKE_CURRENT_SOURCE_DIR
# message(STATUS "CMAKE_ROOT_DIR: ${CMAKE_ROOT_DIR}")
# message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
# message(STATUS "MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}")
# message(STATUS "MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}")
# # Retrieve the Python version
# message(STATUS "Python version: ${Python_VERSION}")

# cpp flag add -m option to enable avx2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -march=znver3 -O3 -D__AVX256__=1 -std=c++17")

# -mavx2 -march=znver3

# message(STATUS "UMPAIRE_INCLUDE_DIRS: ${UMPAIRE_INCLUDE_DIRS}")
# link_directories(${CONDA_PREFIX_PATH}/lib)
# set(CMAKE_BUILD_RPATH "/mnt/data/xly/.conda/envs/edgeml/lib:/usr/local/cuda-12.2/targets/x86_64-linux/lib")
# set(CMAKE_INSTALL_RPATH "/mnt/data/xly/.conda/envs/edgeml/lib:/usr/local/cuda-12.2/targets/x86_64-linux/lib")
include_directories(
    ${UMPIRE_INCLUDE_DIRS}
    ${EXTERNAL_DIR}/umpire/src/tpl/umpire/camp/include

    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    # ${EXTERNAL_DIR}/umpire/src/tpl/umpire/camp/src/errors.cpp
    ${EXTERNAL_DIR}/umpire/src/tpl
    ${EXTERNAL_DIR}/umpire/build/src/tpl/umpire/camp/include
)

link_directories(
    ${UMPIRE_LIBRARY_DIRS}
)

# foreach(TEST_SOURCE ${TEST_SOURCES})
# get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
# add_executable(${TEST_NAME} ${TEST_SOURCE} ${PROJECT_SOURCES})
# target_include_directories(${TEST_NAME} PRIVATE
# ${CMAKE_CURRENT_SOURCE_DIR}/../..
# ${TORCH_INCLUDE_DIRS}
# ${MKL_INCLUDE_DIRS}
# ${CONDA_PREFIX_PATH}/include
# # ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/nvidia/cublas/include/
# # ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/nvidia/cuda_runtime/include/
# # ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/triton/backends/nvidia/include/
# # /usr/local/cuda/include/
# ${CMAKE_INCLUDE_PATH}
# ${MPI_INCLUDE_PATH}
# )
# target_link_libraries(${TEST_NAME}
# umpire
# ${TORCH_LIBRARIES}
# pthread numa
# CUDA::cublas CUDA::cudart MKL::MKL OpenMP::OpenMP_CXX
# ${MPI_LIBRARIES}
# )
# add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
# endforeach(TEST_SOURCE)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(test_shared_pin_memory test_shared_pin_memory.cpp ${MEMORY_SRC})
target_include_directories(test_shared_pin_memory PRIVATE
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(test_shared_pin_memory
    CUDA::cublas CUDA::cudart uuid
)
add_test(NAME test_shared_pin_memory COMMAND test_shared_pin_memory)
set_target_properties(test_shared_pin_memory PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

add_executable(test_memcpy_bw test_memcpy_bw.cpp ${MEMORY_SRC})
target_include_directories(test_shared_pin_memory PRIVATE
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(test_memcpy_bw
    CUDA::cublas CUDA::cudart uuid numa
)
add_test(NAME test_memcpy_bw COMMAND test_memcpy_bw)
set_target_properties(test_memcpy_bw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)