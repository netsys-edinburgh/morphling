cmake_minimum_required(VERSION 3.22)
project(CUDAHostRegisterTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Create executable
add_executable(test_cuda_host_register_tensor test_cuda_host_register_tensor.cpp)

# Link libraries - only CUDA, no Torch needed
target_link_libraries(test_cuda_host_register_tensor
    PUBLIC
        ${CUDA_LIBRARIES}
        cublas
)

target_include_directories(test_cuda_host_register_tensor
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
)
