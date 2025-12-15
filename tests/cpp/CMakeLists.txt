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
find_package(MPI REQUIRED)

set(CMAKE_SKIP_RPATH ON)

# find all files in the directory match with "test_*"
file(GLOB TEST_SOURCES "test_*.cpp")

# Define source file globs for compilation
file(GLOB NETWORK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/network/*.cc")
file(GLOB BASE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/base/*.cc")
file(GLOB TIMER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/timer/*.cc")

# # print CMAKE_ROOT_DIR, CMAKE_CURRENT_SOURCE_DIR
# message(STATUS "CMAKE_ROOT_DIR: ${CMAKE_ROOT_DIR}")
# message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
# message(STATUS "MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}")
# message(STATUS "MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}")
# # Retrieve the Python version
# message(STATUS "Python version: ${Python_VERSION}")

# cpp flag add -m option to enable avx2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -std=c++17 -ggdb3 -fno-omit-frame-pointer")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
# -mavx2 -march=znver3

# message(STATUS "UMPAIRE_INCLUDE_DIRS: ${UMPAIRE_INCLUDE_DIRS}")
# link_directories(${CONDA_PREFIX_PATH}/lib)
# set(CMAKE_BUILD_RPATH "/mnt/data/xly/.conda/envs/edgeml/lib:/usr/local/cuda-12.2/targets/x86_64-linux/lib")
# set(CMAKE_INSTALL_RPATH "/mnt/data/xly/.conda/envs/edgeml/lib:/usr/local/cuda-12.2/targets/x86_64-linux/lib")
include_directories(
    # ${UMPIRE_INCLUDE_DIRS}
    ${EXTERNAL_DIR}/umpire/src/tpl/umpire/camp/include

    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    # ${EXTERNAL_DIR}/umpire/src/tpl/umpire/camp/src/errors.cpp
    ${EXTERNAL_DIR}/umpire/src/tpl
    ${EXTERNAL_DIR}/umpire/build/src/tpl/umpire/camp/include
)


# message(STATUS "Python_VERSION: ${Python_VERSION}")
# message(STATUS "CONDA_PREFIX_PATH: ${CONDA_PREFIX_PATH}")
link_directories(
    # ${UMPIRE_LIBRARY_DIRS}
    ${CONDA_PREFIX_PATH}/lib
    ${CONDA_PREFIX_PATH}/lib/python${Python_VERSION}/site-packages/torch/lib
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

message(STATUS "torch_python_LIBRARY: ${torch_python_LIBRARY}")

add_executable(echo_server_test echo_server_test.cc ${NETWORK_SRC} ${BASE_SRC} ${TIMER_SRC})
target_link_libraries(echo_server_test event uuid)
set_target_properties(echo_server_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME echo_server_test COMMAND echo_server_test)

add_executable(echo_client_test echo_client_test.cc ${NETWORK_SRC} ${BASE_SRC} ${TIMER_SRC})
target_link_libraries(echo_client_test event uuid)
set_target_properties(echo_client_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME echo_client_test COMMAND echo_client_test)

add_executable(test_kern_copy test_kern_copy.cpp)
# target_link_libraries(test_kern_copy uuid)
set_target_properties(test_kern_copy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME test_kern_copy COMMAND test_kern_copy)

add_executable(test_splice_copy test_splice_copy.cpp)
# target_link_libraries(test_splice_copy uuid)
set_target_properties(test_splice_copy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME test_splice_copy COMMAND test_splice_copy)


add_executable(test_uuid test_uuid.cpp)
target_link_libraries(test_uuid uuid)
set_target_properties(test_uuid PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME test_uuid COMMAND test_uuid)

add_executable(test_torch_layout test_torch_layout.cpp)
target_include_directories(test_torch_layout PRIVATE
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_torch_layout ${TORCH_LIBRARIES})
set_target_properties(test_torch_layout PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME test_torch_layout COMMAND test_torch_layout)

set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/backend/server_base.cpp
)
add_executable(test_matrix_partition test_matrix_partition.cpp ${SRC})
target_include_directories(test_matrix_partition PRIVATE
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_matrix_partition ${TORCH_LIBRARIES} ${torch_python_LIBRARY})
set_target_properties(test_matrix_partition PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME test_matrix_partition COMMAND test_matrix_partition)


add_executable(test_mqtt_sender test_mqtt_sender.cpp)
# target_include_directories(test_mqtt_sender PRIVATE
#     ${TORCH_INCLUDE_DIRS}
# )
target_link_libraries(test_mqtt_sender libmosquitto)
set_target_properties(test_mqtt_sender PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME test_mqtt_sender COMMAND test_mqtt_sender)


add_executable(test_rc_copy test_rc_copy.cpp)
target_include_directories(test_rc_copy PRIVATE
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_rc_copy ${TORCH_LIBRARIES} ${torch_python_LIBRARY})
set_target_properties(test_rc_copy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME test_rc_copy COMMAND test_rc_copy)

#/mnt/data/xly/.conda/envs/emulator/lib/python3.9/site-packages/torch/lib/

# set(TEST_INCLUDE_DIRS
#     ${TORCH_INCLUDE_DIRS}
# )
# set(TEST_LIBRARIES
#     uuid ${TORCH_LIBRARIES} rttr_core
# )
# add_test_executable(test_matrix_partition
#     LIBRARIES ${TEST_LIBRARIES}
#     INCLUDE_DIRECTORIES ${TEST_INCLUDE_DIRS}
# )
# add_executable(test_shared_pin_memory test_shared_pin_memory.cpp ${MEMORY_SRC} ${UTILS_SRC})
# target_include_directories(test_shared_pin_memory PRIVATE
#     ${CUDA_INCLUDE_DIRS}
#     ${TORCH_INCLUDE_DIRS}
# )
# target_link_libraries(test_shared_pin_memory
#     CUDA::cublas CUDA::cudart uuid ${TORCH_LIBRARIES} rttr_core
# )
# add_test(NAME test_shared_pin_memory COMMAND test_shared_pin_memory)
# set_target_properties(test_shared_pin_memory PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
# )

# add_executable(test_memcpy_bw test_memcpy_bw.cpp ${MEMORY_SRC}  ${UTILS_SRC})
# target_include_directories(test_shared_pin_memory PRIVATE
#     ${CUDA_INCLUDE_DIRS}
#     ${TORCH_INCLUDE_DIRS}
# )
# target_link_libraries(test_memcpy_bw
#     CUDA::cublas CUDA::cudart uuid numa ${TORCH_LIBRARIES} rttr_core
# )
# add_test(NAME test_memcpy_bw COMMAND test_memcpy_bw)
# set_target_properties(test_memcpy_bw PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
# )

add_executable(echo_server_libuv echo_server_libuv.cc)
target_link_libraries(echo_server_libuv uv)
set_target_properties(echo_server_libuv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME echo_server_libuv COMMAND echo_server_libuv)

add_executable(echo_client_libuv echo_client_libuv.cc)
target_link_libraries(echo_client_libuv uv)
set_target_properties(echo_client_libuv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME echo_client_libuv COMMAND echo_client_libuv)
