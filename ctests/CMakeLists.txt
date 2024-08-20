# gtest make all files in the directory available to the test

project(ctests)

# use gcc for compiling
# set(CMAKE_CXX_COMPILER "g++")

# Add the test files, create one executable for each test file, start with "test_*"

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# find cuda
find_package(CUDA REQUIRED)

#find torch
find_package(Torch REQUIRED)

# find all files in the directory match with "test_*"
file(GLOB TEST_SOURCES "test_*.cpp")
file(GLOB PROJECT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../csrc/memory/*.cpp")

# print CMAKE_ROOT_DIR, CMAKE_CURRENT_SOURCE_DIR
message(STATUS "CMAKE_ROOT_DIR: ${CMAKE_ROOT_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

# cpp flag add -m option to enable avx2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -march=znver3 -O3 -D__AVX256__=1")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE} ${PROJECT_SOURCES})
    # target_compile_options(${TEST_NAME} PRIVATE /arch:AVX2)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME} pthread gtest gtest_main ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach(TEST_SOURCE)

