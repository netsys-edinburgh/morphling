# ============================================================================
# Zero-Copy Test Suite (unit/zerocopy/)
# ============================================================================

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)

# Source files needed by the tests (serialization_buffer + scatter_gather_buffer + matrix_partition + dependencies)
set(ZEROCOPY_TEST_DEPS
  ${PROJECT_ROOT}/csrc/backend/serialization_buffer.cpp
  ${PROJECT_ROOT}/csrc/backend/scatter_gather_buffer.cpp
  ${PROJECT_ROOT}/csrc/backend/matrix_partition.cpp
  ${PROJECT_ROOT}/csrc/core/generator.cpp
  ${PROJECT_ROOT}/csrc/timer/virtual_clock.cpp
)

# Base logging sources
file(GLOB BASE_LOGGING_SRC ${PROJECT_ROOT}/external/muduo_base/*.cc)

# Proto generated sources (already built by parent project)
# We reference them via the PROTO_SRCS variable set by root CMakeLists.txt

# Include directories
set(ZEROCOPY_INCLUDE_DIRS
  ${PROJECT_ROOT}/csrc
  ${PROJECT_ROOT}/external
  ${CMAKE_BINARY_DIR}  # for generated proto headers
  ${TORCH_INCLUDE_DIRS}
)

# Common link libraries
set(ZEROCOPY_COMMON_LIBS
  GTest::gtest
  GTest::gtest_main
  ${Protobuf_LIBRARIES}
  ${TORCH_LIBRARIES}
  uuid
  Threads::Threads
)

# ============================================================================
# Helper function to create test targets
# ============================================================================
function(add_zerocopy_test TEST_NAME)
  add_executable(${TEST_NAME}
    ${TEST_NAME}.cpp
    ${ZEROCOPY_TEST_DEPS}
    ${BASE_LOGGING_SRC}
    ${PROTO_SRCS}
  )
  target_include_directories(${TEST_NAME} PRIVATE ${ZEROCOPY_INCLUDE_DIRS})
  target_link_libraries(${TEST_NAME} PRIVATE ${ZEROCOPY_COMMON_LIBS})
  if(TARGET generate_proto_files)
    add_dependencies(${TEST_NAME} generate_proto_files)
  endif()
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# ============================================================================
# Correctness Tests
# ============================================================================
add_zerocopy_test(test_aligned_buffer_pool)
add_zerocopy_test(test_serialization_buffer)
add_zerocopy_test(test_scatter_gather_buffer)
add_zerocopy_test(test_matrix_partition)

# ============================================================================
# CUDA Pinned Pool Test (uses CUDAToolkit from parent project)
# ============================================================================
if(TARGET CUDA::cudart)
  add_executable(test_cuda_pinned_pool
    test_cuda_pinned_pool.cpp
  )
  target_link_libraries(test_cuda_pinned_pool PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    Threads::Threads
  )
  add_test(NAME test_cuda_pinned_pool COMMAND test_cuda_pinned_pool)
endif()
