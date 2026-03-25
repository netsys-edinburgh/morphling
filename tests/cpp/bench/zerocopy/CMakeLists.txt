# ============================================================================
# Zero-Copy Benchmarks (bench/zerocopy/)
# ============================================================================

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)

# Source files needed by the benchmarks (serialization_buffer + scatter_gather_buffer + matrix_partition + dependencies)
set(ZEROCOPY_BENCH_DEPS
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
set(ZEROCOPY_BENCH_INCLUDE_DIRS
  ${PROJECT_ROOT}/csrc
  ${CMAKE_BINARY_DIR}  # for generated proto headers
  ${TORCH_INCLUDE_DIRS}
)

# Common link libraries for benchmarks
set(ZEROCOPY_BENCH_LIBS
  benchmark::benchmark
  benchmark::benchmark_main
  ${Protobuf_LIBRARIES}
  ${TORCH_LIBRARIES}
  uuid
  Threads::Threads
)

# ============================================================================
# Performance Benchmarks
# ============================================================================
function(add_zerocopy_bench BENCH_NAME)
  add_executable(${BENCH_NAME}
    ${BENCH_NAME}.cpp
    ${ZEROCOPY_BENCH_DEPS}
    ${BASE_LOGGING_SRC}
    ${PROTO_SRCS}
  )
  target_include_directories(${BENCH_NAME} PRIVATE ${ZEROCOPY_BENCH_INCLUDE_DIRS})
  target_link_libraries(${BENCH_NAME} PRIVATE ${ZEROCOPY_BENCH_LIBS})
  if(TARGET generate_proto_files)
    add_dependencies(${BENCH_NAME} generate_proto_files)
  endif()
endfunction()

add_zerocopy_bench(bench_pool_allocation)
add_zerocopy_bench(bench_serialization)
