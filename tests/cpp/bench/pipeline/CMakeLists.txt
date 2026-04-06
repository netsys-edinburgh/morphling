set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)

if(NOT TARGET CUDA::cuda_driver)
  message(WARNING "CUDA driver API target not available; pipeline benchmarks skipped.")
  return()
endif()

set(PIPELINE_BENCH_DEPS
  ${PROJECT_ROOT}/csrc/scheduler/gpu_worker.cpp
  ${PROJECT_ROOT}/csrc/scheduler/cpu_worker.cpp
  ${PROJECT_ROOT}/csrc/memory/caching_allocator.cpp
  ${PROJECT_ROOT}/csrc/core/cuda_utils.cpp
  ${PROJECT_ROOT}/csrc/core/logger.cpp
  ${PROJECT_ROOT}/csrc/core/generator.cpp
  ${PROJECT_ROOT}/csrc/timer/virtual_clock.cpp
)

file(GLOB PIPELINE_BASE_LOGGING_SRC ${PROJECT_ROOT}/external/muduo_base/*.cc)

set(PIPELINE_BENCH_INCLUDE_DIRS
  ${PROJECT_ROOT}/csrc
  ${PROJECT_ROOT}/external
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set(PIPELINE_BENCH_LIBS
  benchmark::benchmark
  benchmark::benchmark_main
  CUDA::cudart
  CUDA::cublas
  CUDA::cuda_driver
  ${TORCH_LIBRARIES}
  Threads::Threads
  uuid
  openblas
)

function(add_pipeline_bench BENCH_NAME)
  add_executable(${BENCH_NAME}
    ${BENCH_NAME}.cpp
    ${PIPELINE_BENCH_DEPS}
    ${PIPELINE_BASE_LOGGING_SRC}
  )
  target_include_directories(${BENCH_NAME} PRIVATE ${PIPELINE_BENCH_INCLUDE_DIRS})
  target_compile_definitions(${BENCH_NAME} PRIVATE HAVE_CUDA)
  target_link_libraries(${BENCH_NAME} PRIVATE ${PIPELINE_BENCH_LIBS})
endfunction()

add_pipeline_bench(bench_h2d_compute_overlap)
add_pipeline_bench(bench_dual_stream_gemm)
add_pipeline_bench(bench_pinned_vs_managed)
add_pipeline_bench(bench_event_pool)
add_pipeline_bench(bench_eventloop_cuda_ops)
