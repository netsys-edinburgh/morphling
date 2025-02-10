// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <spdlog/spdlog.h>

#include <iostream>
#include <mutex>

#include "common/types_and_defs.h"  // for TensorKey
#include "noncopyable.h"

#define PRINT_ARGS(...)                \
  do {                                 \
    std::cout << #__VA_ARGS__ << ": "; \
    print(__VA_ARGS__);                \
  } while (0)

// Helper function to print each argument
inline void print() { std::cout << std::endl; }  // Base case to end recursion

template <typename T, typename... Args>
inline void print(T first, Args... args) {
  std::cout << first;
  if constexpr (sizeof...(args) > 0) {
    std::cout << ", ";
    print(args...);  // Recursive call
  } else {
    std::cout << std::endl;
  }
}

int str2level(const char* level);
std::string level2str(int level);
std::string formatstr();

enum LogLevel { kFatal, kDebug, kInfo, kWarn, kError };

extern std::once_flag kLoggerFlag;
extern int kLogLevel;
extern std::mutex kLogMutex;

extern void InitLogger();

#define LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)
#define LOG_INFO(...) spdlog::info(__VA_ARGS__)
#define LOG_ERROR(...) spdlog::error(__VA_ARGS__)
#define LOG_WARN(...) spdlog::warn(__VA_ARGS__)
#define LOG_FATAL(...)                                 \
  do {                                                 \
    spdlog::critical(__VA_ARGS__);                     \
    throw(std::runtime_error("Logged a FATAL error")); \
  } while (0)

#define LOG_FATAL_IF(cond, ...) \
  if (cond) {                   \
    LOG_FATAL(__VA_ARGS__);     \
  }
#define LOG_ERROR_IF(cond, ...) \
  if (cond) {                   \
    LOG_ERROR(__VA_ARGS__);     \
  }
#define LOG_WARN_IF(cond, ...) \
  if (cond) {                  \
    LOG_WARN(__VA_ARGS__);     \
  }
#define LOG_INFO_IF(cond, ...) \
  if (cond) {                  \
    LOG_INFO(__VA_ARGS__);     \
  }

#define CHECK_CUBLAS_ERROR(call)                                            \
  {                                                                         \
    cublasStatus_t err = (call);                                            \
    LOG_FATAL_IF(err != CUBLAS_STATUS_SUCCESS, "CUBLAS error. message: {}", \
                 cublasGetStatusString(err));                               \
  }

#define CHECK_CUDA_ERROR(call)                                  \
  {                                                             \
    cudaError_t err = (call);                                   \
    LOG_FATAL_IF(err != cudaSuccess, "CUDA error. message: {}", \
                 cudaGetErrorString(err));                      \
  }

// fmt::formatter for std::vector

namespace fmt {
template <typename T>
struct formatter<std::vector<T>> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::vector<T>& v, FormatContext& ctx) {
    auto it = ctx.out();
    *it = '[';
    ++it;
    for (size_t i = 0; i < v.size(); i++) {
      if (i > 0) {
        *it = ',';
        ++it;
      }
      it = format_to(it, "{}", v[i]);
    }
    *it = ']';
    return ++it;
  }
};

// create a spdlog fmt for TensorKey
template <>
struct formatter<TensorKey> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const TensorKey& p, FormatContext& ctx) {
    return format_to(ctx.out(), "[{}:{}:{}:{}]", std::get<0>(p), std::get<1>(p),
                     std::get<2>(p), std::get<3>(p));
  }
};

}  // namespace fmt
