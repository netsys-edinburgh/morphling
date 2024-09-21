// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <spdlog/spdlog.h>

#include <iostream>
#include <mutex>

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
#define LOG_FATAL(...)             \
  do {                             \
    spdlog::critical(__VA_ARGS__); \
    assert(false);                 \
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

// #define LOG_DEBUG(...)                                      \
//   do {                                                      \
//     if (kLogLevel <= kDebug) {                              \
//       std::lock_guard<std::mutex> lock(kLogMutex);          \
//       std::cout << formatstr() << level2str(kDebug) << " "; \
//       print(__VA_ARGS__);                                   \
//     }                                                       \
//   } while (0)

// #define LOG_INFO(...)                                      \
//   do {                                                     \
//     if (kLogLevel <= kInfo) {                              \
//       std::lock_guard<std::mutex> lock(kLogMutex);         \
//       std::cout << formatstr() << level2str(kInfo) << " "; \
//       print(__VA_ARGS__);                                  \
//     }                                                      \
//   } while (0)

// #define LOG_ERROR(...)                                      \
//   do {                                                      \
//     if (kLogLevel <= kError) {                              \
//       std::lock_guard<std::mutex> lock(kLogMutex);          \
//       std::cout << formatstr() << level2str(kError) << " "; \
//       print(__VA_ARGS__);                                   \
//     }                                                       \
//   } while (0)

// #define LOG_WARN(...)                                      \
//   do {                                                     \
//     if (kLogLevel <= kWarn) {                              \
//       std::lock_guard<std::mutex> lock(kLogMutex);         \
//       std::cout << formatstr() << level2str(kWarn) << " "; \
//       print(__VA_ARGS__);                                  \
//     }                                                      \
//   } while (0)

// #define LOG_FATAL(...)                                      \
//   do {                                                      \
//     if (kLogLevel <= kError) {                              \
//       std::lock_guard<std::mutex> lock(kLogMutex);          \
//       std::cout << formatstr() << level2str(kFatal) << " "; \
//       print(__VA_ARGS__);                                   \
//       throw std::runtime_error("Logged a FATAL error");     \
//     }                                                       \
//   } while (0)
