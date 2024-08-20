#pragma once
#include <cuda_runtime_api.h>

#include <condition_variable>
#include <mutex>

struct MemoryShadow {
  cudaIpcMemHandle_t handles[5];
};

struct TeleportArgs {
  std::mutex m;
  std::condition_variable cv;
  bool ready = false;

  void teleport() {
    std::unique_lock<std::mutex> lk(m);
    ready = true;
    cv.notify_all();
  }
};
