#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

class StagingPool {
 public:
  explicit StagingPool(int num_threads = 2) {
    if (num_threads <= 0) {
      num_threads = 1;
    }
    threads_.reserve(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this]() { WorkerLoop(); });
    }
  }

  ~StagingPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      quit_ = true;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void Submit(std::function<void()> work,
              std::function<void()> on_complete = nullptr) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push_back({std::move(work), std::move(on_complete)});
    }
    cv_.notify_one();
  }

 private:
  void WorkerLoop() {
    while (true) {
      StagingTask task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return quit_ || !tasks_.empty(); });
        if (quit_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }

      try {
        if (task.work) {
          task.work();
        }
      } catch (...) {
      }

      if (task.on_complete) {
        task.on_complete();
      }
    }
  }

  struct StagingTask {
    std::function<void()> work;
    std::function<void()> on_complete;
  };

  std::vector<std::thread> threads_;
  std::deque<StagingTask> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool quit_ = false;
};
