#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "utils/noncopyable.h"

class WorkerBase : public noncopyable {
 public:
  typedef std::function<void()> Task;

 public:
  virtual ~WorkerBase() = default;

  virtual void Start() {
    worker_ = std::thread([this] { Run(); });
  }

  virtual void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      quit_ = true;
    }
    cv_.notify_one();
    worker_.join();
  }

  virtual void AddTask(Task&& t) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace_back(std::move(t));
    }
    cv_.notify_one();
  }

  virtual void Run() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      // Wait until we have a task or the worker is quitting
      cv_.wait(lock, [this] { return !tasks_.empty() || quit_; });
      if (quit_) {
        break;
      }

      Task task = tasks_.front();
      tasks_.pop_front();
      lock.unlock();

      task();
    }
  }

 protected:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool quit_ = false;
  std::thread worker_;
  std::deque<Task> tasks_;
};