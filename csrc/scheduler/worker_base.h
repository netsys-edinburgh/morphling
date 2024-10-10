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
    cv_.notify_all();
    worker_.join();
  }

  virtual void AddTask(const std::string& task_id, Task&& t) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace_back(std::make_tuple(task_id, std::move(t)));
      task_count_++;
      task_ids_.insert(task_id);
    }
    cv_.notify_all();
  }

  virtual void Run() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      // Wait until we have a task or the worker is quitting
      cv_.wait(lock, [this] { return !tasks_.empty() || quit_; });
      if (quit_) {
        break;
      }

      auto [task_id, task] = tasks_.front();
      tasks_.pop_front();
      lock.unlock();

      task();
      task_count_--;

      lock.lock();
      task_ids_.erase(task_id);
      lock.unlock();

      cv_.notify_all();
    }
  }

  void WaitTaskDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock,
             [this] { return (tasks_.empty() && task_count_ == 0) || quit_; });
  }

  void WaitTaskDone(const std::string& task_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, &task_id] {
      return (task_ids_.count(task_id) == 0) || quit_;
    });
  }

 protected:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool quit_ = false;
  std::thread worker_;
  std::deque<std::tuple<std::string, Task>> tasks_;
  std::unordered_set<std::string> task_ids_;
  std::atomic_int task_count_{0};
};