#ifndef UEVENTLOOP_THREAD_H
#define UEVENTLOOP_THREAD_H

#include <string>

#include "callbacks.h"
#include "muduo_base/condition.h"
#include "muduo_base/mutex.h"
#include "muduo_base/thread.h"

namespace uevent {

class UeventLoop;

class UeventLoopThread {
 public:
  UeventLoopThread(const CreateLoopHandleCb& cb1 = CreateLoopHandleCb(),
                   const ThreadInitCb& cb2 = ThreadInitCb(),
                   const std::string& name = std::string());
  ~UeventLoopThread();
  UeventLoop* StartLoop();

 private:
  void ThreadFunc();
  std::string name_;
  UeventLoop* loop_;
  base::Thread thread_;
  base::MutexLock mutex_;
  base::Condition cond_;
  CreateLoopHandleCb create_loop_handle_cb_;
  ThreadInitCb thread_init_cb_;
};

}  // namespace uevent

#endif  // UEVENTLOOP_THREAD_H
