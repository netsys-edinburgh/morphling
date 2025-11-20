# 🔴 关键发现：Client Ctrl+C 时连接未正确关闭

## 问题现象

```bash
# Client 正常运行时
$ netstat -anp | grep 8888 | grep ESTABLISHED
tcp  0  0  127.0.0.1:8888  127.0.0.1:45678  ESTABLISHED

# Client Ctrl+C 停止后
$ netstat -anp | grep 8888 | grep ESTABLISHED
tcp  0  0  127.0.0.1:8888  127.0.0.1:45678  ESTABLISHED  ← 还在！

# ConnectionClosedCb 没有被触发
# [ConnectionClosedCb] Device X disconnected 日志不出现
```

---

## 根本原因

### 问题 1：Client 没有正确关闭连接

当 client 执行 `Ctrl+C` 时：

```
Client 进程被杀死
  ↓
TCP 连接没有被正确关闭（没有发送 FIN 或 RST）
  ↓
Server 端仍然认为连接有效
  ↓
ConnectionClosedCb 不被触发（因为连接没有关闭事件）
  ↓
conn_map_ 中仍然存储着这个连接
  ↓
Server 无法知道 Device 已经死了
```

### 问题 2：Server 端没有检测机制

Server 现在的行为：
- ✅ 被动监听连接关闭事件
- ❌ 没有主动检测连接是否还活着（no heartbeat/keep-alive）
- ❌ 没有超时检测机制

---

## 解决方案

### 方案 A：添加 Keep-Alive / 心跳检测（推荐）

在 Server 端定期检查连接是否还活着：

```cpp
// 添加到 ProxySvrImpl 类
void ProxySvrImpl::StartConnectionMonitor() {
  // 启动定期检查线程
  std::thread monitor_thread([this]() {
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(5)); // 每 5 秒检查一次
      
      CheckAndCleanDeadConnections();
    }
  });
  monitor_thread.detach();
}

void ProxySvrImpl::CheckAndCleanDeadConnections() {
  std::vector<std::string> dead_connections;
  
  {
    std::lock_guard<std::mutex> lock(conn_map_mutex_);
    for (auto& [addr, conn] : conn_map_) {
      if (conn && conn->IsClosed()) {
        LOG_WARN << "Found closed connection: " << addr;
        dead_connections.push_back(addr);
      }
    }
  }
  
  // 移除死连接
  for (const auto& addr : dead_connections) {
    LOG_INFO << "Removing dead connection: " << addr;
    
    // 触发故障处理逻辑
    HandleConnectionDeath(addr);
    
    {
      std::lock_guard<std::mutex> lock(conn_map_mutex_);
      conn_map_.erase(addr);
    }
  }
}
```

### 方案 B：使用 TCP Keep-Alive

在 socket 级别启用 TCP Keep-Alive：

```cpp
// 在 ListenerLibevent 中添加
void ProxySvrHandle::EnableKeepAlive(int sockfd) {
  int enable = 1;
  int idle = 5;      // 5 秒无数据后启动探测
  int interval = 2;  // 每 2 秒发送一次探测
  int count = 3;     // 发送 3 次探测失败后关闭
  
  setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable));
  setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
  setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
  setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count));
  
  LOG_INFO << "TCP Keep-Alive enabled on socket " << sockfd;
}
```

### 方案 C：客户端优雅关闭（Client 侧改进）

在 client 端添加信号处理：

```python
# device.py 中
import signal
import atexit

def cleanup():
    """优雅关闭"""
    LOG_INFO("Device shutting down...")
    if client_conn:
        client_conn.close()  # 正确关闭连接
    LOG_INFO("Connection closed")

signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
atexit.register(cleanup)
```

---

## 立即修复方案（最简单有效）

### 方法 1：添加定期连接检查

在 `ProxySvrImpl::Initialize()` 中添加连接监控：

```cpp
void ProxySvrImpl::Initialize(UeventLoop* loop) {
  // ... 现有代码 ...
  
  // 启动连接监控线程
  StartConnectionMonitor();
  
  LOG_INFO << "[ProxySvrImpl::Initialize] Connection monitor started";
}

void ProxySvrImpl::StartConnectionMonitor() {
  std::thread([this]() {
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(10)); // 每 10 秒检查
      CheckAndCleanDeadConnections();
    }
  }).detach();
}

void ProxySvrImpl::CheckAndCleanDeadConnections() {
  std::vector<std::string> dead_addrs;
  
  {
    std::lock_guard<std::mutex> lock(partition_tracker_mutex_);
    
    for (auto it = conn_map_.begin(); it != conn_map_.end(); ++it) {
      const auto& addr = it->first;
      auto& conn = it->second;
      
      // 检查连接是否关闭
      if (conn && conn->IsClosed()) {
        dead_addrs.push_back(addr);
        LOG_WARN << "Found dead connection: " << addr;
      }
    }
  }
  
  // 处理死连接
  for (const auto& addr : dead_addrs) {
    int device_id = -1;
    
    // 找到对应的 device_id
    {
      std::lock_guard<std::mutex> lock(partition_tracker_mutex_);
      auto it = conn_map_.find(addr);
      if (it != conn_map_.end()) {
        device_id = std::distance(conn_map_.begin(), it);
        
        LOG_INFO << "Cleaning up dead connection for device " << device_id;
        
        // 触发故障处理
        bool has_pending = false;
        size_t pending_count = 0;
        
        auto tracker_it = partition_tracker_.find(device_id);
        if (tracker_it != partition_tracker_.end() && !tracker_it->second.empty()) {
          has_pending = true;
          pending_count = tracker_it->second.size();
        }
        
        conn_map_.erase(addr);
        
        if (has_pending && conn_map_.size() > 0) {
          int64_t target = FindTargetDeviceForFailure(device_id);
          if (target != -1) {
            HandleDeviceFailure(device_id, target);
          }
        }
      }
    }
  }
}
```

---

## 更好的解决方案：检测连接超时

使用最后活跃时间戳：

```cpp
// 在 ProxySvrImpl 中添加
std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_activity_;
std::mutex activity_mutex_;

void ProxySvrImpl::UpdateConnectionActivity(const std::string& addr) {
  std::lock_guard<std::mutex> lock(activity_mutex_);
  last_activity_[addr] = std::chrono::steady_clock::now();
}

void ProxySvrImpl::CheckConnectionTimeout() {
  const int TIMEOUT_SECONDS = 30;
  auto now = std::chrono::steady_clock::now();
  std::vector<std::string> dead_addrs;
  
  {
    std::lock_guard<std::mutex> lock(activity_mutex_);
    for (const auto& [addr, last_time] : last_activity_) {
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          now - last_time).count();
      
      if (elapsed > TIMEOUT_SECONDS) {
        LOG_WARN << "Connection timeout: " << addr << " (inactive for " 
                 << elapsed << "s)";
        dead_addrs.push_back(addr);
      }
    }
  }
  
  // 清理超时连接
  for (const auto& addr : dead_addrs) {
    CleanupConnection(addr);
  }
}

// 在 HandleMatMul 中调用
void ProxySvrHandle::HandleMatMul(const void* payload, size_t size) {
  // ... 现有代码 ...
  
  // 更新活跃时间
  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->UpdateConnectionActivity(conn_addr);
}
```

---

## 完整的修复步骤

### 步骤 1：在 `proxy_svr.h` 中添加新的方法声明

```cpp
class ProxySvrImpl {
  // ... 现有代码 ...
  
  // 连接监控
  void StartConnectionMonitor();
  void CheckAndCleanDeadConnections();
  void UpdateConnectionActivity(const std::string& addr);
  
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_activity_;
  std::mutex activity_mutex_;
};
```

### 步骤 2：在 `proxy_svr.cc` 中实现

```cpp
// 在 Initialize() 中添加
void ProxySvrImpl::Initialize(UeventLoop* loop) {
  // ... 现有代码 ...
  
  // 启动连接监控
  StartConnectionMonitor();
  
  LOG_INFO << "[ProxySvrImpl::Initialize] Server initialization completed";
}

// 实现连接监控
void ProxySvrImpl::StartConnectionMonitor() {
  std::thread monitor([this]() {
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(10));
      CheckAndCleanDeadConnections();
    }
  }).detach();
}

void ProxySvrImpl::CheckAndCleanDeadConnections() {
  std::vector<std::string> dead_addrs;
  
  {
    std::lock_guard<std::mutex> lock(partition_tracker_mutex_);
    for (auto it = conn_map_.begin(); it != conn_map_.end(); ++it) {
      if (it->second && it->second->IsClosed()) {
        dead_addrs.push_back(it->first);
      }
    }
  }
  
  for (const auto& addr : dead_addrs) {
    // 触发故障处理...
  }
}

void ProxySvrImpl::UpdateConnectionActivity(const std::string& addr) {
  std::lock_guard<std::mutex> lock(activity_mutex_);
  last_activity_[addr] = std::chrono::steady_clock::now();
}
```

---

## netstat 验证方法

### 修复后的测试流程

```bash
# 1. 启动 Server
./proxy_svr &

# 2. 启动多个 Client
python device.py &
python device.py &
python device.py &

# 3. 检查连接（应该显示 3）
netstat -anp | grep 8888 | grep ESTABLISHED | wc -l
# 输出: 3

# 4. 关闭一个 Client (Ctrl+C)
# 按 Ctrl+C

# 5. 立即检查（可能还是 3）
netstat -anp | grep 8888 | grep ESTABLISHED | wc -l
# 输出: 3（连接还在）

# 6. 等待监控线程检查（10 秒）
sleep 10

# 7. 再次检查（应该变成 2）
netstat -anp | grep 8888 | grep ESTABLISHED | wc -l
# 输出: 2✅

# 8. 查看日志中的故障处理
tail -f server.log | grep "ConnectionClosedCb\|HandleDeviceFailure"
# 应该看到完整的故障处理流程
```

---

## 最根本的解决方案：使用 libevent 的连接活跃检测

libevent 本身支持连接监控，可以配置：

```cpp
// 在设置连接时启用读事件，即使没有数据
// 这样可以检测到连接关闭
listener_->EnableReadEvent();  // 伪代码
```

---

## 问题根源总结

| 问题 | 原因 | 修复 |
|------|------|------|
| **Ctrl+C 后连接未关闭** | Client 进程被杀死，TCP 连接悬挂 | 启动定期检查线程 |
| **ConnectionClosedCb 不触发** | 连接没有关闭事件，只是进程死了 | 检查 `conn->IsClosed()` 状态 |
| **Server 不知道 Device 已死** | 只依赖被动事件，没有主动检测 | 添加心跳/超时检测 |

---

## 推荐方案：3 层防御

1. **第 1 层**：正常关闭时的 ConnectionClosedCb（已有）✅
2. **第 2 层**：定期检查 `IsClosed()` 状态（新增）🔧
3. **第 3 层**：超时检测（可选但推荐）⏰

这样可以捕获所有可能的连接失效情况！
