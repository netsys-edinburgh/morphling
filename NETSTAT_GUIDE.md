# 🔍 使用 netstat 检查 Client 连接的完整指南

## 快速命令速查表

### 1️⃣ 查看 Server 监听的端口

```bash
# 查看所有监听的 TCP 端口
netstat -tlnp | grep LISTEN

# 只查看特定端口（比如 8888）
netstat -tlnp | grep :8888

# 结果示例：
# Proto Recv-Q Send-Q Local Address    Foreign Address   State    PID/Program name
# tcp   0      0      0.0.0.0:8888     0.0.0.0:*         LISTEN   12345/proxy_svr
```

---

### 2️⃣ 查看所有已建立的连接

```bash
# 查看所有已建立的连接
netstat -anp | grep ESTABLISHED

# 只看特定程序（proxy_svr）的连接
netstat -anp | grep proxy_svr

# 结果示例：
# tcp   0   0   127.0.0.1:8888   127.0.0.1:45678   ESTABLISHED   12345/proxy_svr
# tcp   0   0   127.0.0.1:8888   127.0.0.1:45679   ESTABLISHED   12345/proxy_svr
# tcp   0   0   127.0.0.1:8888   127.0.0.1:45680   ESTABLISHED   12345/proxy_svr
```

---

### 3️⃣ 查看连接数统计

```bash
# 统计某程序的连接数
netstat -anp | grep proxy_svr | grep ESTABLISHED | wc -l

# 结果示例：
# 3  (表示有 3 个已建立的连接)
```

---

### 4️⃣ 实时监控连接（每秒更新）

```bash
# 每秒刷新一次，显示 proxy_svr 的连接
watch -n 1 'netstat -anp | grep proxy_svr'

# 按 Ctrl+C 退出
```

---

## 常用场景

### 场景 1：启动 Server 后验证是否正常监听

```bash
# 步骤 1：启动 server
./run_server.py &

# 步骤 2：检查是否在监听
netstat -tlnp | grep python

# 预期输出：
# tcp  0  0  0.0.0.0:8888  0.0.0.0:*  LISTEN  12345/python
```

### 场景 2：Device 连接上后检查连接状态

```bash
# 启动 server
./run_server.py &

# 启动 device
python device.py &

# 检查建立的连接
netstat -anp | grep ESTABLISHED | grep 8888

# 预期看到类似：
# tcp  0  0  127.0.0.1:8888  127.0.0.1:45678  ESTABLISHED
```

### 场景 3：检查特定 Device 的连接

```bash
# 如果知道 device 的 PID（比如 54321）
netstat -anp | grep 54321

# 如果知道 device 的 port（比如 45678）
netstat -anp | grep 45678
```

### 场景 4：Device 断开后检查连接是否移除

```bash
# 记下之前的连接数
BEFORE=$(netstat -anp | grep proxy_svr | grep ESTABLISHED | wc -l)
echo "连接数 (断开前): $BEFORE"

# 关闭一个 device
# Ctrl+C 或 kill PID

# 检查连接数是否减少
AFTER=$(netstat -anp | grep proxy_svr | grep ESTABLISHED | wc -l)
echo "连接数 (断开后): $AFTER"

# 预期：AFTER < BEFORE
```

---

## netstat 参数详解

| 参数 | 含义 |
|------|------|
| `-t` | TCP 连接 |
| `-u` | UDP 连接 |
| `-l` | 监听状态 (LISTEN) |
| `-n` | 显示数字地址，不进行 DNS 解析（速度快）|
| `-p` | 显示相关程序 (PID/程序名) |
| `-a` | 显示所有连接 |

---

## 常见连接状态

| 状态 | 含义 | 说明 |
|------|------|------|
| `LISTEN` | 监听 | Server 在等待连接 |
| `ESTABLISHED` | 已建立 | ✅ Client 已连接 |
| `TIME_WAIT` | 时间等待 | 连接已关闭，等待超时 |
| `CLOSE_WAIT` | 关闭等待 | 对方已关闭，本地等待关闭 |
| `FIN_WAIT1` | 第一次关闭等待 | 已发送关闭信号 |
| `FIN_WAIT2` | 第二次关闭等待 | 等待对方关闭 |

---

## 对于你的项目的具体检查

### 基础信息

- **Server 程序**：`proxy_svr` (C++ 程序)
- **Server 端口**：8888 (根据配置，可能不同)
- **Client 程序**：Python Device 脚本
- **连接类型**：TCP

### 检查清单

```bash
# 1. 检查 Server 是否启动并监听
netstat -tlnp | grep 8888
# ✅ 预期看到：LISTEN

# 2. 检查有多少个 Device 已连接
netstat -anp | grep 8888 | grep ESTABLISHED | wc -l
# ✅ 预期：等于已连接的 Device 数

# 3. 查看所有连接的详细信息
netstat -anp | grep proxy_svr
# ✅ 预期看到每个 Device 的连接

# 4. 检查是否有异常连接（比如 TIME_WAIT、CLOSE_WAIT）
netstat -anp | grep 8888 | grep -E "TIME_WAIT|CLOSE_WAIT"
# ✅ 预期为空（没有异常）

# 5. 实时监控连接数变化
watch -n 1 'netstat -anp | grep 8888 | grep ESTABLISHED | wc -l'
# ✅ 可以看到 Device 上线/下线时连接数的变化
```

---

## 诊断故障的完整流程

### 问题 1：Server 无响应

```bash
# 检查 Server 是否监听
netstat -tlnp | grep 8888

# 如果看不到 LISTEN，说明 Server 没启动或端口错误
# 解决：检查 Server 日志，确认端口配置
```

### 问题 2：Device 无法连接

```bash
# 检查 Server 是否监听
netstat -tlnp | grep 8888
# ✅ 应该看到 LISTEN

# 检查防火墙
sudo ufw status
# ✅ 应该允许 8888 端口

# 测试连接
telnet 127.0.0.1 8888
# ✅ 应该连接成功
```

### 问题 3：Device 连接后立即断开

```bash
# 监控连接变化
watch -n 1 'netstat -anp | grep 8888 | grep ESTABLISHED | wc -l'

# 看是否连接数在增加又立即减少
# 检查日志是否有错误信息
```

### 问题 4：Device 断开后连接未清理

```bash
# 检查是否有 TIME_WAIT 或 CLOSE_WAIT
netstat -anp | grep 8888

# 如果看到大量 TIME_WAIT，说明连接未正确清理
# 解决：确保 ConnectionClosedCb 被正确调用
```

---

## 高级用法

### 脚本：监控连接数并告警

```bash
#!/bin/bash

SERVER_PORT=8888
EXPECTED_DEVICES=3

while true; do
  COUNT=$(netstat -anp 2>/dev/null | grep ":$SERVER_PORT" | grep ESTABLISHED | wc -l)
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  
  if [ $COUNT -eq $EXPECTED_DEVICES ]; then
    echo "[$TIMESTAMP] ✅ 连接数正常: $COUNT/$EXPECTED_DEVICES"
  else
    echo "[$TIMESTAMP] ⚠️ 连接数异常: $COUNT/$EXPECTED_DEVICES"
  fi
  
  sleep 1
done
```

使用方法：
```bash
chmod +x monitor_connections.sh
./monitor_connections.sh
```

---

### 脚本：显示所有连接的详细信息

```bash
#!/bin/bash

echo "=== Server 监听状态 ==="
netstat -tlnp | grep 8888

echo ""
echo "=== 已建立的连接 ==="
netstat -anp | grep 8888 | grep ESTABLISHED | while read line; do
  echo "$line"
done

echo ""
echo "=== 连接数统计 ==="
TOTAL=$(netstat -anp | grep 8888 | grep ESTABLISHED | wc -l)
echo "已建立连接数: $TOTAL"

echo ""
echo "=== 异常连接 ==="
ABNORMAL=$(netstat -anp | grep 8888 | grep -E "TIME_WAIT|CLOSE_WAIT" | wc -l)
echo "异常连接数: $ABNORMAL"
```

---

## 与你的代码修复相关的检查

### 修复前（SetConnectionClosedCb 未注册）

```bash
# Device 断开时，连接仍然显示为 ESTABLISHED 很久
netstat -anp | grep 8888
# ✅ 会看到旧的连接仍然存在

# 可能看到大量 TIME_WAIT
netstat -anp | grep 8888 | grep TIME_WAIT | wc -l
# ⚠️ 数字会很大
```

### 修复后（SetConnectionClosedCb 已注册）

```bash
# Device 断开后，连接立即清理
netstat -anp | grep 8888
# ✅ 连接立即从列表中消失

# 几乎没有 TIME_WAIT
netstat -anp | grep 8888 | grep TIME_WAIT | wc -l
# ✅ 数字会很小（0 或 1）
```

---

## 快速测试流程

### 测试 SetConnectionClosedCb 修复是否有效

```bash
# 步骤 1：启动 Server
cd /app
./proxy_svr &
SERVER_PID=$!

# 步骤 2：等待 Server 启动
sleep 2

# 步骤 3：检查 Server 监听状态
echo "=== Server 监听状态 ==="
netstat -tlnp | grep 8888

# 步骤 4：启动 3 个 Device
for i in {1..3}; do
  python device.py --id $i &
done

# 步骤 5：等待连接建立
sleep 2

# 步骤 6：检查已建立的连接
echo "=== Device 连接状态 (连接前) ==="
netstat -anp | grep 8888 | grep ESTABLISHED | wc -l

# 步骤 7：关闭一个 Device（比如 Device 1）
# 按 Ctrl+C 或 killall device.py

# 步骤 8：立即检查连接状态
echo "=== Device 连接状态 (断开后) ==="
sleep 1
netstat -anp | grep 8888 | grep ESTABLISHED | wc -l
# ✅ 应该少 1 个连接

# 步骤 9：检查是否有异常连接
echo "=== 异常连接检查 ==="
netstat -anp | grep 8888 | grep -E "TIME_WAIT|CLOSE_WAIT"
# ✅ 应该立即清理，没有或很少

# 步骤 10：清理
kill $SERVER_PID
```

---

## 故障排查快速参考

| 症状 | 检查命令 | 解决方案 |
|------|---------|--------|
| Device 无法连接 | `netstat -tlnp \| grep 8888` | 检查 Server 是否启动 |
| 连接数不对 | `netstat -anp \| grep ESTABLISHED \| wc -l` | 检查 Device 是否全部启动 |
| 连接卡住 | `netstat -anp \| grep TIME_WAIT` | 检查 ConnectionClosedCb |
| Server 响应慢 | `netstat -anp \| grep 8888` | 检查是否有大量 TIME_WAIT |

---

## 🎯 对你的项目最相关的命令

```bash
# 1. 验证修复后的效果
netstat -anp | grep proxy_svr | grep ESTABLISHED

# 2. 监控 Device 连接变化
watch -n 1 'netstat -anp | grep ":8888" | grep ESTABLISHED | wc -l'

# 3. 检查连接清理是否正常
netstat -anp | grep 8888 | grep TIME_WAIT | wc -l

# 4. 完整的连接状态检查
netstat -anp | grep 8888
```

---

## 总结

- ✅ **监听检查**：`netstat -tlnp | grep 8888`
- ✅ **连接检查**：`netstat -anp | grep ESTABLISHED`
- ✅ **连接计数**：`netstat -anp | grep ESTABLISHED | wc -l`
- ✅ **实时监控**：`watch -n 1 'netstat -anp | grep ESTABLISHED | wc -l'`
- ✅ **异常检查**：`netstat -anp | grep -E "TIME_WAIT|CLOSE_WAIT"`

这些命令可以帮你快速诊断连接问题和验证你的修复是否有效！
