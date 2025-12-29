# GEMM ID 日志问题诊断和修复

## 问题诊断

根据你提供的日志，发现了几个问题：

### 1. **gemm_id 全部为 0**（最严重的问题）

**原因**：虽然代码中有以下实现：
- ✅ `gemm_id` 字段定义在 `MatrixPartition` 结构体中
- ✅ `gemm_id_count_` 原子计数器在 `ProxySvrImpl` 中定义
- ✅ `DispatchMatMulAsync()` 中有 `partition->gemm_id = gemm_id_count_;`
- ✅ 操作后有 `gemm_id_count_++;` 递增

但你的日志中所有 `gemm_id` 值都是 0，说明**代码还没有重新编译和部署**。

**解决方案**：
```bash
# 重新编译项目
cd /home/yufeng.xia/DeviceEmulator
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 2. **日志文件缺少表头注释**

你的日志没有表头说明，`merge_perf_logs.py` 期望看到格式说明。

**修复信息**：
- ✅ 已更新 `csrc/backend/device_tracker.cc` 的 `InitSeparatePerfLog()` 函数
- ✅ 现在会在日志文件开头写入格式说明注释：
  ```
  # VTIME format: VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,vt_duration_us
  # Throughput format: timestamp_us,device_id,gemm_id,direction,bytes,throughput_b_s,epoch_start_us,epoch_end_us,packet_duration_us
  ```

### 3. **merge_perf_logs.py 脚本问题**

脚本在第 24 行有语法错误（多余的反引号）。

**修复信息**：
- ✅ 已修复语法错误
- ✅ 更新了注释说明新的日志格式（包含 gemm_id）
- ✅ 改进了表头处理逻辑

## 日志格式规范

### VTIME 日志格式
```
VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,vt_duration_us
```

**字段说明**：
- `VTIME`：事件类型标识
- `timestamp_us`：系统时间戳（微秒）
- `device_id`：设备 ID（0, 1, 2, ...）
- `gemm_id`：全局 GEMM 操作 ID（0, 1, 2, ...，每个 DispatchMatMulAsync 调用递增一次）
- `phase`：阶段（COMPUTE, RECEIVE, SEND）
- `event`：事件（START, END）
- `vt_start_us`：虚拟时间开始（微秒）
- `vt_end_us`：虚拟时间结束（微秒）
- `vt_duration_us`：虚拟时间持续时间（微秒）

**示例**：
```
VTIME,1765380077213829,2,0,SEND,END,10098658,10100432,1774
VTIME,1765380077218046,2,0,COMPUTE,START,3024656,3024656,0
VTIME,1765380077218874,2,0,COMPUTE,END,3024656,3025486,830
```

### Throughput 日志格式
```
timestamp_us,device_id,gemm_id,direction,bytes,throughput_b_s,epoch_start_us,epoch_end_us,packet_duration_us
```

**字段说明**：
- `timestamp_us`：系统时间戳（微秒）
- `device_id`：设备 ID
- `gemm_id`：全局 GEMM 操作 ID（对应该数据传输的 GEMM 操作）
- `direction`：方向（UPLOAD, DOWNLOAD）
- `bytes`：数据量（字节）
- `throughput_b_s`：吞吐量（字节/秒）
- `epoch_start_us`：实际传输开始时间（微秒）
- `epoch_end_us`：实际传输结束时间（微秒）
- `packet_duration_us`：包传输时间（微秒）

**示例**：
```
1765380077219527,2,0,DOWNLOAD,131153,70022.96,1765380077219482,1765380077219482,0
1765380077335074,2,1,DOWNLOAD,131154,131878.83,1765380077219482,1765380077335065,115583
```

## 修改的文件清单

### C++ 源代码
1. **csrc/backend/proxy_svr.cc**（两处）
   - 在 `DispatchMatMulAsync()` 的 DEBUG 日志中添加 `gemm_id` 输出
   - 在 INFO 日志中显示当前 `gemm_id_count_` 值

2. **csrc/backend/device_tracker.cc**
   - 更新 `InitSeparatePerfLog()` 的表头，添加格式说明注释

3. **morphling/ops/csrc/backend/device_tracker.cc**（同步）
   - 同步了表头更新

### Python 脚本
1. **scripts/merge_perf_logs.py**
   - 修复语法错误
   - 更新注释说明格式
   - 改进表头处理

## 调试建议

### 1. 检查编译输出
重新编译后，检查是否有关于 `gemm_id` 的警告或错误。

### 2. 查看服务器日志
运行程序后，查看服务器日志中的这些信息：
```
[DispatchMatMulAsync] Created IDLE partitions in ... gemm_id_count=X
```
确认 `gemm_id_count` 在正确递增。

### 3. 验证日志格式
运行第一个 GEMM 操作后，查看生成的日志文件：
```bash
head -20 logs/perf_server.log
head -20 logs/perf_device_0.log
```

应该看到：
- 前两行是格式说明注释
- VTIME 日志中第 4 个字段（gemm_id）从 0 开始递增
- Throughput 日志中第 3 个字段（gemm_id）与对应的 VTIME 匹配

### 4. 运行 merge 脚本
```bash
python3 scripts/merge_perf_logs.py logs/ logs/perf_merged.log
```

检查输出中的统计信息，应该显示：
- VTIME events: N
- Throughput events: M
- Headers/Comments: 2

## 预期修复效果

修复后，你应该看到：

**日志文件开头**（添加表头注释）：
```
# VTIME format: VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,vt_duration_us
# Throughput format: timestamp_us,device_id,gemm_id,direction,bytes,throughput_b_s,epoch_start_us,epoch_end_us,packet_duration_us
```

**日志数据**（gemm_id 值递增）：
```
VTIME,1765380077213829,2,0,SEND,END,10098658,10100432,1774     ← gemm_id=0（第一次操作）
VTIME,1765380077329539,2,1,SEND,END,10215776,10216144,368     ← gemm_id=1（第二次操作）
VTIME,1765380077451254,2,2,SEND,END,10337505,10337859,354     ← gemm_id=2（第三次操作）
```

## 后续步骤

1. ✅ **代码修改完成**（已在此次更新中完成）
2. ⏳ **重新编译**（需要用户执行）
3. ⏳ **部署并运行**（需要用户执行）
4. ⏳ **验证日志**（用户需检查）
5. ⏳ **运行 merge_perf_logs.py**（用户验证）
6. ⏳ **多设备时间同步分析**（用户最终目标）
