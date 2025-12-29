# Virtual Time Synchronization Guide

## 概述

这套脚本用于同步分布式设备仿真器中的虚拟时间。每个 GEMM（矩阵乘法）操作应该在所有设备上同时启动，但由于不同设备的速度差异，COMPUTE START 时间会有偏差。该脚本通过添加校准偏移 (calibration offset, fx) 来校准虚拟时间。

## 核心概念

### 问题

在多设备分布式训练中，即使网络是同步的，不同的计算能力也会导致虚拟时间不同步：

```
GEMM 0 的 COMPUTE START 时间:
  Device 0: vt_start = 2326719 us  ← 最慢（需要延迟）
  Device 1: vt_start = 2298721 us  ← 中等
  Device 2: vt_start = 2215361 us  ← 最快（基准）
  
Spread (不同步程度): 2326719 - 2215361 = 111,358 us
```

### 解决方案

通过添加校准偏移使所有设备在同一虚拟时间启动：

```
调整后的 vt_start:
  Device 0: 2326719 + offset(-111358) = 2215361 us  ✓ 同步
  Device 1: 2298721 + offset(-83360)  = 2215361 us  ✓ 同步
  Device 2: 2215361 + offset(0)       = 2215361 us  ✓ 同步
```

## 同步方法

### 方法选择

脚本提供两种同步方法：

#### 1. **Earliest（最早优先）- 推荐用于性能分析**
- **Baseline**: min(all vt_start_us)
- **策略**: 让所有设备与最快的设备对齐
- **优点**: 保守方式，不改变总时间跨度
- **使用场景**: 性能分析、公平比较不同速度的设备

```
baseline = min(2215361, 2298721, 2326719) = 2215361 us

offset[Device 0] = 2215361 - 2326719 = -111358 us  (ahead)
offset[Device 1] = 2215361 - 2298721 = -83360 us   (ahead)
offset[Device 2] = 2215361 - 2215361 = 0 us        (aligned)
```

#### 2. **Latest（最晚优先）- 推荐用于最坏情况分析**
- **Baseline**: max(all vt_start_us)
- **策略**: 让所有设备与最慢的设备对齐
- **优点**: 激进方式，确保所有设备都赶上
- **使用场景**: 最坏情况分析、确保无设备掉队

```
baseline = max(2215361, 2298721, 2326719) = 2326719 us

offset[Device 0] = 2326719 - 2326719 = 0 us        (aligned)
offset[Device 1] = 2326719 - 2298721 = 28000 us    (behind, need advance)
offset[Device 2] = 2326719 - 2215361 = 111358 us   (behind, need advance)
```

## 使用方法

### 步骤 1: 同步虚拟时间

```bash
cd /home/yufeng.xia/DeviceEmulator

# 使用 earliest 方法（推荐）
python3 scripts/sync_virtual_time.py perf_merged.log --method earliest

# 或使用 latest 方法
python3 scripts/sync_virtual_time.py perf_merged.log --method latest

# 指定输出文件
python3 scripts/sync_virtual_time.py perf_merged.log --method earliest --output my_synced.log
```

**输出**: `perf_merged_synced.log`

### 步骤 2: 分析同步效果

```bash
# 比较原始日志和同步后的日志
python3 scripts/analyze_sync.py perf_merged.log perf_merged_synced.log

# 输出会显示:
# - 每个 GEMM 的同步前后对比
# - 时间差减少的百分比
# - 同步状态（是否在 100us 以内）
# - ASCII 时间线可视化
```

## 日志格式

### 输入格式 (perf_merged.log)

#### VTIME 日志
```
VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,vt_duration_us

示例：
VTIME,1765380680827991,1,0,COMPUTE,START,2298721,2298721,0
       │                 │ │         │       │       └─ vt_start（虚拟时间开始）
       │                 │ │         │       └─ 虚拟时间结束
       │                 │ │         └─ 事件（START/END）
       │                 │ └─ 阶段（COMPUTE/SEND/RECEIVE）
       │                 └─ gemm_id（全局 GEMM 操作 ID）
       │ └─ device_id（设备 ID）
       └─ timestamp_us（系统时间戳，微秒）
```

#### Throughput 日志
```
timestamp_us,device_id,gemm_id,direction,bytes,throughput_b_s,epoch_start_us,epoch_end_us,packet_duration_us

示例：
1765380680829650,1,0,DOWNLOAD,131154,124552.71,0,1765380680829631,1765380680829631
```

### 输出格式 (perf_merged_synced.log)

- 与输入格式相同
- 所有 `vt_start_us` 和 `vt_end_us` 都已应用偏移调整
- Throughput 日志中的 `epoch_start_us` 和 `epoch_end_us` 也已调整

## 脚本详解

### sync_virtual_time.py

**功能**: 读取合并的性能日志，计算每个设备的校准偏移，应用偏移生成同步后的日志。

**关键函数**:

1. `find_compute_start_times()`: 提取所有 COMPUTE START 事件的虚拟时间
   
2. `calculate_offsets()`: 计算校准偏移
   - 对于每个 gemm_id，找出所有设备的 COMPUTE START 时间
   - 选择 baseline（min 或 max）
   - offset[device] = baseline - vt_start[device]
   
3. `apply_offsets()`: 应用偏移到所有虚拟时间值
   - 修改 vt_start_us, vt_end_us（持续时间不变）
   - 修改 epoch_start_us, epoch_end_us（如果有效）

4. `write_synchronized_log()`: 写入同步后的日志（保持时间顺序）

### analyze_sync.py

**功能**: 对比原始日志和同步后日志，生成详细的同步分析报告。

**输出指标**:

| 指标 | 说明 |
|------|------|
| Spread (max-min) | 不同设备虚拟时间的差异 |
| Improvement | 同步后的改进量（微秒）|
| Improvement % | 同步后的改进百分比 |
| is_synchronized | 是否满足同步条件（< 100us）|

**可视化**:

ASCII 时间线展示设备之间的时间差异：

```
GEMM 0 - BEFORE SYNC:
  Device 0: ─────────────────────────────────────────────────────────────── ● 2326719 us
  Device 1: ───────────────────────────────────────────────────────── ● 2298721 us
  Device 2: ─ ● 2215361 us

GEMM 0 - AFTER SYNC:
  Device 0: ─ ● 2215361 us  ✓
  Device 1: ─ ● 2215361 us  ✓
  Device 2: ─ ● 2215361 us  ✓
```

## 实际例子

### 完整示例

```bash
# 1. 生成同步日志
python3 scripts/sync_virtual_time.py perf_merged.log --method earliest

输出：
======================================================================
VIRTUAL TIME SYNCHRONIZATION STATISTICS
======================================================================

Total VTIME events: 3148
Total throughput events: 2342
Total GEMM operations synchronized: 97

Devices:
  Device 0: 97 GEMM operations
  Device 1: 97 GEMM operations
  Device 2: 97 GEMM operations

Offsets applied:
  Devices with non-zero offsets: 194
  Offset range: -123886 to 0 us

✓ Synchronization complete!
Output: perf_merged_synced.log
======================================================================

# 2. 分析同步效果
python3 scripts/analyze_sync.py perf_merged.log perf_merged_synced.log

输出：
==========================================================================================
VIRTUAL TIME SYNCHRONIZATION ANALYSIS
==========================================================================================

OVERALL RESULTS:
  Total GEMM operations: 97
  Successfully synchronized: 97 (100%)
  Synchronization threshold: <100 us

DETAILED SYNCHRONIZATION METRICS BY GEMM OPERATION:
...

TIMELINE VISUALIZATION:
...
```

## 校准偏移解释

### 正偏移 (Positive Offset)

```
Spread before: Device 0 ahead of Device 2 by 111358 us
Offset: -111358 us  (negative because Device 0 is ahead)
Action: 将 Device 0 的虚拟时间向后调整
```

### 负偏移 (Negative Offset)

```
Spread before: Device 1 ahead of Device 2 by 83360 us
Offset: -83360 us   (negative because Device 1 is ahead)
Action: 将 Device 1 的虚拟时间向后调整
```

### 零偏移 (Zero Offset)

```
Device 2 已经是基准，无需调整
Offset: 0 us
Action: 保持不变
```

## 常见问题

### Q1: 应该用 earliest 还是 latest？

**A**: 
- **Earliest**: 用于性能分析。假设最快的设备能力，其他设备赶不上
- **Latest**: 用于最坏情况分析。假设最慢的设备是瓶颈，其他设备需要等待

**推荐**：默认用 earliest（保守）

### Q2: 为什么有些 GEMM 的偏移模式一致？

**A**: Device 0 和 Device 1 比 Device 2 快，且快的程度基本一致，所以偏移量相对稳定。

```
GEMM 0: Device 0 offset -111358, Device 1 offset -83360
GEMM 1: Device 0 offset -111209, Device 1 offset -80110
GEMM 2: Device 0 offset -111713, Device 1 offset -80836
...
```

### Q3: 同步后可以用来做什么？

**A**: 
1. **公平性分析**: 比较设备在相同虚拟时间条件下的性能
2. **负载均衡**: 识别哪些设备总是更慢
3. **性能建模**: 建立虚拟时间与实际墙钟时间的映射关系
4. **系统优化**: 针对瓶颈设备进行优化

## 扩展应用

### 计算有效吞吐量

```python
# 在同步后的日志中，所有设备在相同虚拟时间启动
# 可以计算真实的并行度和吞吐量

for gemm_id in range(0, 97):
    vt_starts_synced = extract_synced_vt_starts(gemm_id)
    # 所有设备现在有相同的 vt_start
    # 可以分析它们的完成时间差异
```

### 多轮迭代的累积偏差

```
Iteration 1: offset = [-111358, -83360, 0]
Iteration 2: offset = [-111209, -80110, 0]
...
Iteration 97: offset = [-123886, 0, -21426]  ← 偏移模式可能改变！

这反映了不同 GEMM 操作的性能变化
```

## 文件清单

```
scripts/
├── sync_virtual_time.py      ← 主同步脚本
├── analyze_sync.py            ← 分析脚本
└── merge_perf_logs.py         ← 日志合并脚本（已有）

输出文件:
├── perf_merged.log            ← 合并的原始日志
├── perf_merged_synced.log     ← 同步后的日志（新建）
```

## 性能指标总结

从最后的统计可以看到：

- **97 个 GEMM 操作** 全部成功同步
- **100% 的同步率**（所有 GEMM 的虚拟时间差 < 100 us）
- **平均改进**: ~111,000 us（每个 GEMM）
- **总改进**: 97 × 111,000 = ~10.8 ms（整个训练）

这说明在没有同步的情况下，97 轮迭代累积了约 10.8ms 的设备时间不同步，通过校准可以完全消除。
