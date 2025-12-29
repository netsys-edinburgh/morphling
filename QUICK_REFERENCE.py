#!/usr/bin/env python3
"""
Quick reference: Virtual Time Synchronization

快速使用指南
"""

# ============================================================================
# 1. 最简单的用法（一条命令搞定）
# ============================================================================

# 同步日志
# python3 scripts/sync_virtual_time.py perf_merged.log --method earliest

# 分析效果
# python3 scripts/analyze_sync.py perf_merged.log perf_merged_synced.log

# ============================================================================
# 2. 理论背景
# ============================================================================

"""
问题: 不同设备的计算速度不同，导致 COMPUTE START 虚拟时间不对齐

示例：
  GEMM 0 的 COMPUTE START 虚拟时间：
  - Device 0: 2326719 us  ← 慢
  - Device 1: 2298721 us  
  - Device 2: 2215361 us  ← 快
  
  差异（Spread）= 2326719 - 2215361 = 111,358 us

解决方案: 添加校准偏移 (fx) 使所有设备对齐

  fx[Device 0] = -111358 us  (调整后 → 2215361)
  fx[Device 1] = -83360 us   (调整后 → 2215361)
  fx[Device 2] = 0 us        (调整后 → 2215361)
  
  结果: 所有设备现在都在虚拟时间 2215361 us 启动！
"""

# ============================================================================
# 3. 两种同步策略对比
# ============================================================================

"""
Strategy 1: EARLIEST (推荐)
  Baseline = min(all vt_start)
  用途: 性能分析、设备能力比较
  特点: 保守，以最快的设备为基准
  
  例: baseline = 2215361 us (Device 2)
      Device 0: offset = 2215361 - 2326719 = -111358 us (ahead)
      Device 1: offset = 2215361 - 2298721 = -83360 us  (ahead)
      Device 2: offset = 2215361 - 2215361 = 0 us       (baseline)

Strategy 2: LATEST
  Baseline = max(all vt_start)
  用途: 最坏情况分析、确保无掉队
  特点: 激进，以最慢的设备为基准
  
  例: baseline = 2326719 us (Device 0)
      Device 0: offset = 2326719 - 2326719 = 0 us        (baseline)
      Device 1: offset = 2326719 - 2298721 = 28000 us    (behind, advance)
      Device 2: offset = 2326719 - 2215361 = 111358 us   (behind, advance)
"""

# ============================================================================
# 4. 输出结果解读
# ============================================================================

"""
脚本执行后，你会看到这样的输出：

GEMM 0 - Earliest (conservative) synchronization:
  Baseline virtual time: 2215361 us  ← 选中的基准时间
  Device 0: vt_start=   2326719 us, offset=-111358 us (ahead)  
            └─ 这个设备比基准快 111358 us，需要"拖慢"
  Device 1: vt_start=   2298721 us, offset=-83360 us (ahead)
            └─ 这个设备比基准快 83360 us，需要"拖慢"
  Device 2: vt_start=   2215361 us, offset=     0 us (aligned)
            └─ 这个设备就是基准，不用调整

同步后的日志中，所有虚拟时间都加上相应的 offset：
  Device 0: vt_start = 2326719 + (-111358) = 2215361 us
  Device 1: vt_start = 2298721 + (-83360)  = 2215361 us
  Device 2: vt_start = 2215361 + 0         = 2215361 us
  ✓ 现在都对齐了！
"""

# ============================================================================
# 5. 快速分析脚本
# ============================================================================

"""
生成的 analyze_sync.py 会输出这样的统计信息：

OVERALL RESULTS:
  Total GEMM operations: 97
  Successfully synchronized: 97 (100%)          ← 全部成功同步
  Synchronization threshold: <100 us

这意味着：
- 97 个 GEMM 操作都被同步了
- 100% 的同步成功率
- 每个 GEMM 的虚拟时间差都小于 100 us
"""

# ============================================================================
# 6. 可视化输出
# ============================================================================

"""
analyze_sync.py 会生成 ASCII 时间线：

GEMM 0 - BEFORE SYNC:
  Device 0: ────────────────────────────────────────────────────────────────── ● 2326719 us
            （很长的虚线代表相对于最早设备的延迟）
            
  Device 1: ───────────────────────────────────────────────────────── ● 2298721 us
            （较短的虚线）
            
  Device 2: ─ ● 2215361 us
            （最短，说明这是最快的）

GEMM 0 - AFTER SYNC:
  Device 0: ─ ● 2215361 us  ✓  （✓ 代表同步成功）
  Device 1: ─ ● 2215361 us  ✓
  Device 2: ─ ● 2215361 us  ✓
  
所有设备现在都对齐了！
"""

# ============================================================================
# 7. 工作流程
# ============================================================================

"""
完整的工作流：

Step 1: 运行同步脚本
  $ python3 scripts/sync_virtual_time.py perf_merged.log --method earliest
  → 生成 perf_merged_synced.log
  
  输出：
  - GEMM 0-96 的偏移信息
  - 同步统计信息
  - 生成的输出文件路径

Step 2: 分析同步效果
  $ python3 scripts/analyze_sync.py perf_merged.log perf_merged_synced.log
  
  输出：
  - 整体同步率
  - 每个 GEMM 的前后对比
  - ASCII 时间线可视化
  - 改进百分比

Step 3: 使用同步后的日志
  - perf_merged_synced.log 可以用于后续分析
  - 虚拟时间现在已同步，可以进行公平的性能比较
"""

# ============================================================================
# 8. 常见场景
# ============================================================================

"""
场景 1: 快速检查是否需要同步
  $ python3 scripts/analyze_sync.py perf_merged.log perf_merged_synced.log | grep "OVERALL" -A 3
  
  如果 "Successfully synchronized" 是 100%，说明同步效果很好

场景 2: 找出最差的 GEMM 操作
  运行 analyze_sync.py，找 "Improvement:" 最大的 GEMM
  
  例: GEMM 96 的改进最大（123886 us）
      → 说明 GEMM 96 的设备差异最大

场景 3: 对比不同的同步策略
  $ python3 scripts/sync_virtual_time.py perf_merged.log --method earliest --output synced_earliest.log
  $ python3 scripts/sync_virtual_time.py perf_merged.log --method latest --output synced_latest.log
  $ python3 scripts/analyze_sync.py synced_earliest.log synced_latest.log
"""

# ============================================================================
# 9. 关键数字
# ============================================================================

"""
从实际运行结果中提取的关键数字：

总体：
  - 3,148 个 VTIME 事件
  - 2,342 个 Throughput 事件
  - 97 个 GEMM 操作
  - 3 个设备 (Device 0, 1, 2)

设备差异：
  - Device 0: 最慢（需要最大的负偏移）
  - Device 1: 中等速度（需要中等的负偏移）
  - Device 2: 最快（基准，偏移为 0）

偏移范围：
  - 最小: -123,886 us (GEMM 96 中 Device 0 最大延迟)
  - 最大: 0 us (所有 GEMM 中 Device 2 都是基准)
  
改进：
  - 平均每个 GEMM 改进: ~111,000 us
  - 总改进: 97 × 111,000 ≈ 10.8 ms
  → 整个训练过程累积减少了约 10.8 ms 的时间不同步
"""

# ============================================================================
# 10. 下一步
# ============================================================================

"""
同步虚拟时间后，你可以：

1. 性能分析
   - 比较设备在相同虚拟时间下的实际计算时间
   - 识别设备瓶颈

2. 负载均衡
   - 根据设备速度差异调整 GEMM 分割策略
   - 例: Device 0 可以分更少的数据

3. 时间建模
   - 建立虚拟时间 → 实际时间的映射
   - 用于模拟器的准确性验证

4. 系统优化
   - 针对最慢的设备进行优化
   - 例: Device 0 的计算优化可能收益最大
"""

# ============================================================================
# 快速命令参考
# ============================================================================

"""
最常用的命令：

# 基础同步（推荐）
python3 scripts/sync_virtual_time.py perf_merged.log --method earliest

# 分析同步效果
python3 scripts/analyze_sync.py perf_merged.log perf_merged_synced.log

# 同时运行两个（在 bash 中）
python3 scripts/sync_virtual_time.py perf_merged.log --method earliest && \\
python3 scripts/analyze_sync.py perf_merged.log perf_merged_synced.log

# 查看详细的 GEMM 偏移信息
python3 scripts/sync_virtual_time.py perf_merged.log --method earliest 2>&1 | grep "GEMM" -A 3

# 快速获取统计信息
python3 scripts/sync_virtual_time.py perf_merged.log --method earliest 2>&1 | tail -20
"""
