# Virtual Time Synchronization in Multi-Device Distributed Training

---

## Slide 1: Title Slide

**Virtual Time Synchronization in Multi-Device Distributed Training Simulators**

- Implementation of Global Virtual Time Synchronization Mechanism
- Performance Analysis & Device Load Balancing Foundation
- *December 2024*

---

## Slide 2: Motivation

### Problem: Device Timing Misalignment

- **Heterogeneous Device Performance**: Different devices compute at different speeds
  - Device 2: ~111 μs faster per GEMM than Device 0
  - Device 1: ~31 μs faster per GEMM than Device 0
  
- **Unaligned Virtual Times**: GEMM 0 starts at different times across devices
  - Device 0: 2,326,719 μs
  - Device 1: 2,298,721 μs
  - Device 2: 2,215,361 μs
  - ❌ These should all be synchronized!

- **Analysis Challenge**: Cannot accurately compare device behavior without synchronized timelines
  - Difficult to identify true bottlenecks
  - Cannot measure real load imbalance
  - Complicates performance debugging

---

## Slide 3: Solution Approach

### Virtual Time Synchronization Framework

- **Per-GEMM Synchronization**: Each matrix multiplication operation synchronized independently
  - 97 GEMM operations in trace
  - Device performance varies slightly per operation

- **Dual-Strategy Design**:
  - **LATEST Method** (default): Align to slowest device → realistic barrier synchronization
  - **EARLIEST Method**: Align to fastest device → theoretical best-case

- **Three-Phase Process**:
  1. **Parse & Analyze**: Extract COMPUTE START times
  2. **Calculate Offsets**: Determine per-device calibration
  3. **Normalize & Validate**: Shift times to start from 0, verify correctness

---

## Slide 4: Technical Approach - LATEST Method

### Algorithm Overview

**For each GEMM operation:**
- Baseline = MAX(device_vt_start_times)  → slowest device
- Offset[device] = baseline - device_vt_start
- Apply offset to all virtual time fields

**Global Normalization:**
- Find minimum vt_start across entire trace
- Shift all times: final_vt = current_vt - min_vt
- Result: Timeline starts at 0 μs

**Why LATEST?**
- ✅ Matches real distributed training (barrier synchronization)
- ✅ Identifies actual bottleneck device
- ✅ Lower total offset burden (13.8M vs 18.6M μs)

---

## Slide 5: GEMM 0 Synchronization Example

### Before Synchronization (Misaligned)

| Device | vt_start_us | Issue |
|--------|------------|-------|
| 0 | 2,326,719 | Slowest |
| 1 | 2,298,721 | 28 μs faster |
| 2 | 2,215,361 | 111 μs faster |

### After LATEST Synchronization

| Device | Original | Offset | Synced |
|--------|----------|--------|--------|
| 0 | 2,326,719 | 0 | **2,326,719** ✓ |
| 1 | 2,298,721 | +27,998 | **2,326,719** ✓ |
| 2 | 2,215,361 | +111,358 | **2,326,719** ✓ |

✅ **All devices now start GEMM 0 at the same virtual time!**

---

## Slide 6: Results Summary

### Synchronization Performance

| Metric | Value |
|--------|-------|
| VTIME Events | 3,148 |
| Throughput Events | 2,342 |
| GEMM Operations | 97 |
| **Synchronization Success Rate** | **100%** ✅ |

### Device Performance Heterogeneity

| Device | Avg Offset | Total Offset | Implication |
|--------|-----------|-------------|------------|
| 0 | 0 μs (baseline) | 0 | Bottleneck |
| 1 | 31.2 ms/GEMM | 3.0M μs | ~31 ms faster |
| 2 | 111.0 ms/GEMM | 10.8M μs | ~111 ms faster |

**Key Insight**: Device 0 limits overall throughput by ~111 ms per GEMM operation

---

## Slide 7: LATEST vs EARLIEST Comparison

### Method Comparison

| Aspect | LATEST | EARLIEST |
|--------|--------|----------|
| Baseline | MAX(vt_start) | MIN(vt_start) |
| Aligns To | Slowest device | Fastest device |
| Offsets | Positive (accelerate) | Negative (decelerate) |
| Total Offset | 13.8M μs | 18.6M μs |
| Real-world Match | ✅ Yes (barrier sync) | ❌ No (unrealistic) |

### Why LATEST is Better

- Represents actual distributed training behavior
- Clear identification of bottleneck device
- More conservative estimate of performance
- ~26% lower offset burden

---

## Slide 8: Data Integrity & Validation

### What We Preserve

✅ **All Virtual Time Dimensions**:
- `vt_start_us`: Operation start time
- `vt_end_us`: Operation end time  
- `vt_duration_us`: Duration (unchanged)
- `epoch_start_us` / `epoch_end_us`: Related timing

✅ **Data Quality**:
- Chronological order maintained
- 100% event preservation (no data loss)
- Consistent offset application across all events
- Automatic normalization to zero-based timeline

### Validation Results

```
✓ 3,148 VTIME events synchronized
✓ 2,342 throughput events synchronized
✓ 291 device-GEMM pairs verified
✓ All devices aligned at same virtual times
✓ Timeline normalized to start from 0
```

---

## Slide 9: Scientific Impact

### What This Enables

1. **Accurate Performance Analysis**
   - Compare device behaviors at synchronized virtual times
   - Identify true performance bottlenecks
   - Quantify load imbalance precisely

2. **Optimization Foundation**
   - Baseline for measuring improvement
   - Test scheduling strategies on synchronized timeline
   - "What-if" analysis: simulate load balancing

3. **Reproducible Research**
   - Consistent timeline for analysis
   - Shareable synchronized logs
   - Clear metrics for comparing optimizations

### Practical Applications

- Device scaling studies (3 → N devices)
- Load balancing algorithm development
- Simulator validation against real training
- Hardware heterogeneity analysis

---

## Slide 10: Technical Implementation

### Code Structure

```
sync_virtual_time.py (Main Engine - 377 lines)
├── Parse VTIME & throughput events
├── Extract COMPUTE START times per device
├── Calculate offsets (LATEST/EARLIEST methods)
├── Apply offsets to all virtual times
├── Normalize to zero-based timeline
└── Maintain chronological order

Supporting Tools:
├── verify_sync_correctness.py    → Validation & detailed analysis
├── compare_sync_methods.py       → Strategy comparison
└── analyze_sync.py               → Performance metrics
```

### Key Features

- Dual-method support: `--method latest|earliest`
- Automatic global normalization
- 100% data preservation
- Comprehensive verification & statistics

---

## Slide 11: Next Steps

### Immediate Applications

- Detailed communication pattern analysis with aligned timelines
- Systematic device bottleneck identification
- Load balancing strategy evaluation

### Medium Term

- Integrate synchronized logs into simulator
- Develop automated optimization algorithms
- Multi-device scaling studies (3 → 8, 16, 32 devices)

### Long Term

- Real-time synchronization during simulation
- Interactive timeline visualization
- Machine learning-based load balancing

---

## Slide 12: Key Takeaways

### What We Delivered

✅ **Virtual Time Logging System**
- Comprehensive event capture at microsecond precision
- Both computation and communication tracking

✅ **Global Synchronization Mechanism**  
- Automatic alignment of device timelines
- 100% success rate (97 GEMMs, 291 device-pairs)
- Two strategies for different analysis goals

✅ **Analysis Framework**
- Verification tools and validation
- Performance metrics & detailed statistics
- Ready for optimization research

### Impact

**This work establishes the foundation for systematic performance analysis and optimization of distributed training simulators with heterogeneous devices.**

### Key Numbers

- 97 GEMM operations analyzed
- 3,148 events synchronized  
- 111 ms per-GEMM performance difference (bottleneck identified)
- 13.8 ms total system optimization potential

---

## Slide 13: Questions?

**How This Helps Your Research:**

- ❓ *What's the bottleneck?* → Device 0 (111 ms slower)
- ❓ *How much time is wasted?* → ~13.8 ms per operation
- ❓ *Can we improve?* → Yes, with load balancing
- ❓ *How do we scale?* → Use synchronized timeline as baseline

**Open for Discussion** 💡

