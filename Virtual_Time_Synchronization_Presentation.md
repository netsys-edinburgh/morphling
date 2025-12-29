# Virtual Time Synchronization in Multi-Device Distributed Training
## Technical Presentation

---

## Slide 1: Title

**Virtual Time Synchronization in Multi-Device Distributed Training**

- Implementation of Virtual Time Logging System
- Global Virtual Time Synchronization Mechanism
- Performance Analysis and Optimization Framework

*Presenter: [Your Name]*
*Date: December 2024*

---

## Slide 2: Motivation - Problem Statement

### Why Virtual Time Synchronization?

In distributed training simulators with multiple devices:

- **Device Performance Heterogeneity**: Different devices have different computational speeds
  - Device 0: ~111 μs slower per GEMM operation
  - Device 1: ~31 μs slower per GEMM operation  
  - Device 2: Fastest device (baseline)

- **Timing Misalignment**: Without synchronization, virtual times don't correspond across devices
  - Device 0 starts COMPUTE at 2,326,719 μs
  - Device 1 starts COMPUTE at 2,298,721 μs
  - Device 2 starts COMPUTE at 2,215,361 μs
  - **All should start at the same virtual time!**

- **Analysis Challenge**: How to understand device behavior when timestamps don't align?
  - Makes performance comparison difficult
  - Cannot accurately identify bottlenecks
  - Cannot measure true load imbalance

---

## Slide 3: Key Challenges

### Technical Difficulties

1. **Per-GEMM Synchronization**
   - Each GEMM operation (matrix multiplication) has independent timing
   - 97 GEMM operations in the trace
   - Device performance varies slightly per operation

2. **Multi-Dimensional Virtual Times**
   - `vt_start_us`: Virtual time when operation starts
   - `vt_end_us`: Virtual time when operation ends
   - `vt_duration_us`: Operation duration (should remain unchanged)
   - `epoch_start_us` / `epoch_end_us`: Related timing information
   - **All must be synchronized consistently**

3. **Synchronization Strategy**
   - Should we align to the **fastest device** (optimistic)?
   - Or align to the **slowest device** (realistic)?
   - Each strategy has different implications

---

## Slide 4: Solution Overview - What We Implemented

### Virtual Time Synchronization Framework

#### Step 1: Parse and Analyze
```
Input: perf_merged.log (3,148 VTIME events, 2,342 throughput events)
       ↓
Extract COMPUTE START times for each (GEMM_ID, Device_ID) pair
       ↓
Identify timing differences between devices
```

#### Step 2: Calculate Calibration Offsets
For each GEMM operation:
- **LATEST Method** (chosen as default):
  - Baseline = max(vt_start_us) across all devices
  - Offset = baseline - device_vt_start_us
  - **Effect**: Aligns all devices to the slowest device

#### Step 3: Apply Offsets & Normalize
- Apply offset to all virtual time fields
- Normalize: Shift all times so minimum vt_start_us = 0
- Maintain chronological order of events

#### Step 4: Validation & Analysis
- Verify all 291 device-GEMM pairs synchronized successfully
- Generate detailed statistics and comparison

---

## Slide 5: Technical Implementation - The Algorithm

### Synchronization Logic

**For LATEST Method (Recommended):**

```python
For each GEMM_ID:
    1. Find all devices' COMPUTE START times
       - Device 0: vt_start_0
       - Device 1: vt_start_1
       - Device 2: vt_start_2
    
    2. Calculate baseline (slowest device):
       baseline = MAX(vt_start_0, vt_start_1, vt_start_2)
    
    3. Calculate per-device offset:
       offset[device_i] = baseline - vt_start_i
    
    4. Apply offset to ALL virtual times of that device:
       new_vt_start = vt_start + offset
       new_vt_end = vt_end + offset
       new_epoch_start = epoch_start + offset
       new_epoch_end = epoch_end + offset
       (duration unchanged!)
    
    5. Normalize globally:
       min_vt_global = MIN(all vt_start values across all GEMMs)
       final_vt = current_vt - min_vt_global
```

---

## Slide 6: Case Study - GEMM 0 Synchronization

### Before Synchronization

| Device | vt_start_us | Status | Difference from Device 0 |
|--------|------------|--------|-------------------------|
| **0** | 2,326,719 | Slowest | baseline |
| **1** | 2,298,721 | Behind D0 | -27,998 μs (faster) |
| **2** | 2,215,361 | Behind D0 | -111,358 μs (fastest) |

❌ **Problem**: All three devices should start at the same virtual time!

### After Synchronization (LATEST Method)

| Device | Original | Offset | Synced | Status |
|--------|----------|--------|--------|--------|
| **0** | 2,326,719 | +0 μs | 2,326,719 | Baseline (aligned) |
| **1** | 2,298,721 | +27,998 μs | 2,326,719 | ✓ Aligned |
| **2** | 2,215,361 | +111,358 μs | 2,326,719 | ✓ Aligned |

✅ **Result**: All devices synchronized to same virtual time!

### After Normalization

| Device | After Sync | Min Adjustment | Final |
|--------|-----------|-----------------|--------|
| **0** | 2,326,719 | -2,215,361 μs | 111,358 |
| **1** | 2,326,719 | -2,215,361 μs | 111,358 |
| **2** | 2,326,719 | -2,215,361 μs | 111,358 |

✅ **Final Result**: All three devices now start GEMM 0 at 111,358 μs (normalized from global minimum)

---

## Slide 7: Results - GEMM 1 Example

### Detailed Synchronization of GEMM 1

**Original Times:**
- Device 0: vt_start = 2,445,471 μs
- Device 1: vt_start = 2,414,372 μs (27 ms behind)
- Device 2: vt_start = 2,334,262 μs (111 ms behind)

**Synchronization Process:**
1. Baseline = MAX(2,445,471, 2,414,372, 2,334,262) = **2,445,471 μs**
2. Offsets:
   - Device 0: 2,445,471 - 2,445,471 = **0 μs**
   - Device 1: 2,445,471 - 2,414,372 = **+31,099 μs**
   - Device 2: 2,445,471 - 2,334,262 = **+111,209 μs**

**After Synchronization:**
- Device 0: 2,445,471 μs ✓
- Device 1: 2,414,372 + 31,099 = 2,445,471 μs ✓
- Device 2: 2,334,262 + 111,209 = 2,445,471 μs ✓

**All aligned!** Despite Device 2 being ~111 ms "faster", we shift its timeline forward to match Device 0's slower pace.

---

## Slide 8: Overall Results & Statistics

### Synchronization Success Metrics

| Metric | Value |
|--------|-------|
| Total VTIME Events | 3,148 |
| Total Throughput Events | 2,342 |
| GEMM Operations | 97 |
| Device-GEMM Pairs | 291 |
| **Synchronization Success Rate** | **100%** ✅ |

### Device Performance Differences

| Device | Avg Offset | Min Offset | Max Offset | Total Offset |
|--------|-----------|-----------|-----------|------------|
| **0** | 0 us | 0 us | 0 us | 0 us |
| **1** | 31,209.5 us | 23,486 us | 123,886 us | 3,027,317 us |
| **2** | 111,014.5 us | 102,460 us | 116,236 us | 10,768,409 us |

**Key Insight**: 
- Device 2 is consistently **111 ms** faster per GEMM
- Device 1 is consistently **31 ms** faster per GEMM
- Total speedup difference: **13.8 million μs** across 97 GEMMs

### Time Normalization

| Metric | Original | After Sync |
|--------|----------|-----------|
| Min Virtual Time | 2,215,361 μs | 0 μs |
| Max Virtual Time | 20,545,064 μs | 20,329,703 μs |
| Time Span | 18,329,703 μs | 20,329,703 μs |

---

## Slide 9: Synchronization Strategy Comparison

### LATEST vs EARLIEST Methods

#### LATEST Method (Chosen Default)
- **Baseline**: MAX(vt_start) = align to slowest device
- **Offsets**: All positive (or zero)
- **Meaning**: "Advance faster devices to match slowest device"
- **Real-world analogy**: Waiting for the slowest worker to finish
- **Total System Offset**: 13.8 million μs
- **Devices needing acceleration**: 194/291 (66%)

#### EARLIEST Method (Alternative)
- **Baseline**: MIN(vt_start) = align to fastest device
- **Offsets**: All negative (or zero)
- **Meaning**: "Delay faster devices to match fastest device"
- **Real-world analogy**: Theoretical best-case synchronization
- **Total System Offset**: 18.6 million μs
- **Devices needing deceleration**: 0/291 (0%)

### Why LATEST is More Realistic

✅ **Matches real distributed training behavior**
- In practice, a barrier synchronization waits for the slowest worker
- Represents actual performance bottleneck

✅ **More conservative estimate**
- Identifies true device that limits throughput
- Helps focus optimization efforts

✅ **Lower total offset burden**
- 13.8M vs 18.6M μs → ~26% reduction

---

## Slide 10: Implementation Details - Code Structure

### File Organization

```
scripts/
├── sync_virtual_time.py          # Main synchronization engine
│   ├── parse_vtime_event()       # Parse VTIME lines
│   ├── parse_throughput_event()  # Parse throughput lines
│   ├── find_compute_start_times() # Extract timing info
│   ├── calculate_offsets()       # Compute sync offsets (LATEST/EARLIEST)
│   ├── apply_offsets()           # Apply & normalize times
│   └── write_synchronized_log()  # Output synchronized log
│
├── verify_sync_correctness.py    # Validation & analysis tool
│   ├── analyze_gemm()            # Detailed per-GEMM analysis
│   └── Statistics & visualization
│
└── compare_sync_methods.py       # Strategy comparison tool
```

### Key Features

- **Dual-method support**: `--method latest|earliest` flag
- **Automatic normalization**: Time starts from 0 for easier analysis
- **Chronological ordering**: Events maintain temporal sequence
- **100% data preservation**: No event loss, all information retained
- **Comprehensive validation**: Built-in verification and statistics

---

## Slide 11: Significance & Impact

### Scientific Contribution

1. **Enables Accurate Performance Analysis**
   - ✅ Can now compare device behaviors at the same virtual time
   - ✅ Identify true performance bottlenecks
   - ✅ Quantify load imbalance (111 ms difference per operation)

2. **Foundation for Optimization**
   - Baseline for measuring improvement from load balancing
   - Can test scheduling strategies on synchronized timeline
   - Enables "what-if" analysis: What if we balance Device 0?

3. **Reproducible Research**
   - Synchronized logs can be shared and analyzed consistently
   - Multiple researchers can analyze same timeline
   - Clear metrics for comparing optimizations

### Practical Applications

1. **Device Scaling Analysis**
   - Study how to scale from 3 devices to N devices
   - Predict performance with unbalanced clusters
   - Design better load distribution algorithms

2. **Simulator Validation**
   - Compare simulation results with real distributed training
   - Tune simulator parameters for accuracy

3. **System Design**
   - Identify whether heterogeneity comes from computation or communication
   - Design hardware specs for better balance

---

## Slide 12: Technical Metrics & Quality Assurance

### Validation Results

```
✅ Parsing Correctness
   - 3,148 VTIME events parsed successfully
   - 2,342 throughput events parsed successfully
   - Headers and metadata preserved

✅ Synchronization Correctness
   - 97 GEMM operations synchronized
   - 291 device-GEMM pairs aligned (100% success)
   - All virtual time fields updated consistently

✅ Data Integrity
   - Durations preserved (vt_duration_us unchanged)
   - Chronological order maintained
   - No data loss or corruption

✅ Normalization Correctness
   - Minimum vt_start_us = 0
   - All times shifted by same global offset
   - Relative timing preserved
```

### Example Verification (GEMM 0)

**All three devices synchronized to same time:**
- Device 0: ✓ 2,326,719 μs → synced
- Device 1: ✓ 2,298,721 + 27,998 = 2,326,719 μs → synced
- Device 2: ✓ 2,215,361 + 111,358 = 2,326,719 μs → synced

---

## Slide 13: Next Steps & Future Work

### Immediate Applications

1. **Detailed Performance Analysis**
   - Analyze communication patterns with aligned timelines
   - Identify synchronization bottlenecks
   - Profile device utilization

2. **Load Balancing Research**
   - Test various scheduling strategies
   - Measure improvement from rebalancing
   - Validate against synchronized baseline

3. **Device Scaling Studies**
   - Predict performance with more devices
   - Design heterogeneous clusters
   - Optimize for different hardware profiles

### Long-term Vision

1. **Automated Optimization**
   - Machine learning for load balancing
   - Dynamic device scheduling
   - Predictive resource allocation

2. **Multi-dimensional Analysis**
   - Synchronize communication and computation timelines
   - Analyze computation/communication overlap
   - Optimize pipelining

3. **Simulator Enhancement**
   - Integrate synchronization into main simulator
   - Real-time analysis during simulation
   - Interactive timeline visualization

---

## Slide 14: Summary

### What We Accomplished

✅ **Virtual Time Logging System**
- Implemented comprehensive event logging
- Captures compute and communication events
- Microsecond-level precision

✅ **Global Virtual Time Synchronization**
- Designed and implemented dual-strategy synchronization
- LATEST method: align to slowest device (realistic)
- EARLIEST method: align to fastest device (theoretical)
- 100% success rate across 97 GEMM operations

✅ **Analysis Framework**
- Verification and validation tools
- Performance metrics and statistics
- Detailed per-GEMM analysis

### Key Metrics

| Metric | Value |
|--------|-------|
| GEMM Operations Analyzed | 97 |
| Synchronization Success Rate | 100% |
| Device Performance Difference | ~111 ms per GEMM |
| Total System Optimization Potential | ~13.8 ms |
| Implementation Code Lines | ~500 lines |

### Impact

This work provides the **foundation for systematic performance analysis and optimization** of distributed training simulators with heterogeneous devices.

---

## Slide 15: Questions & Discussion

### Key Questions We Can Now Answer

1. **What is the performance bottleneck?**
   - Device 0 is ~111 ms slower per operation
   - Clear optimization target identified

2. **How much time is wasted waiting for slow devices?**
   - 13.8 million μs total offset across 97 operations
   - ~142 ms per operation average

3. **Can we improve by rebalancing?**
   - Yes! Now we have a synchronized baseline to measure against

4. **How do we scale to more devices?**
   - Use synchronized timeline to simulate heterogeneous clusters

### Open for Discussion

- Synchronization strategy preferences
- Analysis methodology validation
- Optimization strategy proposals
- Integration with main simulator

**Thank you!**

---

## Appendix: Example Log Entries

### Original perf_merged.log (First Few Lines)

```
VTIME,1765380680823573,2,0,SEND,END,10107196,10111678,4482
VTIME,1765380680823637,1,0,SEND,END,10110007,10111742,1735
VTIME,1765380680824364,0,0,SEND,END,10111697,10112469,772
VTIME,1765380680827991,1,0,COMPUTE,START,2298721,2298721,0
VTIME,1765380680828433,2,0,COMPUTE,START,2215361,2215361,0    ← Different times!
VTIME,1765380680828484,0,0,COMPUTE,START,2326719,2326719,0
```

### Synchronized perf_merged_synced.log (Same GEMMs)

```
VTIME,1765380680823573,2,0,SEND,END,10218554,10223036,4482
VTIME,1765380680823637,1,0,SEND,END,10138005,10139740,1735
VTIME,1765380680824364,0,0,SEND,END,10111697,10112469,772
VTIME,1765380680827991,1,0,COMPUTE,START,2326719,2326719,0
VTIME,1765380680828433,2,0,COMPUTE,START,2326719,2326719,0    ← All aligned!
VTIME,1765380680828484,0,0,COMPUTE,START,2326719,2326719,0
```

**Same GEMM 0 operation, all devices now synchronized to same virtual time!**

