# PRD: Memory Constraints for DT-FM and Confident Schedulers

**Date:** March 5, 2026  
**Status:** Draft  
**Author:** Bessus (agent)

---

## 1. Problem Statement

### 1.1 The Issue

DT-FM and Confident schedulers produce pipeline partition plans that **ignore GPU memory entirely**. They assign layers to devices purely based on compute time balancing, with no check on whether the device can actually hold:

- Model weights for its assigned layers
- Gradient buffers
- Optimizer state (AdamW momentum + variance)
- In-flight activations during pipeline execution

This leads to **OOM crashes at runtime** when a device is assigned more layers than its GPU memory can support. The training job starts, progresses through initialization, then dies mid-forward pass when activation memory spikes.

### 1.2 Current State by Scheduler

| Scheduler | Memory Awareness | What Happens |
|-----------|-----------------|--------------|
| **Asteroid** | Full hard constraint in DP | Prunes infeasible partitions; reduces batch size per device to fit | 
| **Confident** | None — calls `DPPartitioner.partition()` | Plans may OOM at runtime |
| **DT-FM** | None — GCMA + `DPPartitioner.partition()` | Plans may OOM at runtime |

### 1.3 Why It Matters

In heterogeneous edge clusters with varying GPU memory (e.g., 24GB, 40GB, 48GB), a memory-blind scheduler can assign 60% of a 7B model's layers to a 24GB GPU that can only hold 30%. The plan looks optimal on paper (balanced compute), but is physically impossible to execute.

---

## 2. How Asteroid Handles Memory (Reference)

Asteroid's memory model is the gold standard in this codebase. Its formula lives in `asteroid_planner.py` (`_memory_footprint()`):

$$\text{Mem}_p(\beta) = \Big(\underbrace{W}_{\text{weights}} + \underbrace{W}_{\text{gradients}} + \underbrace{2W}_{\text{optimizer (AdamW)}} + \underbrace{K_p \cdot A(\beta)}_{\text{activations}}\Big) \times 1.10$$

Where:
- $W = \sum_{l=\text{start}}^{\text{end}} \text{weight\_sizes}[l]$ — parameter bytes for the stage's layers
- $K_p = \max(1,\; 2(P - p) - 1)$ — **1F1B pipeline bubble factor**: earlier stages must store more in-flight activations. Stage 0 in a 6-stage pipeline has $K_p = 11$; the last stage has $K_p = 1$
- $A(\beta) = \beta \cdot \sum_{l} \text{activation\_sizes}[l]$ — activation memory scales with micro-batch size
- The $\times 1.10$ is a **10% safety margin** for PyTorch overhead, CUDA context, and fragmentation

Asteroid enforces this as a **hard constraint** at three points in its 3D DP:
1. **Single-stage base case** — candidates exceeding `device.memory_budget_mb` are pruned
2. **Multi-stage DP transitions** — splits violating memory are skipped
3. **Micro-batch allocation** — batch size per device is reduced until it fits

---

## 3. Existing (Unused) Memory Infrastructure

`DPPartitioner` already has a `partition_with_memory()` method that is **never called** by either Confident or DT-FM. Its current memory formula:

$$\text{Mem}_{\text{current}}(\text{start}, \text{end}) = \sum_{l=\text{start}}^{\text{end}} \text{output\_sizes}[l]$$

This is **severely underestimating** actual memory by 10-20x because it only counts activation output sizes, missing:
- Model weights ($W$)
- Gradient buffers ($W$)  
- Optimizer state ($2W$ for AdamW)
- Pipeline bubble factor ($K_p$ scaling)
- Safety margin

---

## 4. Proposed Approaches

### 4.1 Approach A: Memory Filter (Bolt-on Constraint)

**Concept:** Keep the existing 2D DP structure. Fix the memory formula in `partition_with_memory()` and wire it into the Confident and DT-FM call paths.

**How it works:**
1. Fix `_segment_memory_mb()` to use the full memory formula: $4W + K_p \cdot A(\beta)$
2. In `partition_with_memory()`, the DP's inner loop adds a `continue` when a candidate split violates memory:

```python
for m in range(j - 1, i):
    if memory_footprint(m+1, i, stage_j) > memory_budgets[j]:
        continue   # ← prune this split
    cost = max(dp[m][j-1], compute_t + comm_t)
```

3. Wire `ConfidentScheduler.partition()` → `partition_with_memory(budgets)`
4. Wire `DTFMStrategy._dp_partition()` → add memory check in its inner loop

**Pros:**
- Minimal code change (fix formula + wire calls)
- Preserves existing DP structure and scheduler identity
- Easy to test — same test cases, just with memory budgets added

**Cons:**
- Cannot reduce batch size to fit memory (batch is fixed externally)
- If memory is tight, may find **no feasible partition** and must error out or fall back
- $K_p$ calculation requires knowing the total number of stages, which is fixed before the DP runs (OK for Confident/DT-FM since they iterate over PP values)

### 4.2 Approach B: Fundamental Formula Change (Embedded Constraint)

**Concept:** Rewrite the Confident/DT-FM DP to add a 3rd dimension (stage count), embed batch allocation inside the DP, and co-optimize memory with compute.

**How it works:**
1. Change DP state from `dp[i][j]` to `dp[l][n][p]` (layers × devices × stages)
2. Inside the inner loop, call a micro-batch allocator that distributes $\beta$ across devices in a group
3. Check memory footprint per device with the full formula
4. If a device can't fit, reduce its $\beta$ until it fits (or skip the candidate)

**Pros:**
- Full memory awareness with batch adaptation
- Can handle extreme heterogeneity (small GPU gets fewer micro-batches)
- Matches Asteroid's capability

**Cons:**
- **Effectively turns Confident/DT-FM into Asteroid** — defeats the purpose of separate schedulers
- Significantly higher complexity: $O(L^2 \cdot D)$ → $O(L^2 \cdot N^2 \cdot P)$
- Major rewrite of both schedulers
- Harder to validate against the original papers

### 4.3 Recommendation

**Approach A (Memory Filter)** is the right choice. The schedulers should remain distinct algorithms with distinct trade-offs:

| Property | Asteroid | Confident/DT-FM + Memory Filter |
|----------|----------|--------------------------------|
| DP dimensions | 3D ($l, n, p$) | 2D ($i, j$) + memory prune |
| Batch adaptation | Yes (per-device) | No (fixed, uniform) |
| Memory handling | Co-optimized | Hard filter (prune infeasible) |
| When it fails | Reduces batch to fit | Reports infeasible (user must reduce batch or add devices) |

The filter approach catches the critical failure mode (OOM) while keeping the schedulers faithful to their original designs.

---

## 5. DP Formula Comparison (All Three Schedulers)

### 5.1 Confident / DPPartitioner — 2D DP

**State:** `dp[i][j]` — minimum bottleneck time to assign layers `0..i` across stages `0..j`

**Recurrence:**
$$dp[i][j] = \min_{m \in [j-1,\, i)} \max\!\left(dp[m][j-1],\;\; \frac{T(j,\, m+1,\, i)}{c_j} + \frac{O_m}{B_{j-1}}\right)$$

| Symbol | Meaning |
|--------|---------|
| $T(j, m+1, i)$ | Sum of layer execution times on device $j$ from layer $m+1$ to $i$ |
| $c_j$ | Computing capacity of device $j$ (normalizes for heterogeneity) |
| $O_m$ | Output tensor size of boundary layer $m$ (bytes for cross-stage transfer) |
| $B_{j-1}$ | Bandwidth between stage $j-1$ and stage $j$ |

**What it optimizes:** Finds the partition that minimizes the slowest (bottleneck) pipeline stage, accounting for compute heterogeneity and inter-stage communication.

**What it ignores:** Memory, batch size, pipeline depth effects, allreduce.

---

### 5.2 DT-FM — 2D DP (with GCMA pre-stage)

DT-FM is a two-stage algorithm:

**Stage 1 — GCMA** (before DP): Assigns devices to pipeline groups using a genetic/crossover topology optimizer that minimizes communication cost across the network graph.

**Stage 2 — DP** (same structure as Confident):

**State:** `dp[end][j]` — minimum bottleneck time to assign layers `0..end` across stages `0..j`

**Recurrence:**
$$dp[\text{end}][j] = \min_{\text{cut}} \max\!\left(dp[\text{cut}][j-1],\;\; R(j,\, \text{cut}+1,\, \text{end}) + C_{\text{boundary}}\right)$$

| Symbol | Meaning |
|--------|---------|
| $R(j, \text{start}, \text{end})$ | Prefix-sum range cost: uses the **bottleneck (slowest) device** in group $j$ as representative |
| $C_{\text{boundary}}$ | Inter-stage communication time considering actual cross-group bandwidth |

**Key difference from Confident:** Device-to-group assignment is pre-computed by GCMA (not decided by the DP). The DP partitions layers across pre-formed groups, using the slowest device in each group as the compute cost representative.

**What it optimizes:** Same min-max bottleneck, but with GCMA-optimized device grouping.

**What it ignores:** Memory, batch size, pipeline depth effects.

---

### 5.3 Asteroid — 3D DP

**State:** `q[l][n][p]` — minimum latency using $l$ layers, $n$ devices, and $p$ pipeline stages

**Recurrence:**
$$q[l][n][p] = \min_{\substack{l' \in [p-1,\, l) \\ n' \in [p-1,\, n)}} \max\!\left(q[l'][n'][p-1],\;\; M \cdot \text{exec}(l,l',n,n') + C_{\text{inter}}\right) + T_{\text{allreduce}}$$

Subject to the **hard memory constraint** for every device $d$ in the new group:

$$\underbrace{4W_{d}}_{\text{weights+grad+opt}} + \underbrace{K_p \cdot A(\beta_d)}_{\text{activations}} \leq \text{MemBudget}_d \quad \forall d \in \text{group}$$

| Symbol | Meaning |
|--------|---------|
| $l, l'$ | Layer range boundaries |
| $n, n'$ | Device range boundaries |
| $p$ | Number of pipeline stages (3rd DP dimension) |
| $M$ | Number of micro-batches |
| $\text{exec}()$ | Stage execution time, computed by `_alloc_microbatch()` which distributes $\beta$ per device based on compute speed |
| $C_{\text{inter}}$ | Inter-stage communication between adjacent groups |
| $T_{\text{allreduce}}$ | Gradient synchronization time within a data-parallel group |
| $K_p = \max(1, 2(P-p)-1)$ | 1F1B activation multiplier — earlier stages store more in-flight micro-batches |
| $\beta_d$ | Micro-batch size assigned to device $d$ (varies per device within a group) |
| $W_d$ | Weight bytes for layers assigned to device $d$'s stage |
| $A(\beta_d)$ | Activation memory = $\beta_d \times \sum \text{activation\_sizes}$ |

**What it co-optimizes:** Layer partitioning, device grouping, stage count, per-device batch allocation, and memory — all simultaneously in a single DP.

---

### 5.4 Side-by-Side Summary

| Property | **Confident** | **DT-FM** | **Asteroid** |
|----------|--------------|-----------|-------------|
| **DP dimensions** | `dp[i][j]` — 2D | `dp[i][j]` — 2D | `q[l][n][p]` — 3D |
| **Objective** | $\min \max(\text{stage time})$ | $\min \max(\text{stage time})$ | $\min \max(\text{stage time})$ |
| **Layer → stage** | DP decides | DP decides | DP decides |
| **Device → group** | 1 device per stage (implicit) | GCMA decides (before DP) | DP co-decides (part of state space) |
| **Batch allocation** | Fixed externally | Fixed externally | Per-device, inside DP |
| **Memory constraint** | **None** | **None** | Hard: $4W + K_p A(\beta) \leq B$ |
| **Communication** | $O_m / B_{j-1}$ | Cross-group bandwidth | Inter-stage + allreduce |
| **Fault tolerance** | Re-partition on failure | None | None |
| **Complexity** | $O(L^2 \cdot D)$ | $O(L^2 \cdot S)$ + GCMA | $O(L^2 \cdot N^2 \cdot P)$ |

Where $L$ = layers, $D$ = devices, $S$ = stages (from GCMA), $N$ = total devices, $P$ = max stage count.

---

## 6. Implementation Plan (Approach A)

### 6.1 Changes Required

1. **Fix `DPPartitioner._segment_memory_mb()`** — Replace `sum(output_sizes)` with the full formula: $4W + K_p \cdot A(\beta)$
   - Add `weight_sizes` and `activation_sizes` as constructor parameters
   - Add `num_stages` and `micro_batch_size` parameters for $K_p$ and $A(\beta)$ calculation

2. **Wire `partition_with_memory()` into Confident** — Change `ConfidentScheduler.partition()` to call `partition_with_memory(budgets)` instead of `partition()`

3. **Wire memory check into DT-FM** — Add memory pruning in `DTFMStrategy._dp_partition()` inner loop

4. **Pass real memory budgets** — Flow `memory_budget_mb` from device profiles through to the DP

5. **Remove silent fallback** — Replace the "falling back to plain DP" behavior with a hard error when no feasible memory-aware partition exists

### 6.2 Files to Modify

| File | Change |
|------|--------|
| `baselines/schedulers/dp_partitioner.py` | Fix `_segment_memory_mb()`, add weight/activation parameters |
| `baselines/schedulers/confident_scheduler.py` | Call `partition_with_memory()` |
| `baselines/strategies/dtfm_strategy.py` | Add memory check in `_dp_partition()` |
| `baselines/strategies/confident_strategy.py` | Pass memory budgets from topology |
| `baselines/core/config.py` | Ensure `DeviceConfig.memory_budget_mb` flows through |
| `baselines/scripts/run_asteroid_planner.py` | Pass real memory values (currently hardcoded 40960) |

### 6.3 Testing

- Unit test: verify `_segment_memory_mb()` output matches Asteroid's `_memory_footprint()` for the same inputs
- Integration test: run Confident and DT-FM planners with tight memory budgets, verify they produce different (feasible) partitions vs. the no-memory path
- Regression test: with large memory budgets, the memory-aware path should produce the same partitions as the current no-memory path
