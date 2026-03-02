# Confidant Profiling & Scheduling: Parameters Overview

This document outlines the low-level hardware and model performance metrics captured by the `ConfidantProfiler` (Layer 5: Profiling & Scheduling). These metrics are aggregated and passed to the `ConfidantScheduler` to determine the optimal model partition points across heterogeneous devices using Dynamic Programming (DP).

---

## 1. Computational Metrics (Per Layer)

The profiler measures the execution time of each individual layer within the model. These are the most critical inputs for the DP scheduling algorithm to evaluate compute costs.

* **`forward_ms`**: The wall-clock time taken for a forward pass through a single layer, measured using CUDA events for precision.
* **`backward_ms`**: The wall-clock time taken for the backward pass (gradient calculation) of a single layer.
* **`time_intervals`**: An aggregated cost calculated as . The scheduler uses this to evaluate the "compute cost" of assigning a specific range of layers to a single GPU.

## 2. Communication & Bandwidth

To account for the cost of moving data between pipeline stages (GPUs over a network), the profiler identifies the following networking metrics:

* **`bandwidths`**: The Peer-to-Peer (P2P) transfer speed between GPUs, measured in **MB/s**. This is determined by timing the transfer of a large dummy tensor between devices.
* **`output_sizes`**: The size of the activations (tensors) produced by each layer. In Transformer models (e.g., GPT-2/BERT), this is typically linked to the `hidden_dim`.
* **`comm_time`**: Calculated by the scheduler as . This represents the latency penalty incurred when a partition point is placed between two layers.

## 3. Hardware Constraints & Capacity

The scheduler uses these parameters to ensure the partition is feasible and to balance the load efficiently across heterogeneous hardware setups.

* **`available_memory` (GB)**: The profiler queries `torch.cuda.get_memory_info` to find the remaining free VRAM on each device. This prevents the scheduler from assigning too many layers to a single GPU and causing an *Out of Memory (OOM)* error.
* **`computing_capacities`**: A ratio (defaulting to `1.0`) representing the relative compute strength of each GPU. The scheduler divides the `compute_time` by this capacity to normalize the cost across different types of GPUs.
* **`peak_mem_mb`**: The maximum memory allocated during a layer's forward and backward pass, used strictly to validate memory constraints during partitioning.

---

## Summary of Scheduler Inputs

| Parameter | Source Method | Description |
| --- | --- | --- |
| **Layer Compute Time** | `profile_layer()` | Combined  latency for DP cost calculation. |
| **P2P Bandwidth** | `profile_bandwidth()` | Speed of data transfer between GPU pairs. |
| **Activation Size** | `output_sizes` | The "weight" of the data being sent over the wire. |
| **Memory Limit** | `get_memory_info()` | Maximum capacity threshold for a partition segment. |
| **Device Capacity** | `computing_capacities` | Scaling factor for balancing load on mixed GPU setups. |

---

### Next Steps & DP Algorithm

The parameters outlined above are fed directly into a Dynamic Programming (DP) algorithm. The DP algorithm calculates the minimum bottleneck time to assign layers across devices, minimizing pipeline bubbles and balancing the load according to the `computing_capacities` and `bandwidths` available.