# VTIME Results Inventory

**Generated**: 2026-04-18
**Purpose**: Understand what data exists in each VTIME result directory for planning the full 64-device sweep.

## Overview

Three directories contain VTIME experiment results:

| Directory | Purpose | Device Count | Status |
|-----------|---------|--------------|--------|
| `results/vtime/` | Initial 4-device baseline | Fixed at 4 | ✓ Complete + Processed |
| `results/vtime_models/` | Multi-model exploration | 2-5 of 64 (partial) | Partial + Processed |
| `results/vtime_sweep/` | Full 64-device sweep | In-progress | In-progress, no processing |

---

## Directory 1: `results/vtime/`

**Location**: `/home/xly/DeviceEmulator/results/vtime/`

### Purpose
Initial baseline experiments with fixed 4-device fleet. All baselines complete and processed.

### Structure
```
vtime/
├── alpa/device_logs/              → 4 non-empty dev_*.log files
├── asteroid/device_logs/          → 5 non-empty dev_*.log files
├── cleave/device_logs/            → 5 non-empty dev_*.log files
├── confident/device_logs/         → 12 non-empty dev_*.log files
├── dtfm/device_logs/              → 4 non-empty dev_*.log files
├── summary/                        → Processed JSON results
│   ├── alpa_runtime.json          (18 KB)
│   ├── asteroid_runtime.json      (19 KB)
│   ├── cleave_runtime.json        (17 KB)
│   ├── confident_runtime.json     (11 KB)
│   ├── dtfm_runtime.json          (18 KB)
│   ├── paper_summary.json         (2.3 KB)
│   └── sweep_final.json           (3.7 KB)
├── cleave_test/device_logs/       → Empty (test run)
├── test/                          → Partial test
├── fleet_4dev.json                → Device config (4 devices)
└── summary.json                   → Run metadata
```

### Data Characteristics

**Raw Data**:
- Device logs: plain text STDOUT from each device process
- Filenames: `dev_0.log`, `dev_1.log`, etc.
- Format: Semi-structured text with timing info

**Processed Data** (in `summary/`):
```json
{
  "baseline_name": "alpa",
  "total_runtime_ms": 12345.67,
  "compute_time_ms": 10234.45,
  // ... additional metrics
}
```

### Baselines Complete
- ✓ Cleave
- ✓ Asteroid
- ✓ DTFM
- ✓ AlPa
- ✓ Confident

### Status
**FULLY COMPLETE** - All 5 baselines have raw logs + processed summaries at 4-device scale.

---

## Directory 2: `results/vtime_models/`

**Location**: `/home/xly/DeviceEmulator/results/vtime_models/`

### Purpose
Multi-model exploration phase. Tests different models with partial 64-device runs, all with processed results.

### Models Present

#### llama2-7b
```
llama2-7b/
├── alpa/device_logs/              → 3/64 devices
├── asteroid/device_logs/          → 3/64 devices
├── cleave/device_logs/            → 3/64 devices
├── confident/device_logs/         → 3/64 devices
├── dtfm/device_logs/              → 3/64 devices
├── alpa_results/
│   ├── summary.json
│   ├── generated_device_fleet.json
│   ├── manifests/alpa_manifest.json
│   └── runtimes/alpa_runtime.json
├── asteroid_results/              → Same structure
├── cleave_results/                → Same structure
├── confident_results/             → Same structure
└── dtfm_results/                  → Same structure
```

#### opt-1.3b
```
opt-1.3b/
├── alpa/device_logs/              → 2/64 devices
├── asteroid/device_logs/          → 2/64 devices
├── cleave/device_logs/            → 3/65 devices (anomaly)
├── confident/device_logs/         → 2/64 devices
├── dtfm/device_logs/              → 2/64 devices
└── [baseline]_results/            → Processed for each baseline
```

#### opt-13b
```
opt-13b/
├── alpa/device_logs/              → 5/64 devices
├── asteroid/device_logs/          → 5/64 devices
├── cleave/device_logs/            → 5/64 devices
├── confident/device_logs/         → 5/64 devices
├── dtfm/device_logs/              → 5/64 devices
└── [baseline]_results/            → Processed for each baseline
```

### Data Structure

**Raw Device Logs**:
```
<model>/<baseline>/device_logs/dev_0.log
<model>/<baseline>/device_logs/dev_1.log
...
```

**Processed Results** (each baseline):
```
<model>/<baseline>_results/
├── summary.json                   → Run summary
├── generated_device_fleet.json    → Fleet configuration
├── manifests/<baseline>_manifest.json   → Device placement info
└── runtimes/<baseline>_runtime.json     → Performance metrics
```

### Root-Level Files
- `vtime_summary.json` - Aggregated summary across all models and baselines

### Key Observations

1. **All baselines processed**: Every model has `*_results/` directories for all 5 baselines
2. **Partial device runs**: Devices range from 2-5 out of 64 target
3. **Manifests & runtimes available**: Device placement and performance data structured and ready
4. **Appears to be exploratory**: Limited device count suggests this was a validation phase

---

## Directory 3: `results/vtime_sweep/`

**Location**: `/home/xly/DeviceEmulator/results/vtime_sweep/`

### Purpose
Full 64-device scaling sweep (active/in-progress). Currently contains raw logs only; no processed results yet.

### Models and Coverage

#### llama2-7b (4/5 baselines)
```
llama2-7b/
├── alpa/device_logs/              → 4/64 devices  (  6%)
├── asteroid/device_logs/          → 4/64 devices  (  6%)
├── confident/device_logs/         → 4/64 devices  (  6%)
├── cleave/device_logs/            → EMPTY/MISSING
└── dtfm/device_logs/              → EMPTY/MISSING
```

#### opt-1.3b (5/5 baselines)
```
opt-1.3b/
├── alpa/device_logs/              → 4/64 devices  (  6%)
├── asteroid/device_logs/          → 4/64 devices  (  6%)
├── cleave/device_logs/            → 4/64 devices  (  6%)
├── confident/device_logs/         → 2/64 devices  (  3%) ← INCOMPLETE
└── dtfm/device_logs/              → 4/64 devices  (  6%)
```

#### opt-125m (5/5 baselines)
```
opt-125m/
├── alpa/device_logs/              → 4/64 devices  (  6%)
├── asteroid/device_logs/          → 5/64 devices  (  7%)
├── cleave/device_logs/            → 5/64 devices  (  7%)
├── confident/device_logs/         → 12/64 devices (18%) ← BEST PROGRESS
└── dtfm/device_logs/              → 4/64 devices  (  6%)
```

#### opt-13b (3/5 baselines)
```
opt-13b/
├── asteroid/device_logs/          → 4/64 devices  (  6%)
├── cleave/device_logs/            → 4/64 devices  (  6%)
├── dtfm/device_logs/              → 4/64 devices  (  6%)
├── alpa/device_logs/              → MISSING
└── confident/device_logs/         → MISSING
```

#### opt-30b (1/5 baselines)
```
opt-30b/
├── cleave/device_logs/            → 4/64 devices  (  6%)
├── asteroid/device_logs/          → MISSING
├── dtfm/device_logs/              → MISSING
├── alpa/device_logs/              → MISSING
└── confident/device_logs/         → MISSING
```

#### opt-66b (1/5 baselines)
```
opt-66b/
├── cleave/device_logs/            → 4/64 devices  (  6%)
├── asteroid/device_logs/          → MISSING
├── dtfm/device_logs/              → MISSING
├── alpa/device_logs/              → MISSING
└── confident/device_logs/         → MISSING
```

### Data Characteristics

**Raw Data Only**:
- Contains `<model>/<baseline>/device_logs/dev_0.log` through `dev_N.log` files
- No processed JSON results yet (unlike vtime_models/)
- Typical pattern: 4 devices per run, a few reaching 5, opt-125m/confident at 12

**No Processed Results**:
- Missing: manifests, runtimes, summary.json for each baseline
- Processing should occur after reaching 64/64 device runs

### Current Status

| Model | Status | Notes |
|-------|--------|-------|
| **opt-125m** | IN PROGRESS | Closest to completion (confident 18%) |
| **opt-1.3b** | IN PROGRESS | All baselines present, mostly at 4/64 |
| **llama2-7b** | IN PROGRESS | Missing 2 baselines (cleave, dtfm) |
| **opt-13b** | IN PROGRESS | Missing 2 baselines (alpa, confident) |
| **opt-30b** | SPARSE | Only cleave baseline |
| **opt-66b** | SPARSE | Only cleave baseline |

**None at 64/64 completion yet.**

---

## Summary Comparison Table

| Directory | Models | Baselines | Device Count | Raw Logs | Processed | Status |
|-----------|--------|-----------|--------------|----------|-----------|--------|
| **vtime** | 1 (implicit) | 5/5 | 4 (fixed) | ✓ | ✓ | Complete |
| **vtime_models** | 3 | 5/5 per model | 2-5 (partial) | ✓ | ✓ | Partial+Processed |
| **vtime_sweep** | 6 | 1-5 per model | Targeting 64 | ✓ | ✗ | In-Progress |

---

## Key Gaps for Full 64-Device Sweep

### Completely Missing Baselines
- **opt-30b**: Missing asteroid, dtfm, alpa, confident (4 baselines)
- **opt-66b**: Missing asteroid, dtfm, alpa, confident (4 baselines)
- **opt-13b**: Missing alpa, confident (2 baselines)
- **llama2-7b**: Missing cleave, dtfm (2 baselines)

### Incomplete (Below 64 Devices)
All models need scaling:
- opt-125m/confident: 12/64 (closest)
- Everything else: 2-5/64

### No Processed Results
`vtime_sweep/` contains only raw logs. Need to:
1. Complete all 64-device runs
2. Generate manifests from device_logs
3. Extract runtimes and performance metrics
4. Create summary.json for each baseline

---

## Recommendations

### Priority 1: Complete Current Models
- Continue and complete opt-125m (already at 12/64 for confident)
- Scale opt-1.3b and llama2-7b to 64 devices per baseline

### Priority 2: Fill Baseline Gaps
- Add missing baselines to opt-13b (alpa, confident)
- Add missing baselines to llama2-7b (cleave, dtfm)

### Priority 3: Expand to Larger Models
- Start opt-30b and opt-66b with all 5 baselines (not just cleave)

### Priority 4: Process Results
- Once each baseline reaches 64/64, generate:
  - Device manifests
  - Runtime/performance JSON
  - Summary statistics

---

## Data Access Paths

For downstream analysis:

```bash
# vtime: Complete 4-device baseline
cat results/vtime/summary/*.json

# vtime_models: Processed results for each model
cat results/vtime_models/<MODEL>/<BASELINE>_results/runtimes/<BASELINE>_runtime.json

# vtime_sweep: Raw logs (for custom processing)
cat results/vtime_sweep/<MODEL>/<BASELINE>/device_logs/dev_*.log
```
