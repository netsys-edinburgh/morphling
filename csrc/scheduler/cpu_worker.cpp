#include "cpu_worker.h"

#include <algorithm>
#include <unordered_set>

#include "utils/logger.h"

// ---------------------------------------------------------------------------
// CpuWorker
// ---------------------------------------------------------------------------

CpuWorker::CpuWorker(std::vector<int> assigned_cores, int partition_idx)
    : assigned_cores_(std::move(assigned_cores)),
      partition_idx_(partition_idx) {
  worker_ = std::thread([this] { Run(); });
  LOG_DEBUG << "CpuWorker created: partition=" << partition_idx_
            << " cores=" << assigned_cores_.size();
}

CpuWorker::~CpuWorker() {
  active_slot_ = nullptr;
  affinity_slots_.clear();
  LOG_DEBUG << "CpuWorker destroyed: partition=" << partition_idx_;
}

void CpuWorker::InitAllAffinitySlots() {
  int total_cores = static_cast<int>(assigned_cores_.size());
  LOG_FATAL_IF(total_cores == 0)
      << "CpuWorker partition " << partition_idx_
      << " has no assigned cores";

  LOG_INFO << "CpuWorker partition " << partition_idx_ << ": "
           << total_cores << " assigned cores [";

  // Create a slot for each valid core count: 1, 2, ..., N
  for (int n = 1; n <= total_cores; n++) {
    auto slot = CreateAffinitySlot(n);
    affinity_slots_.emplace(n, std::move(slot));
    LOG_INFO << "  Created affinity slot: " << n << " cores";
  }

  // Default: use all assigned cores
  active_slot_ = &affinity_slots_.at(total_cores);
  active_slot_->Apply();

  LOG_INFO << "CpuWorker initialized: " << affinity_slots_.size()
           << " affinity slots, active=" << total_cores << " cores";
}

CpuAffinitySlot CpuWorker::CreateAffinitySlot(int num_cores) {
  // Take the first num_cores from assigned_cores_
  std::vector<int> cores(assigned_cores_.begin(),
                         assigned_cores_.begin() + num_cores);
  return CpuAffinitySlot(num_cores, std::move(cores));
}

bool CpuWorker::SwitchAffinity(int num_cores) {
  auto it = affinity_slots_.find(num_cores);
  if (it == affinity_slots_.end()) return false;
  active_slot_ = &it->second;
  active_slot_->Apply();
  return true;
}

bool CpuWorker::SwitchAffinity(const std::vector<int>& cores) {
  // Validate all requested cores are in the assigned set
  std::unordered_set<int> assigned_set(assigned_cores_.begin(),
                                       assigned_cores_.end());
  for (int core : cores) {
    if (assigned_set.find(core) == assigned_set.end()) {
      LOG_WARN << "Core " << core
               << " is not in assigned set for partition "
               << partition_idx_;
      return false;
    }
  }

  // Build a transient cpu_set_t and apply directly
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int core : cores) {
    CPU_SET(core, &cpuset);
  }
  morphling::PinThreadToCores(cpuset);

  // Point active_slot_ to the preset slot if one matches, otherwise
  // leave it at the last preset (callers should use GetActiveCores()
  // to query actual state after arbitrary switch).
  auto it = affinity_slots_.find(static_cast<int>(cores.size()));
  if (it != affinity_slots_.end() && it->second.core_ids == cores) {
    active_slot_ = &it->second;
  }
  return true;
}

int CpuWorker::GetActiveCoreCount() const {
  return active_slot_ ? active_slot_->core_count : 0;
}

std::vector<int> CpuWorker::GetActiveCores() const {
  return active_slot_ ? active_slot_->core_ids : std::vector<int>{};
}

std::vector<int> CpuWorker::GetAvailableCoreCounts() const {
  std::vector<int> counts;
  counts.reserve(affinity_slots_.size());
  for (const auto& [count, _] : affinity_slots_) {
    counts.push_back(count);
  }
  std::sort(counts.begin(), counts.end());
  return counts;
}

void CpuWorker::Run() {
  InitAllAffinitySlots();

  LOG_INFO << "CpuWorker ready: partition=" << partition_idx_;

  // Enter the WorkerBase task loop
  WorkerBase::Run();
}

// ---------------------------------------------------------------------------
// CpuWorkerPool
// ---------------------------------------------------------------------------

CpuWorkerPool::CpuWorkerPool(int num_workers,
                              std::vector<int> assignable_cores,
                              SchedulingPolicyType policy) {
  int total_cores = static_cast<int>(assignable_cores.size());
  int cores_per_worker = total_cores / num_workers;
  LOG_FATAL_IF(cores_per_worker == 0)
      << "Not enough cores (" << total_cores << ") for "
      << num_workers << " workers";

  for (int w = 0; w < num_workers; w++) {
    int start = w * cores_per_worker;
    int end = (w == num_workers - 1) ? total_cores
                                     : start + cores_per_worker;
    std::vector<int> partition(assignable_cores.begin() + start,
                               assignable_cores.begin() + end);
    workers_.emplace_back(
        std::make_shared<CpuWorker>(std::move(partition), w));
  }

  switch (policy) {
    case SchedulingPolicyType::kRoundRobinCpu:
      scheduler_ =
          std::make_unique<RoundRobinCpuPolicy>(num_workers);
      break;
    default:
      LOG_FATAL << "Unsupported scheduling policy for CpuWorkerPool: "
                << SchedulingPolicyTypeToString(policy);
  }

  LOG_INFO << "CpuWorkerPool created: " << num_workers
           << " workers over " << total_cores
           << " cores, policy=" << SchedulingPolicyTypeToString(policy);
}

CpuWorkerPool::~CpuWorkerPool() {
  for (auto& worker : workers_) {
    worker->Stop();
  }
}

void CpuWorkerPool::EnqueueTask(const std::string& task_id,
                                 WorkerBase::Task task) {
  auto [worker_idx, priority] = scheduler_->Schedule(nullptr);
  workers_[worker_idx]->AddTask(task_id, std::move(task));
}

void CpuWorkerPool::WaitAll() {
  for (auto& worker : workers_) {
    worker->WaitTaskDone();
  }
}

void CpuWorkerPool::Wait(const std::string& task_id) {
  for (auto& worker : workers_) {
    worker->WaitTaskDone(task_id);
  }
}
