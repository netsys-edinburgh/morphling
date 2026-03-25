#include "scatter_gather_buffer.h"

ScatterGatherBuffer::~ScatterGatherBuffer() {
  for (auto& [ptr, bucket] : owned_pool_entries_) {
    pool_->Release(ptr, bucket);
  }
}

ScatterGatherBuffer::ScatterGatherBuffer(ScatterGatherBuffer&& other) noexcept
    : segments_(std::move(other.segments_)),
      owned_pool_entries_(std::move(other.owned_pool_entries_)),
      pool_(other.pool_) {
  other.pool_ = &AlignedBufferPool::instance();
}

ScatterGatherBuffer& ScatterGatherBuffer::operator=(
    ScatterGatherBuffer&& other) noexcept {
  if (this != &other) {
    // Free current owned entries
    for (auto& [ptr, bucket] : owned_pool_entries_) {
      pool_->Release(ptr, bucket);
    }
    segments_ = std::move(other.segments_);
    owned_pool_entries_ = std::move(other.owned_pool_entries_);
    pool_ = other.pool_;
    other.pool_ = &AlignedBufferPool::instance();
  }
  return *this;
}

void ScatterGatherBuffer::AddOwnedSegment(uint8_t* data, size_t size,
                                          size_t pool_bucket) {
  segments_.emplace_back(data, size, true);
  owned_pool_entries_.emplace_back(data, pool_bucket);
}

void ScatterGatherBuffer::AddReferenceSegment(const void* data, size_t size) {
  if (data && size > 0) {
    segments_.emplace_back(data, size, false);
  }
}

size_t ScatterGatherBuffer::GetTotalSize() const {
  size_t total = 0;
  for (const auto& seg : segments_) {
    total += seg.size;
  }
  return total;
}
