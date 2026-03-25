#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "serialization_buffer.h"

struct BufferSegment {
  const void* data;
  size_t size;
  bool owned;  // if true, data is freed on destruction

  BufferSegment(const void* d, size_t s, bool o) : data(d), size(s), owned(o) {}
};

class ScatterGatherBuffer {
 public:
  ScatterGatherBuffer() : pool_(&AlignedBufferPool::instance()) {}
  explicit ScatterGatherBuffer(AlignedBufferPool& pool) : pool_(&pool) {}
  ~ScatterGatherBuffer();

  ScatterGatherBuffer(ScatterGatherBuffer&& other) noexcept;
  ScatterGatherBuffer& operator=(ScatterGatherBuffer&& other) noexcept;
  ScatterGatherBuffer(const ScatterGatherBuffer&) = delete;
  ScatterGatherBuffer& operator=(const ScatterGatherBuffer&) = delete;

  // Add an owned segment (will be freed on destruction via pool)
  void AddOwnedSegment(uint8_t* data, size_t size, size_t pool_bucket);

  // Add a reference segment (not owned, caller must ensure lifetime)
  void AddReferenceSegment(const void* data, size_t size);

  size_t GetTotalSize() const;
  const std::vector<BufferSegment>& GetSegments() const { return segments_; }

 private:
  std::vector<BufferSegment> segments_;
  // Track pool bucket sizes for owned segments to return them
  std::vector<std::pair<uint8_t*, size_t>> owned_pool_entries_;
  AlignedBufferPool* pool_;
};

typedef std::shared_ptr<ScatterGatherBuffer> ScatterGatherBufferPtr;
