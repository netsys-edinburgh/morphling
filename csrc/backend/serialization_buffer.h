#pragma once

#include <sys/mman.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

class AlignedBufferPool {
 public:
  static constexpr size_t PAGE_SIZE = 4096;
  static constexpr size_t MAX_BUFFERS_PER_BUCKET = 32;

  // Pluggable pin/unpin callbacks. Return 0 on success.
  using PinFn = int (*)(void* ptr, size_t size);
  using UnpinFn = void (*)(void* ptr, size_t size);

  static AlignedBufferPool& instance() {
    static AlignedBufferPool pool;
    return pool;
  }

  // Override the default mlock/munlock pinning strategy.
  // Must be called before any Acquire() (e.g. at process startup).
  void SetPinFunctions(PinFn pin, UnpinFn unpin) {
    pin_fn_ = pin;
    unpin_fn_ = unpin;
  }

  // Acquire a buffer of at least `size` bytes (page-aligned, pinned)
  // Returns {pointer, actual_bucket_size}
  std::pair<uint8_t*, size_t> Acquire(size_t size) {
    size_t bucket = BucketSize(size);
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket];
    if (!free_list.empty()) {
      uint8_t* ptr = free_list.back();
      free_list.pop_back();
      return {ptr, bucket};
    }
    // Allocate new buffer
    return {AllocateNewBuffer(bucket), bucket};
  }

  // Release a buffer back to the pool
  void Release(uint8_t* ptr, size_t bucket_size) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket_size];
    if (free_list.size() < MAX_BUFFERS_PER_BUCKET) {
      free_list.push_back(ptr);
    } else {
      // Pool is full, free the buffer
      unpin_fn_(ptr, bucket_size);
      free(ptr);
    }
  }

  ~AlignedBufferPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [bucket_size, free_list] : free_lists_) {
      for (auto* ptr : free_list) {
        unpin_fn_(ptr, bucket_size);
        free(ptr);
      }
    }
    free_lists_.clear();
  }

  // Public for testing. Production code should use instance().
  AlignedBufferPool() = default;

 private:
  AlignedBufferPool(const AlignedBufferPool&) = delete;
  AlignedBufferPool& operator=(const AlignedBufferPool&) = delete;

  static int DefaultPin(void* ptr, size_t size) { return mlock(ptr, size); }

  static void DefaultUnpin(void* ptr, size_t size) { munlock(ptr, size); }

  // Round up to the next power-of-two bucket size (min 4KB)
  static size_t BucketSize(size_t size) {
    if (size <= PAGE_SIZE) return PAGE_SIZE;
    // Round up to next power of 2
    size_t bucket = PAGE_SIZE;
    while (bucket < size) bucket <<= 1;
    return bucket;
  }

  uint8_t* AllocateNewBuffer(size_t size) {
    uint8_t* ptr = nullptr;
    int ret = posix_memalign(reinterpret_cast<void**>(&ptr), PAGE_SIZE, size);
    if (ret != 0 || !ptr) {
      throw std::runtime_error("AlignedBufferPool: posix_memalign failed");
    }
    ret = pin_fn_(ptr, size);
    if (ret != 0) {
      // Pin failure is non-fatal, continue without pinning
      throw std::runtime_error("AlignedBufferPool: pin_fn_ failed");
    }
    return ptr;
  }

  PinFn pin_fn_ = DefaultPin;
  UnpinFn unpin_fn_ = DefaultUnpin;
  std::mutex mutex_;
  std::unordered_map<size_t, std::deque<uint8_t*>> free_lists_;
};

std::string BinaryToHex(const unsigned char* data, size_t length);

// Forward declarations
class SerializationBuffer;
typedef std::shared_ptr<SerializationBuffer> SerializationBufferPtr;

enum class SerializationFormat {
  PROTOBUF  // Protobuf-based format (default)
};

// Base interface for serializable messages
class ISerializable {
 public:
  virtual ~ISerializable() = default;

  // Serialize to binary format
  virtual SerializationBufferPtr Serialize(
      SerializationFormat format = SerializationFormat::PROTOBUF) const = 0;

  // Deserialize from binary format
  virtual void Deserialize(
      const void* data, size_t size,
      SerializationFormat format = SerializationFormat::PROTOBUF) = 0;

  // Get the message type
  virtual int32_t GetMessageType() const = 0;

  // Get debug string representation
  virtual std::string DebugString() const = 0;
};

// Buffer handler for reading/writing binary data
class SerializationBuffer {
 public:
  SerializationBuffer();
  explicit SerializationBuffer(const void* data, size_t size,
                               bool take_ownership = false);
  ~SerializationBuffer();

  // Non-copyable, movable
  SerializationBuffer(const SerializationBuffer&) = delete;
  SerializationBuffer& operator=(const SerializationBuffer&) = delete;
  SerializationBuffer(SerializationBuffer&& other) noexcept;
  SerializationBuffer& operator=(SerializationBuffer&& other) noexcept;

  // Allocate new buffer (uses AlignedBufferPool)
  void Allocate(size_t size);
  // Allocate using a specific pool (for testing)
  void Allocate(size_t size, AlignedBufferPool& pool);

  // Write methods
  void WriteUInt32(uint32_t value, bool network_order = false);
  void WriteUInt64(uint64_t value);
  void WriteInt64(int64_t value);
  void WriteBytes(const void* data, size_t size);

  // Read methods
  uint32_t ReadUInt32(bool network_order = false);
  uint64_t ReadUInt64();
  int64_t ReadInt64();
  void ReadBytes(void* dest, size_t size);
  const void* GetCurrentPtr() const;

  // Position management
  void SeekTo(size_t offset);
  size_t GetOffset() const { return offset_; }
  void* GetBuffer() const { return buffer_; }
  size_t GetSize() const { return size_; }

  // Validation
  bool CanRead(size_t bytes) const;
  void ValidateSize(size_t min_size) const;

  std::string HexString(size_t length) const;

 private:
  void FreeBuffer();

  uint8_t* buffer_;
  size_t size_;
  size_t offset_;
  bool owns_buffer_;
  size_t
      pool_bucket_size_;  // 0 = not from pool, >0 = bucket size for pool return
  AlignedBufferPool* pool_ = nullptr;  // nullptr = use singleton
};
