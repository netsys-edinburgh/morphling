#include <arpa/inet.h>

#include <algorithm>

#include "server_base.h"

SerializationBuffer::SerializationBuffer()
    : buffer_(nullptr),
      size_(0),
      offset_(0),
      owns_buffer_(false),
      pool_bucket_size_(0) {}

SerializationBuffer::SerializationBuffer(const void* data, size_t size,
                                         bool take_ownership)
    : buffer_(const_cast<uint8_t*>(static_cast<const uint8_t*>(data))),
      size_(size),
      offset_(0),
      owns_buffer_(take_ownership),
      pool_bucket_size_(0) {}

SerializationBuffer::SerializationBuffer(SerializationBuffer&& other) noexcept
    : buffer_(other.buffer_),
      size_(other.size_),
      offset_(other.offset_),
      owns_buffer_(other.owns_buffer_),
      pool_bucket_size_(other.pool_bucket_size_),
      pool_(other.pool_) {
  other.buffer_ = nullptr;
  other.size_ = 0;
  other.offset_ = 0;
  other.owns_buffer_ = false;
  other.pool_bucket_size_ = 0;
  other.pool_ = nullptr;
}

SerializationBuffer& SerializationBuffer::operator=(
    SerializationBuffer&& other) noexcept {
  if (this != &other) {
    FreeBuffer();
    buffer_ = other.buffer_;
    size_ = other.size_;
    offset_ = other.offset_;
    owns_buffer_ = other.owns_buffer_;
    pool_bucket_size_ = other.pool_bucket_size_;
    pool_ = other.pool_;
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.offset_ = 0;
    other.owns_buffer_ = false;
    other.pool_bucket_size_ = 0;
    other.pool_ = nullptr;
  }
  return *this;
}

SerializationBuffer::~SerializationBuffer() { FreeBuffer(); }

void SerializationBuffer::FreeBuffer() {
  if (owns_buffer_ && buffer_) {
    if (pool_bucket_size_ > 0) {
      // Return to injected pool or singleton
      if (pool_) {
        pool_->Release(buffer_, pool_bucket_size_);
      } else {
        AlignedBufferPool::instance().Release(buffer_, pool_bucket_size_);
      }
    } else {
      free(buffer_);
    }
    buffer_ = nullptr;
    owns_buffer_ = false;
    pool_bucket_size_ = 0;
  }
}

void SerializationBuffer::Allocate(size_t size) {
  FreeBuffer();

  // Acquire from pool (page-aligned, mlocked)
  auto [ptr, bucket] = AlignedBufferPool::instance().Acquire(size);
  buffer_ = ptr;
  size_ = size;
  offset_ = 0;
  owns_buffer_ = true;
  pool_bucket_size_ = bucket;
}

void SerializationBuffer::Allocate(size_t size, AlignedBufferPool& pool) {
  FreeBuffer();

  auto [ptr, bucket] = pool.Acquire(size);
  buffer_ = ptr;
  size_ = size;
  offset_ = 0;
  owns_buffer_ = true;
  pool_bucket_size_ = bucket;
  pool_ = &pool;
}

void SerializationBuffer::WriteUInt32(uint32_t value, bool network_order) {
  if (network_order) {
    value = htonl(value);
  }
  memcpy(buffer_ + offset_, &value, sizeof(uint32_t));
  offset_ += sizeof(uint32_t);
}

void SerializationBuffer::WriteUInt64(uint64_t value) {
  memcpy(buffer_ + offset_, &value, sizeof(uint64_t));
  offset_ += sizeof(uint64_t);
}

void SerializationBuffer::WriteInt64(int64_t value) {
  memcpy(buffer_ + offset_, &value, sizeof(int64_t));
  offset_ += sizeof(int64_t);
}

void SerializationBuffer::WriteBytes(const void* data, size_t size) {
  if (size > 0 && data != nullptr) {
    // Use memcpy with optimization hints for large copies
    // For large buffers, memcpy should use SIMD instructions
    // Compiler will optimize this based on -O3 and -march=native flags
    memcpy(buffer_ + offset_, data, size);

    // Optional: Force memory to be loaded into cache for large copies
    // This helps with subsequent operations on the copied data
    // if (size > 1024 * 1024) {  // > 1 MB
    //   // Clflush hint to cache (compiler may optimize this away)
    //   // In practice, memcpy already does optimal caching
    // }
  }
  offset_ += size;
}

uint32_t SerializationBuffer::ReadUInt32(bool network_order) {
  uint32_t value;
  memcpy(&value, buffer_ + offset_, sizeof(uint32_t));
  offset_ += sizeof(uint32_t);
  return network_order ? ntohl(value) : value;
}

uint64_t SerializationBuffer::ReadUInt64() {
  uint64_t value;
  memcpy(&value, buffer_ + offset_, sizeof(uint64_t));
  offset_ += sizeof(uint64_t);
  return value;
}

int64_t SerializationBuffer::ReadInt64() {
  int64_t value;
  memcpy(&value, buffer_ + offset_, sizeof(int64_t));
  offset_ += sizeof(int64_t);
  return value;
}

void SerializationBuffer::ReadBytes(void* dest, size_t size) {
  memcpy(dest, buffer_ + offset_, size);
  offset_ += size;
}

const void* SerializationBuffer::GetCurrentPtr() const {
  return buffer_ + offset_;
}

void SerializationBuffer::SeekTo(size_t offset) { offset_ = offset; }

bool SerializationBuffer::CanRead(size_t bytes) const {
  return offset_ + bytes <= size_;
}

void SerializationBuffer::ValidateSize(size_t min_size) const {
  if (size_ < min_size) {
    throw std::runtime_error("Buffer size too small: " + std::to_string(size_) +
                             " < " + std::to_string(min_size));
  }
}

std::string SerializationBuffer::HexString(size_t length) const {
  size_t read_length = std::min(length, size_);
  return BinaryToHex(buffer_, read_length);
  // std::stringstream ss;
  // for (size_t i = 0; i < read_length; ++i) {
  //   ss << std::hex << std::setw(2) << std::setfill('0')
  //      << static_cast<int>(buffer_[i]);
  // }
  // return ss.str();
}
