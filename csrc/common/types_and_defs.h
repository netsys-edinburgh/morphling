#pragma once
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "rttr_registration.h"

#define PARAM_META_FILE "param_meta_map.json"
#define BUFFER_SHM_NAME "/emulator_shm_buffer"
#define PARAM_SHM_NAME "/emulator_shm_param"
#define ALIGNMENT 512

typedef std::uint32_t TensorID;
typedef std::size_t HashID;
typedef std::size_t NodeID;
typedef std::uint64_t GraphID;
typedef std::uint64_t RequestID;

typedef std::unordered_map<std::string, ShmMeta> ParamShmMap;

// #define KB 1024
// #define MB (KB * KB)
// #define GB (KB * KB * KB)

constexpr size_t KB = 1024;
constexpr size_t MB = KB * KB;
constexpr size_t GB = KB * KB * KB;

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};

template <typename T>
struct CudaHostDeleter {
  void operator()(T* ptr) const {
    if (ptr) {
      cudaFreeHost(ptr);
    }
  }
};

#define GETENV(name, default_value) \
  (std::getenv(name) ? std::getenv(name) : default_value)

constexpr size_t operator"" _KB(unsigned long long val) { return val * KB; }
constexpr size_t operator"" _MB(unsigned long long val) { return val * MB; }
constexpr size_t operator"" _GB(unsigned long long val) { return val * GB; }

// A constexpr function to convert any pointer (const or non-const) to void*
template <typename T>
constexpr void* pointer_to_void(T* ptr) {
  return const_cast<void*>(reinterpret_cast<const void*>(
      ptr));  // Cast to void* while preserving constness
}

// // Overload for const pointers
// template <typename T>
// constexpr const void* pointer_to_void(const T* ptr) {
//   return static_cast<const void*>(ptr);  // Cast to const void*
// }

// A constexpr function to convert any const T* pointer to void*
template <typename T>
constexpr void* pointer_to_void(const T* ptr) {
  return const_cast<void*>(reinterpret_cast<const void*>(
      ptr));  // Cast to void* while preserving constness
}

// Helper macros to generate enum and string mappings
#define ENUM_ENTRY_COMMA(value, EnumType) value,
#define ENUM_CASE(value, EnumType) \
  case EnumType::value:            \
    return #value;
#define STRING_CASE(value, EnumType) \
  if (s == #value) return EnumType::value;

// Macro to define enum class, enum to string, and string to enum functions
#define DEFINE_ENUM_CLASS(EnumType, ENUM_VALUES)                              \
  enum class EnumType { ENUM_VALUES(ENUM_ENTRY_COMMA, EnumType) Unknown };    \
                                                                              \
  /* Enum to string function */                                               \
  inline const char* EnumType##ToString(EnumType v) {                         \
    switch (v) {                                                              \
      ENUM_VALUES(ENUM_CASE, EnumType)                                        \
      default:                                                                \
        return "Unknown";                                                     \
    }                                                                         \
  }                                                                           \
                                                                              \
  /* String to enum function */                                               \
  inline EnumType StringTo##EnumType(const std::string& s) {                  \
    ENUM_VALUES(STRING_CASE, EnumType)                                        \
    return EnumType::Unknown;                                                 \
  }                                                                           \
                                                                              \
  /* Specialize fmt::formatter for spdlog support */                          \
  template <>                                                                 \
  struct fmt::formatter<EnumType> : fmt::formatter<std::string> {             \
    template <typename FormatContext>                                         \
    auto format(EnumType v, FormatContext& ctx) {                             \
      return fmt::formatter<std::string>::format(EnumType##ToString(v), ctx); \
    }                                                                         \
  };

#include <iostream>
#include <list>
#include <unordered_map>
#include <utility>

template <typename KeyType, typename ValueType>
class LRUCache {
 public:
  explicit LRUCache(size_t cap) : capacity_(cap) {}

  void Put(const KeyType& key, const ValueType& value) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // Update item if it exists and move it to the back of the list
      lru_.erase(it->second.second);
    } else {
      // Check capacity_ and remove the least recently used item
      if (cache_.size() == capacity_) {
        cache_.erase(lru_.front());
        lru_.pop_front();
      }
    }
    // Insert new item at the back of the list
    lru_.push_back(key);
    cache_[key] = {value, --lru_.end()};
  }

  ValueType Get(const KeyType& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      throw std::range_error("Key not found");
    }
    // Move the accessed item to the back of the list
    lru_.erase(it->second.second);
    lru_.push_back(key);
    it->second.second = --lru_.end();
    return it->second.first;
  }

  bool Exist(const KeyType& key) const {
    return cache_.find(key) != cache_.end();
  }

 private:
  std::list<KeyType> lru_;  // Stores keys of cache items
  std::unordered_map<
      KeyType, std::pair<ValueType, typename std::list<KeyType>::iterator>>
      cache_;
  size_t capacity_;
};
