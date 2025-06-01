#pragma once

#include <cstdint>
#include <rttr/registration>
#include <rttr/type>

#include "utils/json_reader.h"

#define RTTR_FORMATTER(T)                                                   \
  template <>                                                               \
  struct fmt::formatter<T> {                                                \
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); } \
    template <typename FormatContext>                                       \
    auto format(const T& p, FormatContext& ctx) {                           \
      auto prop_names = rttr::type::get<T>().get_properties();              \
      std::string prop_str;                                                 \
      for (auto& prop : prop_names) {                                       \
        if (prop.get_type().is_pointer()) {                                 \
          void* ptr = prop.get_value(p).get_value<void*>();                 \
          prop_str +=                                                       \
              fmt::format("{}={:p}, ", prop.get_name().to_string(), ptr);   \
          continue;                                                         \
        }                                                                   \
        prop_str += fmt::format("{}={}, ", prop.get_name().to_string(),     \
                                prop.get_value(p).to_string());             \
      }                                                                     \
      return format_to(ctx.out(), "{{{}}}", prop_str);                      \
    }                                                                       \
  };

struct ShmMeta {
  int id;
  void* ptr;
  size_t size;
  std::string name;
  bool is_remote = false;
};

struct ParamMeta {
  uint32_t id;
  size_t size;
  // size_t shm_offset;
  size_t file_offset;

  FROM_JSON_METHOD(ParamMeta)
};

RTTR_REGISTRATION {
  rttr::registration::class_<ShmMeta>("ShmMeta")
      .property("id", &ShmMeta::id)
      .property("ptr", &ShmMeta::ptr)
      .property("size", &ShmMeta::size)
      .property("name", &ShmMeta::name);

  rttr::registration::class_<ParamMeta>("ParamMeta")
      .property("id", &ParamMeta::id)
      .property("size", &ParamMeta::size)
      // .property("shm_offset", &ParamMeta::shm_offset)
      .property("file_offset", &ParamMeta::file_offset);
}

RTTR_FORMATTER(ShmMeta)
RTTR_FORMATTER(ParamMeta)
