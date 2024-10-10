#pragma once

#include <uuid/uuid.h>

#include <string>

inline std::string GenUUID() {
  uuid_t uuid;
  uuid_generate(uuid);
  char uuid_str[37];
  uuid_unparse(uuid, uuid_str);
  return std::string(uuid_str);
}