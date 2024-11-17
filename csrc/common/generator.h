#pragma once

#include <uuid/uuid.h>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

inline std::string GenUUID() {
  uuid_t uuid;
  uuid_generate(uuid);
  char uuid_str[37];
  uuid_unparse(uuid, uuid_str);
  return std::string(uuid_str);
}

inline std::string CurrentTimeString() {
  // Get current time as time_point
  auto now = std::chrono::system_clock::now();

  // Convert time_point to system time for breaking down into components
  auto now_c = std::chrono::system_clock::to_time_t(now);
  auto now_tm = *std::localtime(&now_c);

  // Get the current time as milliseconds
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) %
                1000;

  // Use stringstream to format the time
  std::ostringstream oss;
  oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();

  return oss.str();
}