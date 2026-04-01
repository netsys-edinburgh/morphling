#pragma once

#include <cstdint>
#include <istream>
#include <set>
#include <string>
#include <vector>

struct GreenTraceEntry {
  int64_t timestamp = 0;
  int num_sms = 0;
  std::string tag;
};

enum class TraceTimeUnit {
  kMicroseconds,
  kMilliseconds,
  kSeconds,
};

enum class TraceClockMode {
  kWall,
  kStep,
};

struct GreenTraceConfig {
  TraceTimeUnit time_unit = TraceTimeUnit::kMicroseconds;
  TraceClockMode clock_mode = TraceClockMode::kWall;
};

class GreenTraceParser {
 public:
  static bool Parse(const std::string& path,
                    std::vector<GreenTraceEntry>& entries,
                    GreenTraceConfig& config, std::string& error);

  static bool ParseV1(std::istream& in, std::vector<GreenTraceEntry>& entries,
                      GreenTraceConfig& config, std::string& error);

  static bool ParseV2(std::istream& in, std::vector<GreenTraceEntry>& entries,
                      GreenTraceConfig& config, std::string& error);

  static bool Validate(const std::vector<GreenTraceEntry>& entries,
                       const std::set<int>& valid_sm_counts,
                       std::string& error);

  static int SmCountAtTime(const std::vector<GreenTraceEntry>& entries,
                           int64_t elapsed, int default_sm_count);
};
