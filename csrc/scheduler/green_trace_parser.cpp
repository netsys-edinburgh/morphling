#include "green_trace_parser.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "utils/logger.h"

namespace {

std::string Trim(const std::string& s) {
  const size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

bool ParseInt64(const std::string& s, int64_t& out) {
  std::istringstream iss(s);
  iss >> out;
  return iss && iss.eof();
}

bool ParseInt(const std::string& s, int& out) {
  std::istringstream iss(s);
  iss >> out;
  return iss && iss.eof();
}

bool ParseMetadata(const std::string& line, GreenTraceConfig& config,
                   std::string& error, int line_num) {
  std::string s = Trim(line);
  if (s.empty() || s[0] != '#') return true;

  s = Trim(s.substr(1));
  if (s.empty()) return true;

  const size_t eq = s.find('=');
  if (eq == std::string::npos) return true;

  const std::string key = Trim(s.substr(0, eq));
  const std::string value = Trim(s.substr(eq + 1));

  if (key == "time_unit") {
    if (value == "us" || value == "microseconds") {
      config.time_unit = TraceTimeUnit::kMicroseconds;
      return true;
    }
    if (value == "ms" || value == "milliseconds") {
      config.time_unit = TraceTimeUnit::kMilliseconds;
      return true;
    }
    if (value == "s" || value == "seconds") {
      config.time_unit = TraceTimeUnit::kSeconds;
      return true;
    }
    error =
        "Invalid time_unit at line " + std::to_string(line_num) + ": " + value;
    return false;
  }

  if (key == "clock_mode") {
    if (value == "wall") {
      config.clock_mode = TraceClockMode::kWall;
      return true;
    }
    if (value == "step") {
      config.clock_mode = TraceClockMode::kStep;
      return true;
    }
    error =
        "Invalid clock_mode at line " + std::to_string(line_num) + ": " + value;
    return false;
  }

  if (key == "semantics") {
    // step_function is the SM schedule semantics (always step function).
    // It does NOT affect clock_mode (wall vs step).
    return true;
  }

  return true;
}

}  // namespace

bool GreenTraceParser::Parse(const std::string& path,
                             std::vector<GreenTraceEntry>& entries,
                             GreenTraceConfig& config, std::string& error) {
  entries.clear();
  config = GreenTraceConfig{};
  error.clear();

  std::ifstream file(path);
  if (!file.is_open()) {
    error = "Cannot open green trace file: " + path;
    LOG_ERROR << error;
    return false;
  }

  bool saw_v2_metadata = false;
  bool decided = false;
  bool use_v2 = false;
  std::string line;

  while (std::getline(file, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) continue;

    if (trimmed[0] == '#') {
      if (trimmed.find('=') != std::string::npos) {
        saw_v2_metadata = true;
      }
      continue;
    }

    use_v2 = saw_v2_metadata || trimmed.find(',') != std::string::npos;
    decided = true;
    break;
  }

  if (!decided) {
    error = "Green trace file has no data rows: " + path;
    LOG_ERROR << error;
    return false;
  }

  file.clear();
  file.seekg(0);
  if (!file.good()) {
    error = "Failed to rewind green trace file: " + path;
    LOG_ERROR << error;
    return false;
  }

  if (use_v2) {
    return ParseV2(file, entries, config, error);
  }
  return ParseV1(file, entries, config, error);
}

bool GreenTraceParser::ParseV1(std::istream& in,
                               std::vector<GreenTraceEntry>& entries,
                               GreenTraceConfig& config, std::string& error) {
  entries.clear();
  config = GreenTraceConfig{};
  error.clear();

  std::string line;
  int line_num = 0;
  while (std::getline(in, line)) {
    ++line_num;
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') continue;

    std::istringstream iss(trimmed);
    int64_t timestamp = 0;
    int num_sms = 0;
    int64_t duration_ignored = 0;
    if (!(iss >> timestamp >> num_sms >> duration_ignored)) {
      error =
          "ParseV1 failed at line " + std::to_string(line_num) + ": " + line;
      LOG_ERROR << error;
      return false;
    }

    std::string extra;
    if (iss >> extra) {
      error = "ParseV1 extra token at line " + std::to_string(line_num) + ": " +
              extra;
      LOG_ERROR << error;
      return false;
    }

    if (!entries.empty() && timestamp < entries.back().timestamp) {
      error = "Non-monotonic timestamp at line " + std::to_string(line_num);
      LOG_ERROR << error;
      return false;
    }

    GreenTraceEntry entry;
    entry.timestamp = timestamp;
    entry.num_sms = num_sms;
    entries.push_back(std::move(entry));
  }

  if (entries.empty()) {
    error = "ParseV1 found no valid data rows";
    LOG_ERROR << error;
    return false;
  }

  return true;
}

bool GreenTraceParser::ParseV2(std::istream& in,
                               std::vector<GreenTraceEntry>& entries,
                               GreenTraceConfig& config, std::string& error) {
  entries.clear();
  config = GreenTraceConfig{};
  // Default clock_mode is kWall (from GreenTraceConfig default)
  error.clear();

  std::string line;
  int line_num = 0;
  bool header_skipped = false;
  while (std::getline(in, line)) {
    ++line_num;
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) continue;

    if (trimmed[0] == '#') {
      if (!ParseMetadata(trimmed, config, error, line_num)) {
        LOG_ERROR << error;
        return false;
      }
      continue;
    }

    std::vector<std::string> fields;
    std::stringstream ss(trimmed);
    std::string field;
    while (std::getline(ss, field, ',')) {
      fields.push_back(Trim(field));
    }

    if (fields.size() < 2 || fields.size() > 3) {
      error = "ParseV2 invalid CSV field count at line " +
              std::to_string(line_num) + ": " + line;
      LOG_ERROR << error;
      return false;
    }

    GreenTraceEntry entry;
    if (!ParseInt64(fields[0], entry.timestamp)) {
      // Allow exactly one header row (e.g., "timestamp,num_sms")
      if (!header_skipped) {
        header_skipped = true;
        continue;
      }
      error = "ParseV2 invalid timestamp at line " + std::to_string(line_num) +
              ": " + fields[0];
      LOG_ERROR << error;
      return false;
    }

    if (!ParseInt(fields[1], entry.num_sms)) {
      error = "ParseV2 invalid num_sms at line " + std::to_string(line_num) +
              ": " + fields[1];
      LOG_ERROR << error;
      return false;
    }

    if (fields.size() == 3) {
      entry.tag = fields[2];
    }

    if (!entries.empty() && entry.timestamp < entries.back().timestamp) {
      error = "Non-monotonic timestamp at line " + std::to_string(line_num);
      LOG_ERROR << error;
      return false;
    }

    entries.push_back(std::move(entry));
  }

  if (entries.empty()) {
    error = "ParseV2 found no valid data rows";
    LOG_ERROR << error;
    return false;
  }

  // Convert timestamps to microseconds based on time_unit
  if (config.time_unit == TraceTimeUnit::kSeconds) {
    for (auto& e : entries) e.timestamp *= 1000000;
  } else if (config.time_unit == TraceTimeUnit::kMilliseconds) {
    for (auto& e : entries) e.timestamp *= 1000;
  }

  return true;
}

bool GreenTraceParser::Validate(const std::vector<GreenTraceEntry>& entries,
                                const std::set<int>& valid_sm_counts,
                                std::string& error) {
  error.clear();

  for (size_t i = 0; i < entries.size(); ++i) {
    if (i > 0 && entries[i].timestamp < entries[i - 1].timestamp) {
      error = "Non-monotonic timestamp at index " + std::to_string(i);
      LOG_ERROR << error;
      return false;
    }

    if (!valid_sm_counts.empty() &&
        valid_sm_counts.find(entries[i].num_sms) == valid_sm_counts.end()) {
      error = "Invalid num_sms=" + std::to_string(entries[i].num_sms) +
              " at index " + std::to_string(i);
      LOG_ERROR << error;
      return false;
    }
  }

  return true;
}

int GreenTraceParser::SmCountAtTime(const std::vector<GreenTraceEntry>& entries,
                                    int64_t elapsed, int default_sm_count) {
  if (entries.empty()) return default_sm_count;

  auto it = std::upper_bound(entries.begin(), entries.end(), elapsed,
                             [](int64_t t, const GreenTraceEntry& entry) {
                               return t < entry.timestamp;
                             });

  if (it == entries.begin()) return default_sm_count;
  return std::prev(it)->num_sms;
}
