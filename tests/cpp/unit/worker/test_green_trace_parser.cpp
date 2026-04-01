#include <gtest/gtest.h>

#if 0
#include "scheduler/green_trace_parser.h"
#endif

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#if __has_include("csrc/scheduler/green_trace_parser.h")
#include "csrc/scheduler/green_trace_parser.h"
#elif __has_include("scheduler/green_trace_parser.h")
#include "scheduler/green_trace_parser.h"
#else
struct GreenTraceEntry {
  int64_t timestamp;
  int num_sms;
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
  TraceTimeUnit time_unit;
  TraceClockMode clock_mode;
};

class GreenTraceParser {
 public:
  static bool Parse(const std::string& path,
                    std::vector<GreenTraceEntry>& entries,
                    GreenTraceConfig& config, std::string& error);

  static bool Validate(const std::vector<GreenTraceEntry>& entries,
                       const std::set<int>& valid_sm_set, std::string& error);

  static int SmCountAtTime(const std::vector<GreenTraceEntry>& entries,
                           int64_t elapsed, int default_sm);
};
#endif

std::string WriteTempTrace(const std::string& content) {
  char file_template[] = "/tmp/test_trace_XXXXXX.txt";
  int fd = mkstemps(file_template, 4);
  if (fd == -1) {
    return "";
  }
  close(fd);

  std::ofstream out(file_template, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    std::remove(file_template);
    return "";
  }

  out << content;
  out.close();
  return std::string(file_template);
}

class GreenTraceParserTest : public ::testing::Test {
 protected:
  std::string MakeTraceFile(const std::string& content) {
    std::string path = WriteTempTrace(content);
    EXPECT_FALSE(path.empty());
    if (!path.empty()) {
      temp_files_.push_back(path);
    }
    return path;
  }

  void TearDown() override {
    for (const auto& path : temp_files_) {
      std::remove(path.c_str());
    }
    temp_files_.clear();
  }

  std::vector<std::string> temp_files_;
};

TEST_F(GreenTraceParserTest, ParseV1_BasicFormat) {
  const std::string path = MakeTraceFile("0 8 1000000\n1000000 32 1000000\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].timestamp, 0);
  EXPECT_EQ(entries[0].num_sms, 8);
  EXPECT_EQ(entries[1].timestamp, 1000000);
  EXPECT_EQ(entries[1].num_sms, 32);
}

TEST_F(GreenTraceParserTest, ParseV1_CommentsAndBlanks) {
  const std::string path = MakeTraceFile(
      "# comment\n"
      "\n"
      "0 8 1000000\n"
      "# another comment\n"
      "1000000 32 1000000\n"
      "\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].timestamp, 0);
  EXPECT_EQ(entries[0].num_sms, 8);
  EXPECT_EQ(entries[1].timestamp, 1000000);
  EXPECT_EQ(entries[1].num_sms, 32);
}

TEST_F(GreenTraceParserTest, ParseV2_BasicCSV) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=s\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "1,32\n"
      "3,16\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 3u);
  EXPECT_EQ(entries[0].timestamp, 0);
  EXPECT_EQ(entries[0].num_sms, 8);
  EXPECT_EQ(entries[1].timestamp, 1000000);
  EXPECT_EQ(entries[1].num_sms, 32);
  EXPECT_EQ(entries[2].timestamp, 3000000);
  EXPECT_EQ(entries[2].num_sms, 16);
}

TEST_F(GreenTraceParserTest, ParseV2_WithTags) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=s\n"
      "timestamp,num_sms,tag\n"
      "0,8,warmup\n"
      "1,32,peak\n"
      "2,16,cooldown\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 3u);
  EXPECT_EQ(entries[0].tag, "warmup");
  EXPECT_EQ(entries[1].tag, "peak");
  EXPECT_EQ(entries[2].tag, "cooldown");
}

TEST_F(GreenTraceParserTest, ParseV2_MillisecondUnit) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=ms\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "1000,32\n"
      "3000,16\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 3u);
  EXPECT_EQ(entries[0].timestamp, 0);
  EXPECT_EQ(entries[1].timestamp, 1000000);
  EXPECT_EQ(entries[2].timestamp, 3000000);
}

TEST_F(GreenTraceParserTest, ParseV2_MicrosecondUnit) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "1000000,32\n"
      "3000000,16\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 3u);
  EXPECT_EQ(entries[0].timestamp, 0);
  EXPECT_EQ(entries[1].timestamp, 1000000);
  EXPECT_EQ(entries[2].timestamp, 3000000);
}

TEST_F(GreenTraceParserTest, Parse_AutoDetectV1) {
  const std::string path = MakeTraceFile("0 8 1000000\n1000000 16 1000000\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].num_sms, 8);
  EXPECT_EQ(entries[1].num_sms, 16);
}

TEST_F(GreenTraceParserTest, Parse_AutoDetectV2) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=s\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "1,32\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].timestamp, 0);
  EXPECT_EQ(entries[1].timestamp, 1000000);
}

TEST_F(GreenTraceParserTest, Parse_NonMonotonic_Fails) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "5,32\n"
      "3,16\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  // ParseV2 rejects non-monotonic timestamps at parse time
  EXPECT_FALSE(GreenTraceParser::Parse(path, entries, config, error));
  EXPECT_FALSE(error.empty());
}
TEST_F(GreenTraceParserTest, Parse_EmptyFile_Fails) {
  const std::string path = MakeTraceFile("");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  // Parse returns false on empty file (no data rows)
  EXPECT_FALSE(GreenTraceParser::Parse(path, entries, config, error));
  EXPECT_FALSE(error.empty());
}

TEST_F(GreenTraceParserTest, Parse_InvalidLine_Fails) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "this,is,invalid\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  EXPECT_FALSE(GreenTraceParser::Parse(path, entries, config, error));
  EXPECT_FALSE(error.empty());
}

TEST_F(GreenTraceParserTest, Validate_InvalidSmCount) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "1,24\n"
      "2,32\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  std::set<int> valid_sm_counts{8, 16, 32};
  std::string validate_error;
  EXPECT_FALSE(
      GreenTraceParser::Validate(entries, valid_sm_counts, validate_error));
  EXPECT_FALSE(validate_error.empty());
}

TEST_F(GreenTraceParserTest, Validate_AllValid) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "1,16\n"
      "2,32\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  std::set<int> valid_sm_counts{8, 16, 32};
  std::string validate_error;
  EXPECT_TRUE(
      GreenTraceParser::Validate(entries, valid_sm_counts, validate_error));
  EXPECT_TRUE(validate_error.empty());
}

TEST_F(GreenTraceParserTest, SmCountAtTime_BinarySearch) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "1000000,32\n"
      "3000000,16\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 3u);

  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 0, 64), 8);
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 500000, 64), 8);
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 1000000, 64), 32);
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 2000000, 64), 32);
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 3000000, 64), 16);
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 9999999, 64), 16);
}

TEST_F(GreenTraceParserTest, SmCountAtTime_BeforeFirstEntry) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,8\n"
      "1000000,32\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, -1, 64), 64);
}

TEST_F(GreenTraceParserTest, SmCountAtTime_SingleEntry) {
  const std::string path = MakeTraceFile(
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,8\n");

  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config{};
  std::string error;

  ASSERT_TRUE(GreenTraceParser::Parse(path, entries, config, error)) << error;
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 0, 64), 8);
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 1, 64), 8);
  EXPECT_EQ(GreenTraceParser::SmCountAtTime(entries, 9999999, 64), 8);
}
