#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "../../../../csrc/core/env_cfg.h"

std::filesystem::path WriteConfigFile(const std::string& filename,
                                      const std::string& worker_body) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / filename;

  std::ofstream os(path);
  os << "[network]\n";
  os << "listen_ip = 127.0.0.1\n";
  os << "listen_port = 39000\n\n";
  os << "[worker]\n";
  os << "thread = 2\n";
  os << worker_body;
  os.close();

  return path;
}

TEST(TransportModeConfigTest, DefaultsToNetworkWhenMissing) {
  const auto cfg_path = WriteConfigFile("morphling_transport_mode_default.ini",
                                        "pool_mode = gpu\n");

  ProxyEnvCfg cfg;
  ASSERT_EQ(cfg.Initialize(cfg_path.string()), 0);
  EXPECT_EQ(cfg.transport_mode, TransportMode::NETWORK);
  EXPECT_STREQ(TransportModeToString(cfg.transport_mode), "network");

  std::error_code ec;
  std::filesystem::remove(cfg_path, ec);
}

TEST(TransportModeConfigTest, ParsesExplicitEmulatorValue) {
  const auto cfg_path = WriteConfigFile("morphling_transport_mode_emulator.ini",
                                        "pool_mode = gpu\n"
                                        "transport_mode = emulator\n");

  ProxyEnvCfg cfg;
  ASSERT_EQ(cfg.Initialize(cfg_path.string()), 0);
  EXPECT_EQ(cfg.transport_mode, TransportMode::EMULATOR);
  EXPECT_STREQ(TransportModeToString(cfg.transport_mode), "emulator");

  std::error_code ec;
  std::filesystem::remove(cfg_path, ec);
}
