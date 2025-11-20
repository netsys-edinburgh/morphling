#pragma once

#include <string>

struct ProxyEnvCfg {
  std::string listen_ip;
  unsigned listen_port;

  // worker
  int thread;
  int64_t block_size;
  int64_t num_device;
  int max_inflight;

  // log
  std::string log_path;
  std::string log_level;
  std::string log_type;

  // internal
  int cleanup_wait;
  int64_t tcp_timeout;
  int enable_cli_cache;

  void* instance;

  int Initialize(const std::string& cfg_file);
};
