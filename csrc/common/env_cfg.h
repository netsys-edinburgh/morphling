#pragma once

#include <string>

struct ProxyEnvCfg {
  std::string listen_ip;
  unsigned listen_port;

  // worker
  int thread;

  // log
  std::string log_path;
  std::string log_level;
  std::string log_type;

  // internal
  int cleanup_wait;
  int64_t tcp_timeout;

  void* instance;

  int Initialize(const std::string& cfg_file);
};
