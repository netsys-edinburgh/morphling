#include "env_cfg.h"

#include "base/ini_config.h"
#include "base/logging.h"

#define PARSE_INT_ENVCFG(cfg, section, key, required, default_value, value) \
  {                                                                         \
    if (cfg.Count(section, key) == 0) {                                     \
      if (!required)                                                        \
        value = default_value;                                              \
      else {                                                                \
        LOG_ERROR << "missing " << section << "." << key;                   \
        return -1;                                                          \
      }                                                                     \
    } else {                                                                \
      value = cfg.IntValue(section, key);                                   \
    }                                                                       \
  }

#define PARSE_STR_ENVCFG(cfg, section, key, required, default_value, value) \
  {                                                                         \
    if (cfg.Count(section, key) == 0) {                                     \
      if (!required)                                                        \
        value = default_value;                                              \
      else {                                                                \
        LOG_ERROR << "missing " << section << "." << key;                   \
        return -1;                                                          \
      }                                                                     \
    } else {                                                                \
      value = cfg.GetValue(section, key);                                   \
    }                                                                       \
  }

int ProxyEnvCfg::Initialize(const std::string& cfg_file) {
  base::IniConfig parser;
  if (parser.LoadFromFile(cfg_file.c_str())) {
    return -1;
  }
  PARSE_STR_ENVCFG(parser, "log", "log_path", false, "/root/log", log_path);
  PARSE_STR_ENVCFG(parser, "log", "log_level", false, "info", log_level);
  PARSE_STR_ENVCFG(parser, "log", "log_type", false, "terminal", log_type);

  PARSE_INT_ENVCFG(parser, "network", "listen_port", true, 0, listen_port);
  PARSE_STR_ENVCFG(parser, "network", "listen_ip", true, "", listen_ip);

  PARSE_INT_ENVCFG(parser, "worker", "thread", false, 2, thread);

  PARSE_INT_ENVCFG(parser, "internal", "cleanup_wait", false, 60, cleanup_wait);
  PARSE_INT_ENVCFG(parser, "internal", "tcp_timeout", false, 5, tcp_timeout);

  return 0;
}
