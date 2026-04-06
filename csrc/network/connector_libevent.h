#ifndef CONNECTOR_LIBEVENT_H
#define CONNECTOR_LIBEVENT_H

#include <string>

#include "uevent.h"

namespace uevent {

class EventLoopLibevent;

struct ReconnectConfig {
  bool enabled = false;
  int max_retries = 5;
  int initial_delay_ms = 100;
  int max_delay_ms = 10000;
  float backoff_multiplier = 2.0;
};

class ConnectorLibevent : public ConnectorUevent {
 public:
  ConnectorLibevent(UeventLoop* loop, const UsockAddress& peer_addr,
                    const std::string& name);

  virtual ~ConnectorLibevent();
  virtual int Connect();
  void SetReconnectConfig(const ReconnectConfig& config);

 private:
  void OnConnectionSuccess(const ConnectionUeventPtr& conn);
  void OnConnectionClosed(const ConnectionUeventPtr& conn);
  void ScheduleReconnect();
  void ResetReconnectState();

  EventLoopLibevent* loop_;
  ReconnectConfig reconnect_config_;
  int reconnect_attempts_;
  int current_delay_ms_;
  TimerId reconnect_timer_id_;
  bool reconnect_timer_pending_;
};

}  // namespace uevent

#endif
