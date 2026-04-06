#include "connector_libevent.h"

#include <event2/bufferevent.h>

#include <algorithm>
#include <functional>

#include "connection_libevent.h"
#include "eventloop_libevent.h"
#include "muduo_base/logging.h"

namespace uevent {

ConnectorLibevent::ConnectorLibevent(UeventLoop* loop,
                                     const UsockAddress& peer_addr,
                                     const std::string& name)
    : ConnectorUevent(loop, peer_addr, name),
      loop_(reinterpret_cast<EventLoopLibevent*>(loop)),
      reconnect_attempts_(0),
      current_delay_ms_(reconnect_config_.initial_delay_ms),
      reconnect_timer_pending_(false) {}

// 析构connector时一定要确保conn是关闭的，否则
// 其中的bev_wrapper持有conn, conn得不到释放
ConnectorLibevent::~ConnectorLibevent() {
  reconnect_config_.enabled = false;
  if (reconnect_timer_pending_) {
    loop_->CancelTimer(reconnect_timer_id_);
    reconnect_timer_pending_ = false;
  }
  if (conn_ && conn_->IsClosed() == false) {
    conn_->ForceClose();
  }
}

int ConnectorLibevent::Connect() {
  loop_->AssertInLoopThread();
  if (conn_) {
    LOG_ERROR << "connection has exist in connector, peer address:"
              << peer_addr_.ToString();
    return -1;
  }
  int64_t conn_id = 0;
  // 对于connect产生的连接使用 Ip << 16 | port 作为id
  if (peer_addr_.family() == AF_INET) {
    conn_id = (static_cast<int64_t>(peer_addr_.IpNetEndian()) << 16) |
              peer_addr_.PortNetEndian();
  }
  ConnectionLibevent* conn_libevent =
      new ConnectionLibevent(loop_, -1, conn_id, name_, peer_addr_);
  ConnectionUeventPtr conn(conn_libevent);
  LOG_DEBUG << "connect to peer address:" << peer_addr_.ToString();
  conn_ = conn;
  conn_->Init();
  conn_->SetConnectionSuccessCb(std::bind(
      &ConnectorLibevent::OnConnectionSuccess, this, std::placeholders::_1));
  conn_->SetConnectionClosedCb(std::bind(&ConnectorLibevent::OnConnectionClosed,
                                         this, std::placeholders::_1));
  conn_->SetMessageReadCb(message_read_cb_);
  conn_->SetMessageWriteCb(message_write_cb_);
  conn_->ConnectionEnable();
  if (bufferevent_socket_connect(conn_libevent->GetInnerBev(),
                                 peer_addr_.GetSockAddr(),
                                 peer_addr_.GetSockAddrLen()) < 0) {
    LOG_SYSERR << "connect return error, peer address: "
               << peer_addr_.ToString();
    // bufferevent_socket_connect 失败, 如果会触发连接失败的回调
    // 则会调用ForceClose, 关闭这个无效的连接, 如果不触发回调会有
    // 一个无效连接，但是是正常状态，这样很危险，长期使用无效的连接(bufferevent)
    // 而无法感知（如果没有判断返回值的话),
    // 所以这里ForceClose一下，如果已经closed 直接返回就行了。
    conn_->ForceClose();
    return -1;
  }
  // connect 是同步的， 到这里连接已经成功,可以设置fd
  conn_->SetFd();
  conn_->SetState(ConnectionUevent::kConnecting);
  return 0;
}

void ConnectorLibevent::SetReconnectConfig(const ReconnectConfig& config) {
  reconnect_config_ = config;
  reconnect_config_.max_retries = std::max(0, reconnect_config_.max_retries);
  reconnect_config_.initial_delay_ms =
      std::max(0, reconnect_config_.initial_delay_ms);
  reconnect_config_.max_delay_ms = std::max(reconnect_config_.initial_delay_ms,
                                            reconnect_config_.max_delay_ms);
  reconnect_config_.backoff_multiplier =
      std::max(1.0f, reconnect_config_.backoff_multiplier);
  ResetReconnectState();
}

void ConnectorLibevent::OnConnectionSuccess(const ConnectionUeventPtr& conn) {
  ResetReconnectState();
  connection_success_cb_(conn);
}

void ConnectorLibevent::OnConnectionClosed(const ConnectionUeventPtr& conn) {
  connection_closed_cb_(conn);
  if (reconnect_config_.enabled) {
    ScheduleReconnect();
  }
}

void ConnectorLibevent::ScheduleReconnect() {
  if (reconnect_timer_pending_) {
    return;
  }
  if (reconnect_attempts_ >= reconnect_config_.max_retries) {
    LOG_ERROR << "reconnect max retries reached, peer address: "
              << peer_addr_.ToString() << ", retries: " << reconnect_attempts_;
    return;
  }

  int delay_ms = current_delay_ms_;
  reconnect_timer_pending_ = true;
  reconnect_timer_id_ =
      loop_->RunAfter(static_cast<double>(delay_ms) / 1000.0, [this]() {
        reconnect_timer_pending_ = false;
        if (!reconnect_config_.enabled) {
          return;
        }
        if (conn_ && !conn_->IsClosed()) {
          return;
        }

        ++reconnect_attempts_;
        LOG_INFO << "reconnect attempt " << reconnect_attempts_
                 << " to peer address: " << peer_addr_.ToString();
        ReConnect();

        int next_delay = static_cast<int>(current_delay_ms_ *
                                          reconnect_config_.backoff_multiplier);
        next_delay = std::max(next_delay, reconnect_config_.initial_delay_ms);
        current_delay_ms_ =
            std::min(next_delay, reconnect_config_.max_delay_ms);
      });
}

void ConnectorLibevent::ResetReconnectState() {
  reconnect_attempts_ = 0;
  current_delay_ms_ = reconnect_config_.initial_delay_ms;
  if (reconnect_timer_pending_) {
    loop_->CancelTimer(reconnect_timer_id_);
    reconnect_timer_pending_ = false;
  }
}

}  // namespace uevent
