#pragma once

#include "common/env_cfg.h"
#include "common/lru.h"
#include "common/pytorch_defs.h"
#include "morphling.pb.h"
#include "network/connector_libevent.h"
#include "network/uevent.h"
#include "network/ueventloop_thread.h"
#include "server_base.h"

class ProxyCliHandle : public uevent::LoopHandle {
 public:
  ProxyCliHandle(ProxyEnvCfg& ctx, uevent::UeventLoop* loop);

  static void ThreadInit(uevent::UeventLoop* loop);
  static uevent::LoopHandle* CreateMyself(ProxyEnvCfg& ctx,
                                          uevent::UeventLoop* loop) {
    return new ProxyCliHandle(ctx, loop);
  }

 public:
  // void PbRequestCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionSuccessCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionClosedCb(const uevent::ConnectionUeventPtr& conn);

  void ResponseToCaller(const uevent::ConnectionUeventPtr& conn,
                        const MatrixPartition& partition);
  void HandlePartition(const uevent::ConnectionUeventPtr& conn,
                       const MatrixPartition& partition);

 private:
  ProxyEnvCfg& ctx_;
  uevent::UeventLoop* loop_;
};

class ProxyCliImpl : public std::enable_shared_from_this<ProxyCliImpl> {
 public:
  ProxyCliImpl(ProxyEnvCfg& context);
  void Initialize(uevent::UeventLoop* loop);

 private:
  void ConnectionSuccessCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionClosedCb(const uevent::ConnectionUeventPtr& conn);
  void RequestCb(const uevent::ConnectionUeventPtr& conn);

  MatrixPartition DecodeRequest(const void* payload, size_t size);
  void HandleMatMul(const uevent::ConnectionUeventPtr& conn,
                    MatrixPartition& partition);

  void SendPbRequest(const std::string& topic, const std::string& payload);
  void ReceivePbRequest(const std::string& topic,
                        const google::protobuf::Message& pb);

 private:
  void FillPartition(MatrixPartition& partition);
  void CacheTensor(const TensorKey& key, void* ptr, int64_t size,
                   int64_t h_dim);
  void SavePartition(MatrixPartition& partition);
  void HandlePartition(const uevent::ConnectionUeventPtr& conn,
                       const MatrixPartition& partition);
  void CheckCachedPartition(const uevent::ConnectionUeventPtr& conn);

 private:
  ProxyEnvCfg& ctx_;
  std::shared_ptr<uevent::ConnectorLibevent> connector_;

  // sw::redis::Redis* redis_;
  // std::unordered_map<std::string, uint64_t> device_info_;

  std::string uuid_;
  std::vector<MatrixPartition> cached_partitions_;
  std::unordered_set<PtrData> cached_msgs_;
  FixSizeLRUCache<TensorKey, torch::Tensor> cached_tensors_;
};

typedef std::shared_ptr<ProxyCliImpl> ProxyCliImplPtr;

class ProxyCli {
 public:
  class Status {
   public:
    enum StatusType {
      kOK = 0,
      kFatal,
      kUnknown,
    };

   public:
    Status() : status_(kOK), err_() {}
    const std::string& err() { return err_; }
    bool OK() { return status_ == kOK; }
    void SetError(StatusType status, const std::string& err) {
      status_ = status;
      if (status_str_.find(status) == status_str_.end()) status_ = kUnknown;
      err_ = status_str_.at(status_) + err_;
    }

   private:
    StatusType status_;
    std::string err_;
    static const std::map<StatusType, std::string> status_str_;
  };

 public:
  ProxyCli();
  void Initialize(const std::string& cfg_file);
  void Start();
  void Send(const torch::Tensor& tensor,
            std::optional<int64_t> rank = std::nullopt);
  void Receive(torch::Tensor& tensor,
               std::optional<int64_t> rank = std::nullopt);
  void AsyncSend(const torch::Tensor& tensor,
                 std::optional<int64_t> rank = std::nullopt);
  void AsyncReceive(torch::Tensor& tensor,
                    std::optional<int64_t> rank = std::nullopt);

 private:
  ProxyCliImplPtr svr_;
  ProxyEnvCfg context_;
  std::shared_ptr<uevent::UeventLoopThread> loop_thread_;
};
