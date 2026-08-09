#include "proxy_svr.h"

#include <cblas.h>

#include <chrono>

#include "core/generator.h"
#include "core/logger.h"
#include "core/stats.h"
#include "core/thread_affinity.h"
#include "device_measurement_session.h"
#include "device_tracker.h"
#include "global_api.pb.h"
#include "muduo_base/my_uuid.h"
#include "network/eventloop_libevent.h"
#include "network/listener_libevent.h"
#include "partition_tracker.h"
#include "proto_base.h"
#include "server_base.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <atomic>
#include <iostream>
#include <set>

namespace morphling {
namespace backend {

// Global atomic counter to assign each thread to a different CPU core
static std::atomic<int> g_thread_core_counter(0);

static void PinThreadToNextAvailableCore() {
  int num_cpus = morphling::GetOnlineCoreCount();
  int core_id =
      g_thread_core_counter.fetch_add(1, std::memory_order_relaxed) % num_cpus;
  morphling::PinThreadToCore(core_id);
}

/*********************************ProxySvrHandle***********************************/

ProxySvrHandle::ProxySvrHandle(ProxyEnvCfg& ctx, UeventLoop* loop)
    : ctx_(ctx),
      loop_(loop),
      measurement_(DeviceMeasurementService::Config{
          ctx.measure_latency,
          ctx.measure_bandwidth,
          ctx.measure_flops,
          static_cast<uint32_t>(ctx.measure_latency_payload_bytes),
          static_cast<uint32_t>(ctx.measure_bandwidth_payload_bytes),
          static_cast<uint32_t>(ctx.measure_flops_matrix_dim),
          ctx.measure_probe_timeout_sec,
          ctx.measure_flops_tolerance,
      }) {
  SRV_STATS->Initialize();
  LOG_INFO << "[ProxySvrHandle] device_measurement: lat="
           << measurement_.LatencyEnabled()
           << " bw=" << measurement_.BandwidthEnabled()
           << " flops=" << measurement_.FlopsEnabled();
}

void ProxySvrHandle::ThreadInit(uevent::UeventLoop* loop) {
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);

  // Pin this thread to the next available CPU core in round-robin fashion
  // This is called in the worker thread context to ensure correct thread
  // binding
  PinThreadToNextAvailableCore();
}

// Helper for zero-copy send cleanup of SerializationBuffer (registration msgs)
static void SerializationBufferSendCleanup(const void* /*data*/, size_t /*len*/,
                                           void* arg) {
  delete static_cast<SerializationBufferPtr*>(arg);
}

void ProxySvrHandle::RequestWriteCb(const uevent::ConnectionUeventPtr& conn) {
  size_t readable = conn->ReadableLength();
  LOG_DEBUG << "RequestWriteCb readable: " << readable;
}

void ProxySvrHandle::SendRegisterRequest(const ConnectionUeventPtr& conn) {
  LOG_INFO << "Sending registration request to "
           << conn->GetPeerAddress().ToString();

  DeviceRegisterRequest request;
  auto buffer = request.Serialize();

  LOG_DEBUG << "Raw registration request data (hex): "
            << BinaryToHex(static_cast<const uint8_t*>(buffer->GetBuffer()),
                           buffer->GetSize());

  // Zero-copy send: buffer ref-count prevents deallocation until libevent done
  auto* ref = new SerializationBufferPtr(buffer);
  int ret = conn->SendDataZeroCopy(buffer->GetBuffer(), buffer->GetSize(),
                                   SerializationBufferSendCleanup, ref);
  if (ret < 0) {
    LOG_ERROR << "Failed to send registration request";
    conn->ForceClose();
    return;
  }
}

void ProxySvrHandle::RequestCb(const ConnectionUeventPtr& conn) {
  while (true) {
    size_t readable = conn->ReadableLength();

    int ret = 0;
    uint32_t packsize = 0;
    ret = conn->ReceiveData(&packsize, sizeof(uint32_t));
    if (ret < 0) {
      LOG_ERROR << "ReceiveData packsize err";
      return;
    }
    packsize = ntohl(packsize);
    size_t datasize = packsize + sizeof(packsize);

    LOG_TRACE << "packsize: " << packsize << ", datasize: " << datasize
              << ", readable: " << readable;

    if (readable < datasize) {
      return;
    }

    // Zero-copy receive: get contiguous pointer into evbuffer
    unsigned char* raw_data = conn->PullupData(datasize);
    if (raw_data == nullptr) {
      LOG_ERROR << "PullupData failed for size " << datasize;
      return;
    }

    // Decode and dispatch message (processes data in-place)
    DecodeAndDispatch(conn, raw_data, datasize);

    // Drain after processing is complete
    ret = conn->DrainData(datasize);
    if (ret < 0) {
      LOG_ERROR << "DrainData err";
      return;
    }
  }
}

void ProxySvrHandle::DecodeAndDispatch(const ConnectionUeventPtr& conn,
                                       const void* payload, size_t size) {
  // Step 1: Decode proto message header to get message type
  int32_t message_type = GetMessageType(payload, size);

  if (message_type < 0) {
    LOG_ERROR << "Failed to decode message type";
    return;
  }

  // Step 2: Dispatch to appropriate handler based on message type
  string client_addr = conn->GetPeerAddress().ToString();

  switch (message_type) {
    case morphling::global_api::DEVICE_PROFILE_DATA:
      HandleRegisterResponse(conn, payload, size);
      break;

    case morphling::global_api::COMPUTE_GEMM_DATA:
      // Check if client is connected via tracker
      {
        auto& tracker = DEVICE_TRACKER;
        int64_t device_id = tracker.GetDeviceIdByAddr(client_addr);
        if (device_id == -1 || !tracker.IsDeviceConnected(device_id)) {
          LOG_ERROR << "Client " << client_addr
                    << " not registered or not connected, disconnecting";
          conn->ForceClose();
          return;
        }
      }
      HandleMatMul(conn, payload, size);
      conn_inflight_[client_addr] -= 1;

      if (!task_queue_.empty()) {
        auto task = task_queue_.front();
        task_queue_.pop_front();
        task();
      }
      loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, this));
      break;

    case morphling::global_api::PROBE_LATENCY_RESPONSE:
      HandleProbeLatencyResponse(conn, payload, size);
      break;

    case morphling::global_api::PROBE_BANDWIDTH_RESPONSE:
      HandleProbeBandwidthResponse(conn, payload, size);
      break;

    case morphling::global_api::PROBE_FLOPS_RESPONSE:
      HandleProbeFlopsResponse(conn, payload, size);
      break;

    default:
      LOG_ERROR << "Unknown message type: " << message_type;
      break;
  }
}

void ProxySvrHandle::HandleProbeLatencyResponse(const ConnectionUeventPtr& conn,
                                                const void* payload,
                                                size_t size) {
  string client_addr = conn->GetPeerAddress().ToString();
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(client_addr);
  auto session = DEVICE_TRACKER.GetMeasurementSession(device_id);
  if (!session) {
    LOG_WARN << "PROBE_LATENCY_RESPONSE from " << client_addr
             << " (device_id=" << device_id << ") without active session";
    return;
  }
  session->OnProbeLatencyResponse(payload, size);
}

void ProxySvrHandle::HandleProbeBandwidthResponse(
    const ConnectionUeventPtr& conn, const void* payload, size_t size) {
  string client_addr = conn->GetPeerAddress().ToString();
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(client_addr);
  auto session = DEVICE_TRACKER.GetMeasurementSession(device_id);
  if (!session) {
    LOG_WARN << "PROBE_BANDWIDTH_RESPONSE from " << client_addr
             << " (device_id=" << device_id << ") without active session";
    return;
  }
  session->OnProbeBandwidthResponse(payload, size);
}

void ProxySvrHandle::HandleProbeFlopsResponse(const ConnectionUeventPtr& conn,
                                              const void* payload,
                                              size_t size) {
  string client_addr = conn->GetPeerAddress().ToString();
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(client_addr);
  auto session = DEVICE_TRACKER.GetMeasurementSession(device_id);
  if (!session) {
    LOG_WARN << "PROBE_FLOPS_RESPONSE from " << client_addr
             << " (device_id=" << device_id << ") without active session";
    return;
  }
  session->OnProbeFlopsResponse(payload, size);
}

void ProxySvrHandle::HandleRegisterResponse(const ConnectionUeventPtr& conn,
                                            const void* payload, size_t size) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_DEBUG << "Received device profile data from " << client_addr
            << ", size=" << size << ", hex: "
            << BinaryToHex(static_cast<const uint8_t*>(payload), size) << "";

  // Use standard Deserialize interface
  DeviceProfileData profile;
  profile.Deserialize(payload, size);

  // Register device in tracker
  int64_t device_id = DEVICE_TRACKER.RegisterDevice(client_addr, profile);

  // Store connection in tracker
  DEVICE_TRACKER.SetDeviceConnection(device_id, conn);

  LOG_INFO << "Client " << client_addr
           << " registered with device_id=" << device_id
           << ", reported flops=" << profile.flops << " ul_bw=" << profile.ul_bw
           << " dl_bw=" << profile.dl_bw;
  LOG_DEBUG << "Profile: " << profile.DebugString();

  StartMeasurementOrDispatch(conn, device_id);
}

void ProxySvrHandle::StartMeasurementOrDispatch(const ConnectionUeventPtr& conn,
                                                int64_t device_id) {
  if (!measurement_.AnyEnabled()) {
    loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, this));
    return;
  }

  DeviceMeasurementSession::Config cfg;
  cfg.enable_latency = measurement_.LatencyEnabled();
  cfg.enable_bandwidth = measurement_.BandwidthEnabled();
  cfg.enable_flops = measurement_.FlopsEnabled();
  cfg.latency_payload_bytes = measurement_.LatencyPayloadBytes();
  cfg.bandwidth_payload_bytes = measurement_.BandwidthPayloadBytes();
  cfg.flops_matrix_dim = measurement_.FlopsMatrixDim();
  cfg.probe_timeout_sec = measurement_.ProbeTimeoutSec();
  cfg.flops_tolerance = measurement_.FlopsTolerance();
  // Seed from the just-registered profile so verification matches what the
  // codec stamped on the wire.
  cfg.seed = DEVICE_TRACKER.GetDeviceLiveness(device_id).profile.uuid;

  // Wire send: zero-copy with keepalive that pins the SerializationBuffer
  // until libevent drains.
  SendFn send_fn = [conn](SerializationBufferPtr buf) -> bool {
    auto* keepalive = new SerializationBufferPtr(buf);
    int rc = conn->SendDataZeroCopy(buf->GetBuffer(), buf->GetSize(),
                                    SerializationBufferSendCleanup, keepalive);
    if (rc < 0) {
      delete keepalive;
      return false;
    }
    return true;
  };

  // Per-probe timeout via libevent's RunAfter/CancelTimer. uevent::TimerId is
  // a value type; stash it on the heap so callbacks can carry it through the
  // uint64_t opaque-id slot.
  UeventLoop* loop = conn->GetLoop();
  TimerOps timer_ops;
  timer_ops.Arm = [loop](double seconds, std::function<void()> cb) -> uint64_t {
    auto* id = new uevent::TimerId(loop->RunAfter(seconds, std::move(cb)));
    return reinterpret_cast<uint64_t>(id);
  };
  timer_ops.Cancel = [loop](uint64_t opaque) {
    if (opaque == 0) return;
    auto* id = reinterpret_cast<uevent::TimerId*>(opaque);
    loop->CancelTimer(*id);
    delete id;
  };

  // Reference GEMM: MKL sgemm on the libevent worker thread. 256x256x256
  // f32 is ~5 ms on a single core; acceptable inline. For larger probe
  // matrices we would need to offload, but cfg.flops_matrix_dim defaults
  // to 256 and is bounded by the operator.
  ReferenceGemmFn ref_gemm = [](const ReferenceGemmRequest& req,
                                ReferenceGemmDoneFn done) {
    auto c = std::make_shared<std::vector<float>>(static_cast<size_t>(req.m) *
                                                  req.n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, req.m, req.n, req.k,
                1.0f, req.a, req.k, req.b, req.n, 0.0f, c->data(), req.n);
    done(true, c);
  };

  auto session = std::make_shared<DeviceMeasurementSession>(
      device_id, cfg, std::move(send_fn), std::move(timer_ops),
      std::move(ref_gemm));
  DEVICE_TRACKER.SetMeasurementSession(device_id, session);

  auto* self = this;
  session->Start([self, device_id](const MeasurementResult& r) {
    LOG_INFO << "Device " << device_id << " measurement complete: ok=" << r.ok
             << " state=" << MeasurementStateName(r.terminal_state)
             << " measured_lat_ns=" << r.measured_profile.measured_lat_ns
             << " measured_ul_bw_bps=" << r.measured_profile.measured_ul_bw_bps
             << " measured_dl_bw_bps=" << r.measured_profile.measured_dl_bw_bps
             << " measured_flops=" << r.measured_profile.measured_flops
             << " verified=" << r.measured_profile.measured_flops_verified;
    DEVICE_TRACKER.UpdateMeasuredProfile(device_id, r.measured_profile);
    DEVICE_TRACKER.RemoveMeasurementSession(device_id);
    self->loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, self));
  });
}

void ProxySvrHandle::HandleMatMul(const ConnectionUeventPtr& conn,
                                  const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();

  // Record RECEIVE/DOWNLOAD start time (virtual time)
  uint64_t vt_receive_start = VirtualClockNow();

  // Use standard Deserialize interface
  MatrixPartition partition;
  partition.Deserialize(payload, size);

  // Log RECEIVE START after getting device_id from partition
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "RECEIVE", "START", vt_receive_start,
                                     vt_receive_start);

  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << part_key << " RSP Deserialization time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  // Record bytes received (download request from device)
  DEVICE_TRACKER.RecordBytesReceived(partition.dev_id, size);

  // Log download throughput after receiving response
  double download_tp = DEVICE_TRACKER.GetDownloadThroughput(partition.dev_id);
  double last_packet_tp =
      DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
  double avg_packet_tp =
      DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
  double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();

  uint64_t start_us, end_us;
  DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us,
                                              end_us);

  LOG_INFO << "[HandleMatMul] Device " << partition.dev_id
           << " - Received: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Download TP: " << download_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";

  // Record RECEIVE end time (virtual time)
  uint64_t vt_receive_end = VirtualClockNow();

  // Log virtual time event for RECEIVE
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "RECEIVE", "END", vt_receive_start,
                                     vt_receive_end);

  // Log throughput to file
  // DEVICE_TRACKER.LogThroughputToFile(partition.dev_id, partition.gemm_id,
  // "DOWNLOAD",
  //                                    size, download_tp, start_us, end_us);

  auto [o_ptr, o_size] = partition.mat[0];
  int64_t row_size = o_size / partition.h_dim / sizeof(float);
  int64_t col_size = partition.h_dim;

  uint64_t ul_overhead = CurrentTimeMicros() - partition.timestamp;

  LOG_DEBUG << part_key << " partition: " << partition.DebugString();

  start = std::chrono::high_resolution_clock::now();
  auto output = torch::from_blob(o_ptr, {row_size, col_size},
                                 FLOAT32_TENSOR_OPTIONS(torch::kCPU));
  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->WriteResultBlock(partition.oid, output, partition.row, partition.col,
                         partition.pivot, ctx_.block_size);
  end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << "UpdateMatrixBlock time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "RECEIVE", "END", vt_receive_start,
                                     vt_receive_end);

  // Synthetic VTIME phases derived from device profile (FLOPS, bandwidth)
  {
    auto sm_it = send_meta_.find(part_key);
    auto liveness = DEVICE_TRACKER.GetDeviceLiveness(partition.dev_id);
    if (sm_it != send_meta_.end() && liveness.device_id >= 0) {
      const auto& sm = sm_it->second;
      const auto& prof = liveness.profile;

      double dl_bw = prof.dl_bw > 0 ? prof.dl_bw : 1.0;
      double ul_bw = prof.ul_bw > 0 ? prof.ul_bw : 1.0;
      double flops = prof.flops > 0 ? prof.flops : 1.0;

      uint64_t dl_dur_us =
          static_cast<uint64_t>(sm.send_bytes / dl_bw * 1e6 + prof.dl_lat);
      uint64_t comp_dur_us =
          static_cast<uint64_t>(2.0 * sm.m * sm.n * sm.h_dim / flops * 1e6);
      uint64_t ul_dur_us =
          static_cast<uint64_t>(size / ul_bw * 1e6 + prof.ul_lat);

      uint64_t vt_dl_start = sm.vt_send_end_us;
      uint64_t vt_dl_end = vt_dl_start + dl_dur_us;
      uint64_t vt_comp_start = vt_dl_end;
      uint64_t vt_comp_end = vt_comp_start + comp_dur_us;
      uint64_t vt_ul_start = vt_comp_end;
      uint64_t vt_ul_end = vt_ul_start + ul_dur_us;

      DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                         "DOWNLOAD", "START", vt_dl_start,
                                         vt_dl_start);
      DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                         "DOWNLOAD", "END", vt_dl_start,
                                         vt_dl_end);
      DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                         "COMPUTE", "START", vt_comp_start,
                                         vt_comp_start);
      DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                         "COMPUTE", "END", vt_comp_start,
                                         vt_comp_end);
      DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                         "UPLOAD", "START", vt_ul_start,
                                         vt_ul_start);
      DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                         "UPLOAD", "END", vt_ul_start,
                                         vt_ul_end);

      send_meta_.erase(sm_it);
    }
  }

  PARTITION_TRACKER.MarkPartitionFinished(part_key);
  PARTITION_TRACKER.RemovePartitionByKey(part_key);
  DEVICE_TRACKER.RecordPartitionProcessed(partition.dev_id);

  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->IncRspCbCount(partition.oid, 1);
}

// Helper for zero-copy send cleanup: releases ScatterGatherBuffer and
// MatrixPartition when libevent is done sending a segment.
struct ZeroCopySendContext {
  ScatterGatherBufferPtr sg_buffer;
  MatrixPartitionPtr partition;
};

static void ZeroCopySendCleanup(const void* /*data*/, size_t /*len*/,
                                void* arg) {
  // shared_ptr ref count decrements; when last segment cleanup fires, the
  // ZeroCopySendContext (and thus sg_buffer + partition) are freed.
  delete static_cast<std::shared_ptr<ZeroCopySendContext>*>(arg);
}

void ProxySvrHandle::SendInLoop(const ConnectionUeventPtr& conn,
                                const MatrixPartitionPtr partition) {
  // check connection valid first
  if (conn->IsClosed()) {
    LOG_ERROR << "Connection to " << conn->GetPeerAddress().ToString()
              << " is not valid. Cannot send partition.";
    return;
  }

  string client_addr = conn->GetPeerAddress().ToString();

  task_queue_.push_back([this, conn, partition, client_addr]() {
    // Record SEND start time (virtual time)
    uint64_t vt_send_start = VirtualClockNow();
    DEVICE_TRACKER.LogVirtualTimeEvent(partition->dev_id, partition->gemm_id,
                                       "SEND", "START", vt_send_start,
                                       vt_send_start);

    // Zero-copy scatter-gather serialization (avoids tensor memcpy)
    auto t_serialize_start = std::chrono::high_resolution_clock::now();
    auto sg_buffer = partition->SerializeZeroCopy();
    auto t_serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            t_serialize_end - t_serialize_start)
                            .count();

    size_t size = sg_buffer->GetTotalSize();

    // Create shared context to keep scatter-gather buffer and partition alive
    // until all segments are sent
    auto ctx = std::make_shared<ZeroCopySendContext>();
    ctx->sg_buffer = sg_buffer;
    ctx->partition = partition;

    // Zero-copy send each segment
    auto t_send_start = std::chrono::high_resolution_clock::now();
    for (const auto& segment : sg_buffer->GetSegments()) {
      // Each cleanup callback holds a shared_ptr copy of the context
      auto* ref = new std::shared_ptr<ZeroCopySendContext>(ctx);
      conn->SendDataZeroCopy(segment.data, segment.size, ZeroCopySendCleanup,
                             ref);
    }
    auto t_send_end = std::chrono::high_resolution_clock::now();
    auto send_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       t_send_end - t_send_start)
                       .count();

    conn_inflight_[client_addr] += 1;

    double actual_send_tp_bs =
        (send_us > 0) ? (size * 1000000.0 / send_us) : 0.0;
    double actual_send_tp_gbs = actual_send_tp_bs / (1024.0 * 1024.0 * 1024.0);
    LOG_INFO << "[SendInLoop-Timing] Device " << partition->dev_id
             << ", gemm_id=" << partition->gemm_id << ", Size: " << size
             << " bytes"
             << " | Serialize(ZC): " << serialize_us << " us"
             << ", SendData(ZC): " << send_us << " us"
             << ", Actual TP: " << actual_send_tp_gbs << " GB/s";

    // Record SEND end time (virtual time)
    uint64_t vt_send_end = VirtualClockNow();
    DEVICE_TRACKER.LogVirtualTimeEvent(partition->dev_id, partition->gemm_id,
                                       "SEND", "END", vt_send_start,
                                       vt_send_end);

    {
      auto pkey = partition->GetPartitionKey();
      SendMeta sm;
      sm.vt_send_end_us = vt_send_end;
      sm.send_bytes = size;
      sm.m = 0;
      sm.n = 0;
      sm.h_dim = partition->h_dim;
      for (const auto& m : partition->mat) {
        auto mat_bytes = std::get<1>(m);
        int64_t dim = mat_bytes / partition->h_dim / sizeof(float);
        if (sm.m == 0)
          sm.m = dim;
        else
          sm.n = dim;
      }
      send_meta_[pkey] = sm;
    }

    DEVICE_TRACKER.RecordBytesSent(partition->dev_id, size);

    double last_packet_tp =
        DEVICE_TRACKER.GetLastPacketThroughput(partition->dev_id);
    uint64_t start_us, end_us;
    DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition->dev_id, start_us,
                                                end_us);
    DEVICE_TRACKER.LogThroughputToFile(partition->dev_id, partition->gemm_id,
                                       "SEND", size, last_packet_tp, start_us,
                                       end_us);
  });

  while (!task_queue_.empty()) {
    auto task = task_queue_.front();
    task_queue_.pop_front();
    task();
  }
}

void ProxySvrHandle::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "connected from " << client_addr;
  conn_inflight_[client_addr] = 0;

  // Send registration request to client
  SendRegisterRequest(conn);
}

void ProxySvrHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected from " << client_addr;
  conn_inflight_.erase(client_addr);
  // device_info_.erase(client_addr);

  std::string conn_addr = conn->GetPeerAddress().ToString();

  // Find device ID by connection address
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(conn_addr);

  LOG_INFO << "[ConnectionClosedCb] Device " << device_id
           << " (addr: " << conn_addr << ") disconnected";

  // Step 1: Check if failed device has pending partitions
  bool has_pending_partitions =
      PARTITION_TRACKER.HasPendingPartitions(device_id);
  size_t pending_count = PARTITION_TRACKER.GetDevicePartitionCount(device_id);

  if (has_pending_partitions) {
    LOG_WARN << "[ConnectionClosedCb] Device " << device_id << " failed with "
             << pending_count << " pending partitions";
  }

  // Step 2: Mark all running partitions as failed
  PARTITION_TRACKER.MarkDevicePartitionsFailed(device_id);

  // Step 3: Remove connection from maps
  // conn_map_.erase(conn_addr);
  if (device_id != -1) {
    DEVICE_TRACKER.RemoveMeasurementSession(device_id);
    DEVICE_TRACKER.RemoveDeviceConnection(device_id);
    DEVICE_TRACKER.UnregisterDevice(device_id);
  }
}

/********************************ProxySvrImpl****************************************/

ProxySvrImpl::ProxySvrImpl(ProxyEnvCfg& ctx)
    : ctx_(ctx),
      listener_(nullptr),
      rsp_cb_counts_(kMaxLifetimeOperationCount) {
  // Initialize with greedy scheduling policy by default
}

ProxySvrImpl::~ProxySvrImpl() {
  LOG_INFO << "[ProxySvrImpl::~ProxySvrImpl] Shutting down ProxySvrImpl";
  loop_->CancelTimer(failed_partition_check_timer_);
}

void ProxySvrImpl::Initialize(UeventLoop* loop) {
  LOG_INFO << "[ProxySvrImpl::Initialize] Starting server initialization";
  LOG_INFO << "[ProxySvrImpl::Initialize] Config - listen_ip=" << ctx_.listen_ip
           << ", listen_port=" << ctx_.listen_port;
  LOG_INFO << "[ProxySvrImpl::Initialize] Config - num_device="
           << ctx_.num_device << ", thread=" << ctx_.thread;

  // Initialize virtual clock
  base::VirtualClock::instance().Initialize();
  LOG_INFO << "[ProxySvrImpl::Initialize] Virtual clock initialized";

  // Initialize performance logging (server side)
  DEVICE_TRACKER.InitSeparatePerfLog("./logs", "server");
  LOG_INFO << "[ProxySvrImpl::Initialize] Performance logging initialized at "
              "./logs/perf_server.log";

  loop_ = loop;

  auto create_handle_cb = bind(ProxySvrHandle::CreateMyself, ref(ctx_), _1);
  UsockAddress addr(ctx_.listen_ip, ctx_.listen_port);
  listener_ =
      make_shared<ListenerLibevent>(loop, addr, "proxy_listener", Option());
  listener_->SetCreateLoopHandleCb(create_handle_cb);
  listener_->SetThreadInitCb(ProxySvrHandle::ThreadInit);
  listener_->SetConnectionSuccessCb(
      bind(&ProxySvrImpl::ConnectionSuccessCb, shared_from_this(), _1));
  listener_->SetMessageReadCb(
      bind(&ProxySvrImpl::RequestCb, shared_from_this(), _1));
  listener_->SetConnectionClosedCb(
      bind(&ProxySvrImpl::ConnectionClosedCb, shared_from_this(), _1));
  listener_->SetMessageWriteCb([](const ConnectionUeventPtr& conn) {});
  listener_->SetThreadNum(ctx_.thread);
  listener_->StartPrimaryLoop();

  LOG_INFO << "[ProxySvrImpl::Initialize] ProxySvrImpl listen on:"
           << ctx_.listen_ip << ":" << ctx_.listen_port;

  ctx_.instance = this;

  // Start();
  // InitLogger();

  outputs_.resize(kMaxLifetimeOperationCount);
  for (size_t i = 0; i < outputs_.size(); i++) {
    outputs_[i] = torch::empty({0, 0});
    rsp_cb_counts_[i] = 0;
  }

  LOG_INFO << "[ProxySvrImpl::Initialize] Server initialization completed. "
              "Waiting for connections...";

  // Start periodic partition health check (every 0.1 seconds)
  // failed_partition_check_timer_ = loop->RunEvery(
  //     0.1, std::bind(&ProxySvrImpl::CheckFailedPartitions, this));
  // idle_partition_redistribute_timer_ =
  //     loop->RunEvery(0.5, std::bind(&ProxySvrImpl::SendIdlePartitions,
  //     this));
  LOG_INFO << "[ProxySvrImpl::Initialize] Started periodic partition health "
              "check (interval=0.1s)";
}

void ProxySvrImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);
  // loop->RunInLoop(bind(&ProxySvrHandle::ConnectionSuccessCb, handle, conn));
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));
}

void ProxySvrImpl::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionClosedCb(conn);
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));

  // Unregister device from tracker
  string client_addr = conn->GetPeerAddress().ToString();
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(client_addr);
  if (device_id != -1) {
    DEVICE_TRACKER.UnregisterDevice(device_id);
  }

  // loop->RunInLoop(bind(&ProxySvrHandle::ConnectionClosedCb, handle, conn));

  // LOG_INFO << "[ConnectionClosedCb] Connection removed. Remaining
  // connections: "
  //          << conn_map_.size();
  // LOG_INFO << "[ConnectionClosedCb] Connection map contents:";
  // for (const auto& conn_pair : conn_map_) {
  //   LOG_INFO << "  - " << conn_pair.first << " -> "
  //            << (conn_pair.second ? "valid" : "null");
  // }

  // // Step 4: Handle partition redistribution if needed
  // if (has_pending_partitions && conn_map_.size() > 0) {
  //   LOG_INFO << "[ConnectionClosedCb] Starting partition redistribution for "
  //               "failed device "
  //            << device_id;
  //   HandleDeviceFailure(device_id);
  //   LOG_INFO << "[ConnectionClosedCb] Partition redistribution completed";
  // } else if (has_pending_partitions && conn_map_.empty()) {
  //   LOG_ERROR << "[ConnectionClosedCb] Device " << device_id << " failed with
  //   "
  //             << pending_count
  //             << " pending partitions but no other devices available!";
  // }
}

void ProxySvrImpl::RequestWriteCb(const uevent::ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->RequestWriteCb(conn);
}

void ProxySvrImpl::RequestCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  loop->RunInLoop(bind(&ProxySvrHandle::RequestCb, handle, conn));
}

// void ProxySvrImpl::RequestCb(const uevent::ConnectionUeventPtr& conn) {
//   size_t readable = conn->ReadableLength();

//   int ret = 0;
//   size_t packsize;
//   ret = conn->ReceiveData(&packsize, sizeof(size_t));
//   if (ret < 0) {
//     conn->ForceClose();
//     LOG_ERROR << "ReceiveData on a closed connection";
//     return;
//   }

//   auto* loop = conn->GetLoop();
//   loop->AssertInLoopThread();
//   auto* loop_handle = loop->GetLoopHandle();
//   auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
//   handle->RequestCb(conn);
// }

int ProxySvrImpl::DispatchMatMulAsync(torch::Tensor& mat_a,
                                      torch::Tensor& mat_b) {
  auto* gate = DEVICE_TRACKER.GetDispatchGate();
  if (gate != nullptr) {
    if (gate->GetMode() == DeviceMode::BARRIER) {
      if (!gate->WaitForReady()) {
        LOG_ERROR << "[DispatchMatMulAsync] DispatchGate WaitForReady timeout";
        return -1;
      }
    } else if (gate->GetMode() == DeviceMode::DYNAMIC &&
               DEVICE_TRACKER.GetConnectedDeviceCount() == 0) {
      auto mat_a_clone = mat_a.clone();
      auto mat_b_clone = mat_b.clone();
      gate->EnqueueWork([this, mat_a_clone, mat_b_clone]() mutable {
        auto queued_a = mat_a_clone;
        auto queued_b = mat_b_clone;
        this->DispatchMatMulAsync(queued_a, queued_b);
      });
      LOG_INFO << "[DispatchMatMulAsync] No connected devices in DYNAMIC mode, "
                  "work enqueued";
      return -1;
    }
  }

  int oid = ReserveOperationId(mm_count_);
  if (oid < 0) {
    LOG_ERROR << "[DispatchMatMulAsync] Operation capacity exhausted at "
              << kMaxLifetimeOperationCount << " lifetime dispatches";
    return -1;
  }
  LOG_INFO << "[DispatchMatMulAsync] Starting dispatch - oid=" << oid;

  int gemm_id = gemm_id_count_.fetch_add(1);
  auto partitions = PartitionMatrices(mat_a, mat_b, ctx_.block_size);
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  if (partitions.empty()) {
    LOG_ERROR << "[DispatchMatMulAsync] No partitions generated!";
    return -1;
  }

  auto cur_ver = partitions[0]->version;
  LOG_INFO << "[" << cur_ver << "] Number of partitions: " << partitions.size()
           << " for A: " << a_shape << " and B: " << b_shape;

  LOG_INFO << "[DispatchMatMulAsync] Before random_shuffle - partitions.size()="
           << partitions.size();
  std::random_shuffle(partitions.begin(), partitions.end());

  // LOG_INFO
  //     << "[DispatchMatMulAsync] Before RephrasePartitions - ctx_.num_device="
  //     << ctx_.num_device;

  // RephrasePartitions(partitions);

  // LOG_INFO << "[DispatchMatMulAsync] After RephrasePartitions - created "
  //          << partitions.size() << " partitions";
  auto start = std::chrono::high_resolution_clock::now();

  {
    std::lock_guard<std::mutex> lk(outputs_mutex_);
    outputs_[oid].set_data(CreateOutputMatrix(mat_a, mat_b));
    DecRspCbCount(oid, partitions.size());
  }

  LOG_INFO << "[DispatchMatMulAsync] Creating " << partitions.size()
           << " partitions as IDLE";

  // Add all partitions to tracker as IDLE - they will be dispatched by
  // SendIdlePartitions
  for (auto& partition : partitions) {
    partition->oid = oid;
    partition->gemm_id = gemm_id;  // assign global gemm_id
    partition->dev_id =
        -1;  // Mark as unassigned, will be assigned by scheduling policy

    // Add partition to tracker with dev_id=-1 (unassigned, to be scheduled)
    // The tracker will use owner_device_id=-1 until scheduling assigns a real
    // device
    PARTITION_TRACKER.AddPartition(-1, partition->GetPartitionKey(), oid,
                                   partition);

    LOG_DEBUG << "[DispatchMatMulAsync] Created IDLE partition key="
              << partition->GetPartitionKey()
              << ", dev_id=" << partition->dev_id << " (unassigned)"
              << ", oid=" << oid << ", gemm_id=" << partition->gemm_id;
  }
  auto end = std::chrono::high_resolution_clock::now();
  LOG_INFO << "[DispatchMatMulAsync] Created " << partitions.size()
           << " IDLE partitions in "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count()
           << "us. Partitions will be sent by SendIdlePartitions timer. "
              "gemm_id="
           << gemm_id;

  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_->GetLoopHandle());
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));
  return oid;
}

torch::Tensor ProxySvrImpl::WaitMatMul(int oid) {
  ValidateOperationId(oid);
  std::unique_lock<std::mutex> lk(outputs_mutex_);
  while (!outputs_cv_.wait_for(lk, std::chrono::seconds(5),
                               [&] { return rsp_cb_counts_[oid] == 0; })) {
    LOG_WARN << "[WaitMatMul] Still waiting for oid=" << oid
             << ", remaining=" << rsp_cb_counts_[oid];
  }
  return outputs_[oid].clone();
}

void ProxySvrImpl::WriteResultBlock(int oid, torch::Tensor& block, int64_t row,
                                    int64_t col, int64_t pivot,
                                    int block_size) {
  std::lock_guard<std::mutex> lock(outputs_mutex_);
  IndexPutMatrixBlock(outputs_[oid], block, row, col, pivot, block_size);
}

void ProxySvrImpl::IncRspCbCount(int oid, size_t count) {
  std::lock_guard<std::mutex> lock(outputs_mutex_);
  uint64_t prev = rsp_cb_counts_[oid];
  if (prev < count) {
    LOG_WARN << "[IncRspCbCount] Clamping underflow for oid=" << oid
             << ", current=" << prev << ", decrement=" << count;
    rsp_cb_counts_[oid] = 0;
  } else {
    rsp_cb_counts_[oid] -= count;
  }
  if (rsp_cb_counts_[oid] == 0) {
    outputs_cv_.notify_all();
  }
  LOG_DEBUG << "[IncRspCbCount] oid=" << oid << ", count=" << count
            << ", prev=" << prev << ", now=" << rsp_cb_counts_[oid];
}

void ProxySvrImpl::RephrasePartitions(
    std::vector<MatrixPartitionPtr>& partitions,
    const std::unordered_set<int64_t>& excluded_devices) {
  // Get connected devices from tracker
  auto& tracker = DEVICE_TRACKER;
  std::vector<int64_t> device_ids = tracker.GetConnectedDevices();

  int actual_num_devices = static_cast<int>(device_ids.size());
  LOG_INFO << "[RephrasePartitions] Starting with " << partitions.size()
           << " partitions, actual connected devices=" << actual_num_devices
           << ", excluded_devices=" << excluded_devices.size();

  if (actual_num_devices == 0) {
    LOG_ERROR << "[RephrasePartitions] No devices connected!";
    return;
  }

  std::sort(device_ids.begin(), device_ids.end());

  auto assignments = ctx_.sched_policy->AssignPartitionsToDevices(
      partitions, excluded_devices);

  if (assignments.size() != partitions.size()) {
    LOG_ERROR << "[RephrasePartitions] Policy returned " << assignments.size()
              << " assignments for " << partitions.size() << " partitions";
    return;
  }

  // Apply assignments to partitions
  for (size_t i = 0; i < partitions.size(); ++i) {
    partitions[i]->dev_id = assignments[i];
    LOG_DEBUG << "[RephrasePartitions] Partition " << i
              << " assigned to device_id " << assignments[i];
  }

  LOG_INFO << "[RephrasePartitions] Completed partitioning";
}

void ProxySvrImpl::HandleDeviceFailure(int64_t failed_device_id) {
  // Get partitions before redistribution to count OIDs
  auto failed_partitions =
      PARTITION_TRACKER.GetDevicePartitions(failed_device_id);

  if (failed_partitions.empty()) {
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " has no pending partitions to redistribute";
    return;
  }

  size_t num_failed_partitions = 0;
  std::unordered_map<int64_t, size_t> oid_counts;
  for (const auto& part : failed_partitions) {
    if (part->state == PartitionState::RUNNING) {
      num_failed_partitions++;
      oid_counts[part->oid]++;
    }
  }

  if (num_failed_partitions == 0) {
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " has no FAILED partitions to redistribute";
    return;
  }

  LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
           << " failed with " << num_failed_partitions
           << " partitions. Redistributing across all connected devices";

  LOG_INFO << "[HandleDeviceFailure] OID breakdown for failed partitions:";
  for (const auto& [oid, count] : oid_counts) {
    LOG_INFO << "  - OID " << oid << ": " << count << " partitions";
  }

  // Redistribute partitions across all connected devices
  // PARTITION_TRACKER.RedistributeFailedDevicePartitions(failed_device_id);

  // Get connected devices to send redistributed partitions to them
  std::vector<int64_t> connected_devices = DEVICE_TRACKER.GetConnectedDevices();

  // CRITICAL: Decrement response counters for partitions from failed device
  // These partitions were in-flight when the device failed, so they will never
  // produce responses. We must decrement their response counters to prevent
  // WaitMatMul from hanging forever.
  for (const auto& [oid, count] : oid_counts) {
    LOG_INFO << "[HandleDeviceFailure] Decrementing response counter for OID "
             << oid << " by " << count << " (in-flight partitions lost)";
    for (size_t i = 0; i < count; ++i) {
      IncRspCbCount(oid, 1);  // Decrement the counter
    }
  }

  LOG_INFO << "[HandleDeviceFailure] Completed failure handling for device "
           << failed_device_id;
}

void ProxySvrHandle::SendIdlePartitions() {
  auto idle_partitions = PARTITION_TRACKER.ClaimIdlePartitions();

  if (idle_partitions.empty()) {
    LOG_DEBUG << "[SendIdlePartitions] No IDLE partitions to send";
    return;
  }

  LOG_INFO << "[SendIdlePartitions] Found " << idle_partitions.size()
           << " IDLE partitions, running scheduling policy";

  auto redistributed =
      ctx_.sched_policy->RedistributePartitions(idle_partitions);

  if (redistributed.empty()) {
    LOG_DEBUG << "[SendIdlePartitions] No available devices for redistribution";
    for (const auto& part_info : idle_partitions) {
      PARTITION_TRACKER.MarkPartitionIdle(part_info->key);
    }
    return;
  }

  LOG_INFO << "[SendIdlePartitions] Scheduling complete, moving partitions to "
              "assigned devices";

  std::vector<PartitionInfoPtr> ready_partitions;
  ready_partitions.reserve(idle_partitions.size());
  for (const auto& part_info : idle_partitions) {
    auto assignment = redistributed.find(part_info->key);
    if (assignment == redistributed.end() ||
        !PARTITION_TRACKER.ReassignPartitionToDevice(part_info->key,
                                                     assignment->second)) {
      LOG_ERROR << "[SendIdlePartitions] Failed to reassign partition "
                << part_info->key;
      PARTITION_TRACKER.MarkPartitionIdle(part_info->key);
      continue;
    }
    ready_partitions.push_back(part_info);
  }

  LOG_INFO << "[SendIdlePartitions] Sending " << ready_partitions.size()
           << " partitions to devices";

  // Send each partition
  for (const auto& part_info : ready_partitions) {
    LOG_DEBUG << "[SendIdlePartitions] Sending partition " << part_info->key
              << " to device " << part_info->owner_device_id;

    auto target_conn =
        DEVICE_TRACKER.GetDeviceConnection(part_info->owner_device_id);
    if (!target_conn) {
      LOG_ERROR << "[SendIdlePartitions] No connection for device "
                << part_info->owner_device_id;
      // Mark as IDLE again so it can be rescheduled
      PARTITION_TRACKER.MarkPartitionIdle(part_info->key);
      continue;
    }
    auto* loop = target_conn->GetLoop();
    auto* handle = reinterpret_cast<ProxySvrHandle*>(loop->GetLoopHandle());
    loop->RunInLoop(bind(&ProxySvrHandle::SendInLoop, handle, target_conn,
                         part_info->partition));
  }

  LOG_INFO << "[SendIdlePartitions] Completed sending "
           << ready_partitions.size() << " partitions to devices";
}

void ProxySvrImpl::CheckFailedPartitions() {
  auto& tracker = DEVICE_TRACKER;

  // Get all devices
  std::vector<int64_t> all_devices = tracker.GetAllDevices();

  LOG_DEBUG << "[CheckFailedPartitions] Checking " << all_devices.size()
            << " devices for failed partitions";

  for (int64_t device_id : all_devices) {
    // Skip connected devices
    if (tracker.IsDeviceConnected(device_id)) {
      continue;
    }

    // Check if disconnected device has pending partitions
    if (PARTITION_TRACKER.HasPendingPartitions(device_id)) {
      size_t pending_count =
          PARTITION_TRACKER.GetDevicePartitionCount(device_id);
      LOG_WARN << "[CheckFailedPartitions] Detected disconnected device "
               << device_id << " with " << pending_count
               << " pending partitions. Initiating failure handling.";

      HandleDeviceFailure(device_id);
    }
  }
}

/*********************************ProxySvr***************************************/
typedef ProxySvr::Status ProxyStatus;
typedef ProxyStatus::StatusType ProxyStatusType;
const map<ProxyStatusType, string> ProxyStatus::status_str_ = {
    {kOK, ""},
    {kFatal, "Proxy svr fatal:"},
    {kUnknown, "Proxy svr unknown:"},
};

ProxySvr::ProxySvr() : svr_(nullptr), loop_thread_(nullptr) {}

void ProxySvr::Initialize(const std::string& cfg_file) {
  context_.Initialize(cfg_file);

  const int64_t barrier_count =
      context_.barrier_count > 0 ? context_.barrier_count : context_.num_device;
  DEVICE_TRACKER.InitDispatchGate(context_.device_mode, barrier_count,
                                  context_.barrier_timeout_ms,
                                  context_.max_queue_size);

  svr_ = make_shared<ProxySvrImpl>(context_);
  loop_thread_ = make_shared<UeventLoopThread>(
      bind(ProxySvrHandle::CreateMyself, ref(context_), _1),
      bind(&ProxySvrImpl::Initialize, svr_, _1), "Proxy svr main thread");
}

void ProxySvr::Start() { loop_thread_->StartLoop(); }

}  // namespace backend
}  // namespace morphling
