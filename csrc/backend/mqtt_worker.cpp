#include "mqtt_worker.h"

#include <torch/torch.h>

#include <iostream>

#include "common/pytorch_defs.h"
#include "server_base.h"
#include "utils/logger.h"

void MQTTWorker::OnMessage(struct mosquitto* mosq, void* obj,
                           const struct mosquitto_message* message) {
  auto topic = std::string(message->topic);
  if (topic.find(MQTT_COMPUTE_TOPIC_REQ) != std::string::npos) {
    HandleMatMul(message);
  }

  if (topic.find(MQTT_TIMER_TOPIC_REQ) != std::string::npos) {
    HandleTimer(message);
  }
}

void MQTTWorker::HandleTimer(const struct mosquitto_message* message) {}

void MQTTWorker::HandleMatMul(const struct mosquitto_message* message) {
  // std::cerr << "Handling matmul" << std::endl;
  at::InferenceMode infer_guard(true);
  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  partition.Deserialize(message->payload, message->payloadlen);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG("REQ Deserialization time: {}us", duration.count());

  // get logical time from redis
  auto time = redis_->hget(uuid_, "logical_time");
  LOG_FATAL_IF(!time, "Failed to get logical time from redis, key: {}", uuid_);

  logical_time_ = std::stoull(*time);
  logical_time_ += (double)message->payloadlen / device_info_["dl_bw"] *
                   1e6;  // in microseconds

  // create tensor from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  assert(row_size * partition.h_dim * sizeof(float) == r_size);
  // fprintf(stderr, "Row size: %ld, r_size: %ld\n", row_size, r_size);
  assert(col_size * partition.h_dim * sizeof(float) == c_size);
  // fprintf(stderr, "Col size: %ld, c_size: %ld\n", col_size, c_size);

  // std::cerr << "Row size: " << row_size << ", Col size: " << col_size <<
  // std::endl; fprintf(stderr, "Row: %p, Col: %p\n", r_ptr, c_ptr);

  start = std::chrono::high_resolution_clock::now();
  // torch::TensorOptions options = torch::TensorOptions()
  //                                    .dtype(torch::kFloat32)
  //                                    .device(torch::kCPU)
  //                                    .layout(torch::kStrided)
  //                                    .requires_grad(false);
  int num_gpus = torch::cuda::device_count();
  // pick a random gpu
  int device = rand() % num_gpus;
  auto row = torch::from_blob(r_ptr, {row_size, partition.h_dim},
                              FLOAT32_TENSOR_OPTIONS(torch::kCPU))
                 .to(torch::kCUDA, device);
  auto col = torch::from_blob(c_ptr, {col_size, partition.h_dim},
                              FLOAT32_TENSOR_OPTIONS(torch::kCPU))
                 .to(torch::kCUDA, device);

  // std::cerr << "Row: " << row << std::endl;
  // std::cerr << "Col: " << col << std::endl;

  auto result = torch::mm(row, col.transpose(0, 1)).to(torch::kCPU);
  auto matmul_logical_time = (double)row_size * partition.h_dim * col_size * 2 /
                             device_info_["flops"] * 1e6;  // in microseconds
  logical_time_ += (double)row_size * partition.h_dim * col_size * 2 /
                   device_info_["flops"] * 1e6;  // in microseconds
  // auto zeros = torch::zeros_like(result);
  // LOG_FATAL_IF(torch::allclose(result, zeros),
  //              "Result is all zero, something went wrong, worker cannot be "
  //              "launched as child process");

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG("Matmul real time: {}us, Matmul logical time: {}us",
            duration.count(), matmul_logical_time);

  start = std::chrono::high_resolution_clock::now();
  MatrixPartition response = partition;
  response.h_dim = result.size(1);
  response.mat.clear();
  response.mat.push_back({result.data_ptr(), result.numel() * sizeof(float)});
  // fprintf(stderr, "Response size: %ld\n", result.numel() * sizeof(float));

  auto [data, size] = response.Serialize();
  // replace req with rsp
  auto pos = std::string(message->topic).find("req");
  std::string topic = std::string(message->topic).replace(pos, 3, "rsp");
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG("RSP Serialization time: {}us", duration.count());

  Publish(topic, data, size);
  logical_time_ +=
      (double)size / device_info_["ul_bw"] * 1e6;  // in microseconds
  // LOG_INFO("Logical time: {}us", logical_time_.load());
  redis_->hset(uuid_, "logical_time", std::to_string(logical_time_.load()));
}

void MQTTWorker::OnConnect(struct mosquitto* mosq, void* userdata, int result) {
  if (result == 0) {
    // Connection successful
    int ret = 0;
    ret = mosquitto_subscribe(mosq, NULL, comp_topic_.c_str(), 0);
    LOG_FATAL_IF(ret != MOSQ_ERR_SUCCESS,
                 "Failed to subscribe to topic, error code: {}", ret);
    fprintf(stderr, "Subscribed to topic %s\n", comp_topic_.c_str());
    // ret = mosquitto_subscribe(mosq, NULL, timer_topic_.c_str(), 0);
    // LOG_FATAL_IF(ret != MOSQ_ERR_SUCCESS,
    //              "Failed to subscribe to topic, error code: {}", ret);
    // fprintf(stderr, "Subscribed to topic %s\n", timer_topic_.c_str());
  } else {
    LOG_FATAL("Connect failed with code {}", result);
  }
}

void MQTTWorker::SetUpMosq(struct mosquitto* mosq) {
  mosquitto_connect_callback_set(
      mosq, [](struct mosquitto* mosq, void* obj, int result) {
        static_cast<MQTTWorker*>(obj)->OnConnect(mosq, obj, result);
      });
  mosquitto_message_callback_set(
      mosq, [](struct mosquitto* mosq, void* obj,
               const struct mosquitto_message* message) {
        static_cast<MQTTWorker*>(obj)->OnMessage(mosq, obj, message);
      });
}

MQTTWorker::MQTTWorker(const std::string& uuid) : MQTTBase() {
  for (int i = 0; i < num_mosq_; i++) {
    // auto key = GenUUID();
    struct mosquitto* mosq = mosquitto_new(NULL, clean_session_, this);
    mosq_[i] = mosq;
    SetUpMosq(mosq);
  }
  // Start();
  InitLogger();

  uuid_ = uuid;

  comp_topic_ = std::string(MQTT_COMPUTE_TOPIC_REQ) + uuid;
  // timer_topic_ =
  //     std::string(MQTT_TIMER_TOPIC_REQ) +
  //     std::to_string(device_info_.at("id"));

  // CUDA context warmup and do random matmul
  torch::Tensor warmup_a = torch::rand({128, 4096}).to(torch::kCUDA, 0);
  torch::Tensor warmup_b = torch::rand({4096, 128}).to(torch::kCUDA, 0);

  torch::mm(warmup_a, warmup_b);

  redis_ = GetRedisConnection();
  std::unordered_map<std::string, std::string> info;
  redis_->hgetall(uuid, std::inserter(info, info.begin()));
  // convert to uint64_t
  for (auto& [key, value] : info) {
    device_info_[key] = std::stoull(value);
    // std::cerr << key << " " << value << std::endl;
  }

  // create a thread that refreshes the key in redis every 1 second
  std::thread t([this]() {
    while (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      auto set =
          redis_->expire(uuid_, 5);  // refresh the key, means device is alive
      LOG_FATAL_IF(!set, "Failed to refresh key in redis for device: {}",
                   uuid_);
    }
  });
  t.detach();
}
