#include "mqtt_worker.h"

#include <torch/torch.h>

#include <iostream>

#include "common/pytorch_defs.h"
#include "server_base.h"
#include "utils/logger.h"

void MQTTWorker::OnMessage(struct mosquitto* mosq, void* obj,
                           const struct mosquitto_message* message) {
  // std::cout << "Received message: " << *(int64_t*)message->payload <<
  // std::endl; auto payload = torch::from_blob(message->payload, {128, 4096});

  // // // to gpu +1 and back to cpu
  // // payload = payload.to(torch::kCUDA, 0);
  // // payload = payload * 2;
  // // payload = payload.to(torch::kCPU);
  // auto zeros = torch::zeros({128, 4096}).to(torch::kCUDA, 0);
  // std::cout << "Payload: "
  //           << torch::allclose(payload.to(torch::kCUDA, 0), zeros) <<
  //           std::endl;
  at::InferenceMode infer_guard(true);
  HandleMatMul(message);
}

void MQTTWorker::HandleMatMul(const struct mosquitto_message* message) {
  // std::cerr << "Handling matmul" << std::endl;

  MatrixPartition partition;
  partition.Deserialize(message->payload, message->payloadlen);

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
  torch::TensorOptions options = torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(torch::kCPU)
                                     .layout(torch::kStrided)
                                     .requires_grad(false);
  int num_gpus = torch::cuda::device_count();
  // pick a random gpu
  int device = rand() % num_gpus;
  auto row = torch::from_blob(r_ptr, {row_size, partition.h_dim}, options)
                 .to(torch::kCUDA, device);
  auto col = torch::from_blob(c_ptr, {col_size, partition.h_dim}, options)
                 .to(torch::kCUDA, device);

  // std::cerr << "Row: " << row << std::endl;
  // std::cerr << "Col: " << col << std::endl;

  auto result = torch::mm(row, col.transpose(0, 1)).to(torch::kCPU);
  auto zeros = torch::zeros_like(result);

  LOG_FATAL_IF(torch::allclose(result, zeros),
               "Result is all zero, something went wrong, worker cannot be "
               "launched as child process");

  MatrixPartition response;
  response.row = partition.row;
  response.col = partition.col;
  response.h_dim = col_size;
  response.pivot = partition.pivot;
  response.mat.push_back({result.data_ptr(), result.numel() * sizeof(float)});
  // fprintf(stderr, "Response size: %ld\n", result.numel() * sizeof(float));

  auto [data, size] = response.Serialize();
  // replace req with rsp
  auto pos = std::string(message->topic).find("req");
  std::string topic = std::string(message->topic).replace(pos, 3, "rsp");

  Publish(topic, data, size);
  LOG_DEBUG("Published response to topic {}", topic);
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

MQTTWorker::MQTTWorker(const std::string& topic) : MQTTBase(), topic_(topic) {
  for (int i = 0; i < num_mosq_; i++) {
    // auto key = GenUUID();
    struct mosquitto* mosq = mosquitto_new(NULL, clean_session_, this);
    mosq_[i] = mosq;
    SetUpMosq(mosq);
  }
  Start();
  InitLogger();
}
