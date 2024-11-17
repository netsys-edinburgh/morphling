#include "mqtt_server.h"

#include <iostream>

#include "common/types_and_defs.h"
#include "utils/logger.h"

void MQTTServer::OnMessage(struct mosquitto* mosq, void* obj,
                           const struct mosquitto_message* message) {
  // std::cout << "Received message: " << (char*)message->payload << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  partition.Deserialize(message->payload, message->payloadlen);
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG("Deserialization time: {}us",
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());

  auto [o_ptr, o_size] = partition.mat[0];
  int64_t row_size = o_size / partition.h_dim / sizeof(float);
  int64_t col_size = partition.h_dim;

  start = std::chrono::high_resolution_clock::now();
  auto output = torch::from_blob(o_ptr, {row_size, col_size},
                                 FLOAT32_TENSOR_OPTIONS(torch::kCPU));
  // std::cout << "Output: " << output << std::endl;
  UpdateMatrixBlock(output_, output, partition.row, partition.col,
                    partition.pivot, block_size_);
  end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG("UpdateMatrixBlock time: {}us",
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());

  int nan_count = torch::isnan(output_).sum().item<int64_t>();
  if (nan_count == 0) {
    output_cv_.notify_one();
  }
  // count the number of nans in the output
  LOG_DEBUG("Number of NaNs in output: {}, output size: {}", nan_count,
            output.numel());
}

torch::Tensor MQTTServer::DispatchMatMul(torch::Tensor& mat_a,
                                         torch::Tensor& mat_b) {
  output_ = CreateOutputMatrix(mat_a, mat_b);
  auto partitions = PartitionMatrices(mat_a, mat_b, block_size_);
  LOG_DEBUG("Number of partitions: {}", partitions.size());

  std::vector<std::future<void>> futures;
  // int64_t num_devices = std::stoi(GETENV("NUM_DEVICES", "1"));
  pub_buffer_.resize(partitions.size());
  pub_cb_count_ = partitions.size();
  // pub_count_ = 0;
  int64_t count = 0;
  for (auto& partition : partitions) {
    auto future = std::async(std::launch::async, [this, &partition, count] {
      auto [data, size] = partition.Serialize();
      auto topic =
          std::string("/morphling/req/") + std::to_string(count % num_devices_);

      Publish(topic, data, size);
      LOG_DEBUG("Published message to topic {}, count {}", topic, count);
      pub_buffer_[count] = data;
    });
    // pub_count_ = (pub_count_++) % num_devices_;
    count++;
    futures.push_back(std::move(future));
  }

  // auto start = std::chrono::high_resolution_clock::now();
  for (auto& f : futures) {
    f.wait();
  }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::cerr << "DispatchMatMul time: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(end -
  //                                                                    start)
  //                  .count()
  //           << "ms" << std::endl;
  // output needs to be all non-nan
  // while (std::isnan(output_.sum().item<float>())) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(100));
  // }
  auto start = std::chrono::high_resolution_clock::now();
  std::unique_lock<std::mutex> lock(output_mutex_);
  output_cv_.wait(lock,
                  [this] { return !torch::isnan(output_).any().item<bool>(); });
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG("Waiting time: {}us",
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());
  return output_;
}

void MQTTServer::OnConnect(struct mosquitto* mosq, void* userdata, int result) {
  if (result == 0) {
    // Connection successful
    int mosq_idx = 0;
    for (size_t i = 0; i < num_devices_; i++) {
      auto topic = std::string("/morphling/rsp/") + std::to_string(i);
      int ret =
          mosquitto_subscribe(mosq_[i % num_mosq_], NULL, topic.c_str(), 0);
      if (ret != MOSQ_ERR_SUCCESS) {
        LOG_ERROR("Failed to subscribe to topic {}, error code {}", topic, ret);
      }
      fprintf(stderr, "Subscribed to topic %s\n", topic.c_str());
    }
  } else {
    fprintf(stderr, "Connect failed\n");
  }
}

MQTTServer::MQTTServer(int64_t block_size)
    : MQTTBase(), block_size_(block_size) {
  num_devices_ = std::stoi(GETENV("NUM_DEVICES", "1"));
  for (int i = 0; i < num_mosq_; i++) {
    // auto key = GenUUID();
    struct mosquitto* mosq = mosquitto_new(NULL, clean_session_, this);
    mosq_[i] = mosq;
    SetUpMosq(mosq);
  }

  Start();
  InitLogger();
}

// void MQTTServer::OnPublish(struct mosquitto* mosq, void* obj, int mid) {
//   pub_cb_count_--;
//   fprintf(stderr, "Published message %d\n", mid);
//   if (pub_cb_count_ == 0) {
//     std::cout << "All messages published" << std::endl;
//     // for (auto* ptr : pub_buffer_) {
//     //   free(ptr);
//     // }
//     pub_buffer_.clear();
//   }
// }

void MQTTServer::SetUpMosq(struct mosquitto* mosq) {
  mosquitto_publish_callback_set(
      mosq, [](struct mosquitto* mosq, void* obj, int mid) {
        static_cast<MQTTServer*>(obj)->OnPublish(mosq, obj, mid);
      });
  mosquitto_connect_callback_set(
      mosq, [](struct mosquitto* mosq, void* obj, int result) {
        static_cast<MQTTServer*>(obj)->OnConnect(mosq, obj, result);
      });
  mosquitto_message_callback_set(
      mosq, [](struct mosquitto* mosq, void* obj,
               const struct mosquitto_message* message) {
        static_cast<MQTTServer*>(obj)->OnMessage(mosq, obj, message);
      });
}
