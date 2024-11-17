#pragma once

#include "common/generator.h"
#include "common/pytorch_defs.h"
#include "mqtt_base.h"
#include "server_base.h"

class MQTTServer : public MQTTBase {
 public:
  MQTTServer(int64_t block_size = 32);
  ~MQTTServer() = default;

  torch::Tensor DispatchMatMul(torch::Tensor& mat_a, torch::Tensor& mat_b);

 private:
  void OnMessage(struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) override;
  // void OnPublish(struct mosquitto* mosq, void* obj, int mid);
  void OnConnect(struct mosquitto* mosq, void* userdata, int result);
  void SetUpMosq(struct mosquitto* mosq);
  //   void Publish(MatrixPartition& partition);
  //   void RunPublishThread(int idx);

  // void Subscribe(const std::string& topic) {
  //   mosquitto_subscribe(mosq_, NULL, topic.c_str(), 0);
  //   mosquitto_message_callback_set(mosq_, [](struct mosquitto* mosq, void*
  //   obj, const struct mosquitto_message* message) {
  //     static_cast<MQTTServer*>(obj)->OnMessage(mosq, obj, message);
  //   });
  // }

  // void Publish(std::string& topic, void* payload, int payloadlen);

 private:
  int64_t block_size_;
  int num_devices_;
  torch::Tensor output_;
  std::mutex output_mutex_;
  std::condition_variable output_cv_;
};