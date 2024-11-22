#pragma once

#include "common/generator.h"
#include "mqtt_base.h"

class MQTTWorker : public MQTTBase {
 public:
  // MQTTWorker(const std::unordered_map<std::string, uint64_t>& device_info);
  MQTTWorker(const std::string& uuid);
  ~MQTTWorker() { Stop(); }
  void OnMessage(struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) override;

  void SetUpMosq(struct mosquitto* mosq);
  void OnConnect(struct mosquitto* mosq, void* userdata, int result);

 private:
  void HandleMatMul(const struct mosquitto_message* message);
  void HandleTimer(const struct mosquitto_message* message);

 private:
  std::string comp_topic_;
  std::string timer_topic_;
  std::unordered_map<std::string, uint64_t> device_info_;
  std::string uuid_;

  std::atomic_ullong logical_time_{0};
  sw::redis::Redis* redis_;
};