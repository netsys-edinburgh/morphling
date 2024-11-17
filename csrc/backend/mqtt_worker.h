#pragma once

#include "common/generator.h"
#include "mqtt_base.h"

class MQTTWorker : public MQTTBase {
 public:
  MQTTWorker(const std::string& topic);
  ~MQTTWorker() { Stop(); }
  void OnMessage(struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) override;

  void SetUpMosq(struct mosquitto* mosq);
  void OnConnect(struct mosquitto* mosq, void* userdata, int result) {
    if (result == 0) {
      // Connection successful
      mosquitto_subscribe(mosq, NULL, topic_.c_str(), 0);
      fprintf(stderr, "Subscribed to topic %s\n", topic_.c_str());
    } else {
      fprintf(stderr, "Connect failed\n");
    }
  }

  // void Subscribe(const std::string& topic) {

  //   fprintf(stderr, "Subscribed to topic %s\n", topic.c_str());
  // }

 private:
  void HandleMatMul(const struct mosquitto_message* message);

 private:
  std::string topic_;
};