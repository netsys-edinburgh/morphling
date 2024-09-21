#pragma once

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rttr/registration.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "common/reflection.h"
#include "utils/logger.h"

#define FROM_JSON_METHOD(struct_type)                                \
  void FromJson(const rapidjson::Value& json_obj) {                  \
    auto t = rttr::type::get<struct_type>();                         \
    for (auto& prop : t.get_properties()) {                          \
      const char* prop_name = prop.get_name().to_string().c_str();   \
      if (json_obj.HasMember(prop_name)) {                           \
        if (prop.get_type() == rttr::type::get<std::string>()) {     \
          prop.set_value(*this, json_obj[prop_name].GetString());    \
        } else if (prop.get_type() == rttr::type::get<int>()) {      \
          prop.set_value(*this, json_obj[prop_name].GetInt());       \
        } else if (prop.get_type() == rttr::type::get<double>()) {   \
          prop.set_value(*this, json_obj[prop_name].GetDouble());    \
        } else if (prop.get_type() == rttr::type::get<float>()) {    \
          prop.set_value(*this, json_obj[prop_name].GetFloat());     \
        } else if (prop.get_type() == rttr::type::get<bool>()) {     \
          prop.set_value(*this, json_obj[prop_name].GetBool());      \
        } else if (prop.get_type() == rttr::type::get<uint64_t>()) { \
          prop.set_value(*this, json_obj[prop_name].GetUint64());    \
        } else if (prop.get_type() == rttr::type::get<uint32_t>()) { \
          prop.set_value(*this, json_obj[prop_name].GetUint());      \
        } else {                                                     \
          LOG_FATAL("Unsupported type for property");                \
        }                                                            \
      }                                                              \
    }                                                                \
  }

template <typename T>
class JsonReader {
 public:
  JsonReader(const std::string& json_file) {
    json_file_ = json_file;
    std::ifstream ifs(json_file_);
    rapidjson::IStreamWrapper isw(ifs);
    d_.ParseStream(isw);
    LOG_DEBUG("Read json file: {}", json_file_);
  }

  std::unordered_map<std::string, T> ParseIntoMap() {
    std::unordered_map<std::string, T> result;
    for (auto& m : d_.GetObject()) {
      T obj;
      obj.FromJson(m.value);
      result[m.name.GetString()] = obj;
      // LOG_DEBUG("Read object: {}", m.name.GetString());
    }
    return result;
  }

 private:
  void property_setter(T obj, std::string prop_name, const std::string& value) {
    // // support type: int, string, double, float, bool, uint64_t, uint32_t
    // auto prop = obj.get_type().get_property(prop_name);
    // auto prop = var.get_type().get_property(prop_name);
    // T obj = var.get_value<T>();
    auto prop = rttr::type::get<T>().get_property(prop_name);

    LOG_DEBUG("Set property {} {}: {} = {}, type: {}", prop.is_valid(),
              prop.is_readonly(), prop_name, value,
              prop.get_type().get_name().to_string());

    if (prop) {
      bool success = false;
      if (prop.get_type() == rttr::type::get<int>()) {
        success = prop.set_value(obj, std::stoi(value));
        LOG_DEBUG("Set property int: {} = {}", prop_name, std::stoi(value));
      } else if (prop.get_type() == rttr::type::get<std::string>()) {
        success = prop.set_value(obj, value);
      } else if (prop.get_type() == rttr::type::get<double>()) {
        success = prop.set_value(obj, std::stod(value));
      } else if (prop.get_type() == rttr::type::get<float>()) {
        success = prop.set_value(obj, std::stof(value));
      } else if (prop.get_type() == rttr::type::get<bool>()) {
        success = prop.set_value(obj, value == "true");
      } else if (prop.get_type() == rttr::type::get<uint64_t>()) {
        success = prop.set_value(obj, std::stoull(value));
        LOG_DEBUG("Set property uint64_t: {} = {}", prop_name,
                  std::stoull(value));
      } else if (prop.get_type() == rttr::type::get<uint32_t>()) {
        success = prop.set_value(obj, std::stoul(value));
        LOG_DEBUG("Set property uint32_t: {} = {}", prop_name,
                  std::stoul(value));
      } else {
        LOG_FATAL("Unsupported type for property");
      }
      LOG_FATAL_IF(!success, "Failed to set property");
    }
  }

  std::string ValueToString(const rapidjson::Value& value) {
    if (value.IsString()) {
      return value.GetString();
    } else if (value.IsInt()) {
      return std::to_string(value.GetInt());
    } else if (value.IsUint()) {
      return std::to_string(value.GetUint());
    } else if (value.IsUint64()) {
      return std::to_string(value.GetUint64());
    } else if (value.IsDouble()) {
      return std::to_string(value.GetDouble());
    } else if (value.IsBool()) {
      return value.GetBool() ? "true" : "false";
    } else {
      LOG_FATAL("Unsupported type for json");
    }
  }

  T from_json(const rapidjson::Value& json_obj) {
    auto t = rttr::type::get<T>();
    // rttr::variant var = t.create();
    T obj;

    for (auto& prop : t.get_properties()) {
      auto prop_name = prop.get_name().to_string();
      LOG_DEBUG("Read property: {}", prop_name);
      if (json_obj.HasMember(prop_name.c_str())) {
        const rapidjson::Value& value = json_obj[prop_name.c_str()];

        if (value.IsObject()) {
          return from_json(value);
        }

        auto value_str = ValueToString(value);
        LOG_DEBUG("Set property: {} = {}", prop_name, value_str);
        property_setter(obj, prop_name, value_str);
      }
    }
    return obj;
  }

 private:
  rapidjson::Document d_;
  std::string json_file_;
};
