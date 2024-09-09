#pragma once

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rttr/registration.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "common/reflection.h"

template <typename T>
class JsonReader {
 public:
  JsonReader(const std::string& json_file) {
    json_file_ = json_file;
    std::ifstream ifs(json_file_);
    rapidjson::IStreamWrapper isw(ifs);
    d.ParseStream(isw);
  }

  std::unordered_map<std::string, T> ParseIntoMap() {
    std::unordered_map<std::string, T> result;
    for (auto& m : d.GetObject()) {
      T t;
      from_json(m.value, t);
      result[m.name.GetString()] = t;
    }
    return result;
  }

 private:
  void property_setter(rttr::instance obj, const std::string& prop_name,
                       const std::string& value) {
    rttr::type t = obj.get_type();
    rttr::property prop = t.get_property(prop_name);

    // support type: int, string, double, float, bool, uint64_t, uint32_t
    if (prop) {
      if (prop.get_type() == rttr::type::get<int>()) {
        prop.set_value(obj, std::stoi(value));
      } else if (prop.get_type() == rttr::type::get<std::string>()) {
        prop.set_value(obj, value);
      } else if (prop.get_type() == rttr::type::get<double>()) {
        prop.set_value(obj, std::stod(value));
      } else if (prop.get_type() == rttr::type::get<float>()) {
        prop.set_value(obj, std::stof(value));
      } else if (prop.get_type() == rttr::type::get<bool>()) {
        prop.set_value(obj, value == "true");
      } else if (prop.get_type() == rttr::type::get<uint64_t>()) {
        prop.set_value(obj, std::stoull(value));
      } else if (prop.get_type() == rttr::type::get<uint32_t>()) {
        prop.set_value(obj, std::stoul(value));
      } else {
        LOG(FATAL) << "Unsupported type for property: " << prop_name;
      }
    }
  }

  void from_json(const rapidjson::Value& json_obj, rttr::instance obj) {
    rttr::type t = obj.get_type();
    for (auto& prop : t.get_properties()) {
      const char* prop_name = prop.get_name().to_string().c_str();

      if (json_obj.HasMember(prop_name)) {
        const rapidjson::Value& json_value = json_obj[prop_name];

        // always get string and convert
        std::string value = json_value.GetString();
        property_setter(obj, prop_name, value);
      }
    }
  }

 private:
  rapidjson::Document d;
  std::string json_file_;
};
