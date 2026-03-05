package com.example.confidant.globalStates;

import org.yaml.snakeyaml.Yaml;

import java.util.Map;

public class Config {
    public static Yaml yaml;
    public static Map<String, Object> cfg;

    public static Yaml getYaml() {
        if (yaml == null) {
            yaml = new Yaml();
        }
        return yaml;
    }

    public static Map<String, Object> getCfg() {
        return cfg;
    }
}
