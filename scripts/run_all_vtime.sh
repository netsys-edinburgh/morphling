#!/usr/bin/env bash
set -euo pipefail

for baseline in cleave dtfm asteroid confident alpa; do
  ./scripts/run_vtime_experiment.sh "$baseline" \
    "results/comparison/manifests/${baseline}_manifest.json" \
    "results/comparison/generated_device_fleet.json" \
    "results/vtime/${baseline}/"
done
