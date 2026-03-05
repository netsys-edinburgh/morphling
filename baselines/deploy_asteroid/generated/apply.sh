#!/bin/bash
# Apply baselines-asteroid K8s manifests
set -e

MANIFEST_DIR="/home/ubuntu/Desktop/dheeraj/DeviceEmulator/baselines/scripts/../deploy_asteroid/generated"
NAMESPACE="default"

echo "Applying manifests from $MANIFEST_DIR"
echo "Namespace: $NAMESPACE"

if ! kubectl get namespace "$NAMESPACE"     &>/dev/null; then
    echo "Creating namespace $NAMESPACE..."
    kubectl create namespace "$NAMESPACE"
fi

echo "Applying ConfigMap..."
kubectl apply -f "$MANIFEST_DIR/00-configmap.yaml"

echo "Applying Headless Service..."
kubectl apply -f     "$MANIFEST_DIR/01-headless-service.yaml"

echo "Applying Worker Jobs..."
for job in "$MANIFEST_DIR"/02-job-rank-*.yaml; do
    echo "  Applying $(basename $job)..."
    kubectl apply -f "$job"
done

echo "All manifests applied!"
echo ""

echo "Waiting for pods to start..."
kubectl wait --for=condition=Ready     pod -l app=baselines-asteroid     -n "$NAMESPACE" --timeout=300s || true

echo ""
echo "Pod Status:"
kubectl get pods -l app=baselines-asteroid     -n "$NAMESPACE" -o wide
