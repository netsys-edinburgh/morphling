#!/bin/bash
set -e

# DeviceEmulator Docker入口点脚本

echo "=== DeviceEmulator Container Starting ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "No NVIDIA GPU detected"
fi

# 验证morphling安装
echo "Verifying morphling installation..."
python -c "import morphling; print('Morphling imported successfully')" || {
    echo "ERROR: Failed to import morphling"
    exit 1
}

# 如果传入了参数，执行参数
if [ $# -gt 0 ]; then
    echo "Executing command: $@"
    exec "$@"
else
    echo "No command provided. Starting interactive bash shell."
    echo "Available commands:"
    echo "  morphling_emulator - Start device emulator"
    echo "  morphling_device_config - Generate device config"
    echo "  morphling_device - Run device"
    echo "  morphling_cmd - Command line interface"
    exec bash
fi