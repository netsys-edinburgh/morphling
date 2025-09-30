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

# 检查是否为开发模式
if [ "$MORPHLING_DEV_MODE" = "1" ]; then
    echo "=== Development Mode Detected ==="
    echo "Source code is mounted from host"
    
    # 检查是否需要重新构建
    if [ ! -f "/app/build/.dev_built" ] || [ "/app/setup.py" -nt "/app/build/.dev_built" ] || [ "/app/pyproject.toml" -nt "/app/build/.dev_built" ]; then
        echo "=== Building morphling in development mode ==="
        
        # 设置环境变量
        export Python3_ROOT_DIR=/usr
        export Python3_EXECUTABLE=/usr/bin/python3.10
        export MORPHLING_PYTHON_EXECUTABLE=/usr/bin/python3.10
        export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
        export CPPFLAGS="-I/usr/include/python3.10"
        export CMAKE_ARGS="-DPython3_EXECUTABLE=/usr/bin/python3.10 -DPython3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so -DPython3_INCLUDE_DIR=/usr/include/python3.10 -DCMAKE_PREFIX_PATH=/usr"
        
        # 开发模式安装
        python3 -m pip install --no-build-isolation --no-cache-dir -e . || {
            echo "Failed to build in development mode, trying regular install..."
            python3 -m pip install --no-build-isolation --no-cache-dir .
        }
        
        # 标记构建完成
        touch /app/build/.dev_built
        echo "=== Development build completed ==="
    else
        echo "=== Using cached development build ==="
    fi
    
    echo "=== Available development commands ==="
    echo "  scripts/dev_build.sh - Rebuild morphling"
    echo "  python3 scripts/run_devices.py ... - Run device emulator"
    echo "  morphling_device_config - Generate device config"
fi

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