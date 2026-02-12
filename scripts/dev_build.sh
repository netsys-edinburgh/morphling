#!/bin/bash
# 开发模式下的快速构建脚本

set -e

echo "=== Morphling Development Build Script ==="

# 检查是否在容器内
if [ ! -f /.dockerenv ]; then
    echo "Warning: This script is designed to run inside Docker container"
    echo "Run: docker-compose exec device-emulator bash scripts/dev_build.sh"
    exit 1
fi

# 设置环境变量
export Python3_ROOT_DIR=/usr
export Python3_EXECUTABLE=/usr/bin/python3.10
export MORPHLING_PYTHON_EXECUTABLE=/usr/bin/python3.10
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CPPFLAGS="-I/usr/include/python3.10"
export CMAKE_ARGS="-DPython3_EXECUTABLE=/usr/bin/python3.10 -DPython3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so -DPython3_INCLUDE_DIR=/usr/include/python3.10 -DCMAKE_PREFIX_PATH=/usr"

# 进入项目目录
cd /app

echo "=== Building morphling in development mode ==="

# 使用 pip install -e . 进行开发模式安装（可编辑安装）
python3 -m pip install --no-build-isolation --no-cache-dir --verbose -e .

echo "=== Build completed ==="
echo "You can now run your Python scripts to test the changes"
echo "Example: python3 scripts/run_devices.py --num_devices 4 --model_name facebook/opt-125m --backend proxy --seq_length 128 --batch_size 1 --cfg config/proxy/svr.ini"
