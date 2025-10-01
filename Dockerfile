# DeviceEmulator Dockerfile
# Based on Ubuntu 22.04 with CUDA support

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST=8.6
# Python 环境变量
ENV PYTHON_EXECUTABLE=/usr/bin/python3.10
ENV MORPHLING_PYTHON_EXECUTABLE=/usr/bin/python3.10

# 安装系统依赖（使用 Python 3.10）
RUN apt-get update && apt-get install -y \
    # 基础工具
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    # CMake和编译工具
    cmake \
    ninja-build \
    # Python 3.10 相关
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libpython3.10-dev \
    python3.10-distutils \
    # gRPC依赖
    libssl-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgrpc-dev \
    libgrpc++-dev \
    # 日志和格式化库
    libfmt-dev \
    libspdlog-dev \
    # libevent 事件库
    libevent-dev \
    # 其他依赖
    libxml2-dev \
    xsltproc \
    uuid-dev \
    libtool \
    autoconf \
    automake \
    # 调试和编辑工具
    gdb \
    vim \
    iputils-ping \
    lsof \
    net-tools \
    librabbitmq-dev \
    libmosquitto-dev \
    libhiredis-dev \
    # 清理缓存
    && rm -rf /var/lib/apt/lists/*

# 编译安装 redis-plus-plus
RUN git clone --depth=1 https://github.com/sewenew/redis-plus-plus.git /tmp/redis-plus-plus && \
    cd /tmp/redis-plus-plus && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j && make install && \
    ldconfig && \
    rm -rf /tmp/redis-plus-plus

# 安装 libxslt（系统包）
RUN apt-get update && apt-get install -y libxslt1-dev && rm -rf /var/lib/apt/lists/*
    # rapidjson 源码安装
RUN git clone --depth=1 https://github.com/redis/hiredis.git /tmp/hiredis && \
    mkdir -p /tmp/hiredis/build && cd /tmp/hiredis/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j && make install && \
    rm -rf /tmp/hiredis
RUN git clone --depth=1 https://github.com/Tencent/rapidjson.git /tmp/rapidjson && \
    cp -r /tmp/rapidjson/include/rapidjson /usr/local/include/ && \
    rm -rf /tmp/rapidjson

# 设置 Python3.10 为默认 python3 和 python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    # 创建Python库符号链接，解决链接器找不到libpython的问题
    if [ -f /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 ]; then \
        ln -sf /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/libpython3.10.so; \
    fi && \
    # 检查Python库文件是否存在
    ls -la /usr/lib/x86_64-linux-gnu/libpython* || true && \
    # 检查Python开发包头文件
    ls -la /usr/include/python3.10/ || true

# 升级 pip 并安装基础 Python 包
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 创建工作目录
WORKDIR /app

# copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/
# 安装 Python 依赖和 PyTorch（直接用 pip）
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt


# 复制项目文件
COPY . /app/

# 构建和安装项目（使用系统 python）
# 先设置CMake环境变量来帮助找到Python库
ENV CMAKE_PREFIX_PATH="/usr/lib/x86_64-linux-gnu/cmake:$CMAKE_PREFIX_PATH"

# 构建前创建 libpython3.10.12.so 的软链接，解决 ld 找不到该库的问题
RUN if [ -f /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 ]; then \
    ln -sf /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/libpython3.10.12.so; \
    echo "创建了libpython3.10.12.so软链接"; \
fi && \
ls -la /usr/lib/x86_64-linux-gnu/libpython3.10* || echo "Warning: No libpython3.10 found"

# # 安装 RTTR (librttr) v0.9.6 from GitHub
# RUN git clone --branch v0.9.6 --depth=1 https://github.com/rttrorg/rttr.git /tmp/rttr && \
#     mkdir -p /tmp/rttr/build && cd /tmp/rttr/build && \
#     cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_DOCUMENTATION=OFF -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
#     make -j && make install && \
#     rm -rf /tmp/rttr
# 临时修复CMakeLists.txt以启用Python Development包查找
RUN sed -i 's/# find_package(Python COMPONENTS Development REQUIRED)/find_package(Python COMPONENTS Development REQUIRED)/' /app/CMakeLists.txt

RUN export Python3_ROOT_DIR=/usr && \
    export Python3_EXECUTABLE=/usr/bin/python3.10 && \
    export MORPHLING_PYTHON_EXECUTABLE=/usr/bin/python3.10 && \
    export LDFLAGS="-L/usr/lib/x86_64-linux-gnu" && \
    export CPPFLAGS="-I/usr/include/python3.10" && \
    export CMAKE_ARGS="-DPython3_EXECUTABLE=/usr/bin/python3.10 -DPython3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so -DPython3_INCLUDE_DIR=/usr/include/python3.10 -DCMAKE_PREFIX_PATH=/usr" && \
    echo "=== 开始构建 ===" && \
    python3 -m pip install --no-build-isolation --no-cache-dir --verbose .


# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/config

# 设置权限
RUN chmod +x /app/scripts/*.sh || true && chmod +x /app/csrc/*.sh || true

# 暴露端口
EXPOSE 443 6379 8080 39000 28516

# 设置环境变量用于运行时
ENV SPDLOG_LEVEL=debug
ENV MORPHLING_HOME=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import morphling; print('OK')" || exit 1

# 复制并设置入口点脚本
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 设置入口点和默认命令
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
