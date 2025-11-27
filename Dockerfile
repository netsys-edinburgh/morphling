# DeviceEmulator Dockerfile
# Based on Ubuntu 22.04 with CUDA support

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

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

RUN apt-get update && apt-get upgrade -y


# 安装系统依赖（使用 Python 3.10）
RUN apt-get install -y \
    # 基础工具
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    # CMake和编译工具
    cmake \
    ninja-build \
    ccache \
    # Python 3.10 相关
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libpython3.10-dev \
    python3.10-distutils \
    # gRPC依赖
    libssl-dev \
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
    libmosquitto-dev \
    libhiredis-dev \
    # 清理缓存
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update \
 && apt-get install -y --no-install-recommends --fix-missing -o Acquire::Retries=3 dsniff \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Compile and install rabbitmq-c from source (v0.14.0)
RUN git clone --depth=1 -b v0.14.0 https://github.com/alanxz/rabbitmq-c.git /tmp/rabbitmq-c && \
    cd /tmp/rabbitmq-c && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
          -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
          -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .. && \
    make -j && make install && \
    ldconfig && \
    rm -rf /tmp/rabbitmq-c

# 编译安装 redis-plus-plus with _GLIBCXX_USE_CXX11_ABI=0 for torch compatibility
RUN git clone --depth=1 https://github.com/sewenew/redis-plus-plus.git /tmp/redis-plus-plus && \
    cd /tmp/redis-plus-plus && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .. && \
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

# 安装 uv (快速 Python 包管理器) pip install
RUN pip install uv && \
    # 检查 uv 是否安装成功
    uv --version


# 升级基础 Python 包并安装 PyTorch
RUN uv pip install --system --no-cache torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 创建工作目录
WORKDIR /app

# copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/
# 安装 Python 依赖和 PyTorch（使用 uv）
RUN uv pip install --system --no-cache -r /app/requirements.txt


ARG PROTOBUF_VER=v3.21.12
RUN git clone -b ${PROTOBUF_VER} https://github.com/protocolbuffers/protobuf.git /tmp/protobuf \
    && cd /tmp/protobuf \
    && git submodule update --init --recursive \
    && cmake -S . -B build -Dprotobuf_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
    && cmake --build build -j \
    && cmake --install build && \
    rm -rf /tmp/protobuf


# 构建前创建 libpython3.10.12.so 的软链接，解决 ld 找不到该库的问题
RUN if [ -f /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 ]; then \
    ln -sf /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/libpython3.10.12.so; \
    echo "创建了libpython3.10.12.so软链接"; \
fi && \
ls -la /usr/lib/x86_64-linux-gnu/libpython3.10* || echo "Warning: No libpython3.10 found"

# Copy CMake files first to leverage Docker layer caching
COPY CMakeLists.txt /app/
COPY cmake/ /app/cmake/

# Copy the rest of the project files
COPY . /app/

# 构建和安装项目（使用系统 python）with BuildKit cache mounts
# Cache mounts persist across builds: CMake deps in /app/build/_deps, ccache in /ccache
RUN --mount=type=cache,target=/app/build/_deps \
    --mount=type=cache,target=/ccache \
    --mount=type=cache,target=/root/.cache/cmake \
    export CCACHE_DIR=/ccache && \
    export CCACHE_MAXSIZE=5G && \
    export Python3_ROOT_DIR=/usr && \
    export Python3_EXECUTABLE=/usr/bin/python3.10 && \
    export MORPHLING_PYTHON_EXECUTABLE=/usr/bin/python3.10 && \
    export LDFLAGS="-L/usr/lib/x86_64-linux-gnu" && \
    export CPPFLAGS="-I/usr/include/python3.10" && \
    export CMAKE_ARGS="-DPython3_EXECUTABLE=/usr/bin/python3.10 -DPython3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so -DPython3_INCLUDE_DIR=/usr/include/python3.10 -DCMAKE_PREFIX_PATH=/usr -DFETCHCONTENT_BASE_DIR=/app/build/_deps -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache" && \
    echo "=== 开始构建 (with cache mounts) ===" && \
    uv pip install --system --no-build-isolation --no-cache --verbose . && \
    echo "=== Build complete ===" && \
    (ccache -s || echo "ccache stats not available")


# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/config

# 设置权限
RUN chmod +x /app/scripts/*.sh || true && chmod +x /app/csrc/*.sh || true

# 暴露端口
EXPOSE 39000

# 设置环境变量用于运行时
ENV SPDLOG_LEVEL=debug
ENV MORPHLING_HOME=/app