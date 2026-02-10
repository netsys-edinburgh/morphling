# DeviceEmulator Dockerfile
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

RUN apt-get update && apt-get upgrade -y

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
    libmosquitto-dev \
    libhiredis-dev \
    rapidjson-dev \
    libxslt1-dev \
    # 调试和编辑工具
    gdb \
    vim \
    nano \
    psmisc \
    tmux \
    iputils-ping \
    lsof \
    net-tools

# 创建工作目录
WORKDIR /app

# copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache -r /app/requirements.txt

# Copy the rest of the project files
COPY . /app/

# ccache: persist compiler cache across Docker builds (requires BuildKit)
ENV CCACHE_DIR=/ccache
RUN ccache -M 5G

# 构建和安装项目（使用系统 python）with BuildKit cache mount for ccache
RUN --mount=type=cache,target=/ccache \
    pip install --no-build-isolation --verbose ./

# Build standalone C++ tests (CUDA/cuBLAS)
RUN --mount=type=cache,target=/ccache \
    cmake -S tests/cpp -B tests/cpp/build -DENABLE_ZEROCOPY_TESTS=OFF && \
    cmake --build tests/cpp/build -j

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/config

# 设置权限
RUN chmod +x /app/scripts/*.sh || true && chmod +x /app/csrc/*.sh || true

# 暴露端口
EXPOSE 39000

# 设置环境变量用于运行时
ENV SPDLOG_LEVEL=debug
ENV MORPHLING_HOME=/app
