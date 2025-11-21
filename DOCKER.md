# DeviceEmulator Docker 部署指南

本文档描述如何使用Docker部署DeviceEmulator项目。

## 前置要求

### 基本要求
- Docker 20.10+
- Docker Compose 2.0+
- 至少8GB内存
- 至少10GB磁盘空间

### GPU支持（可选）
如果需要GPU支持，还需要：
- NVIDIA GPU（计算能力7.0+）
- NVIDIA Docker支持
- nvidia-docker2 或 Docker with nvidia-container-toolkit

## 快速开始

### 1. 一键启动（推荐）
```bash
chmod +x quick-start.sh
./quick-start.sh
```

### 2. 手动启动

#### 构建镜像
```bash
DOCKER_BUILDKIT=1 docker build -t device-emulator:latest .
```

#### 启动服务（GPU模式）
```bash
docker-compose up -d
```

#### 启动服务（CPU模式）
```bash
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## 使用方法

### 进入容器
```bash
docker-compose exec device-emulator bash
```

### 运行设备模拟器
```bash
# 在容器内运行
morphling_emulator --num_devices 4 --model_name facebook/opt-125m --backend proxy --seq_length 128 --batch_size 1

# 或者直接从外部运行
docker-compose exec device-emulator morphling_emulator --num_devices 4 --model_name facebook/opt-125m
```

### 生成设备配置
```bash
docker-compose exec device-emulator morphling_device_config --num_devices 4 --device_type virtual
```

### 查看日志
```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs device-emulator
docker-compose logs redis

# 实时跟踪日志
docker-compose logs -f device-emulator
```

## 服务说明

### 服务列表
- **redis**: Redis数据库服务（端口6379）
- **device-emulator**: 主要的DeviceEmulator服务
- **nginx-proxy**: 可选的Nginx代理服务（使用profile启动）

### 端口映射
- `6379`: Redis服务
- `8080`: DeviceEmulator HTTP服务
- `443`: DeviceEmulator HTTPS服务

### 数据卷
- `./logs`: 日志文件
- `./data`: 数据文件
- `./config`: 配置文件
- `redis_data`: Redis持久化数据

## 高级配置

### 环境变量
可以通过修改`docker-compose.yml`中的environment部分来配置：

```yaml
environment:
  - SPDLOG_LEVEL=debug          # 日志级别
  - CUDA_VISIBLE_DEVICES=0      # 可见GPU设备
  - MORPHLING_HOME=/app         # 应用根目录
```

### GPU配置
如果有多个GPU，可以指定使用的GPU：
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1   # 使用GPU 0和1
```

### 启动Nginx代理
```bash
docker-compose --profile proxy up -d
```

## 故障排除

### 常见问题

#### 1. CUDA相关错误
```bash
# 检查NVIDIA Docker支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# 如果不支持GPU，使用CPU模式
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

#### 2. 内存不足
```bash
# 增加Docker内存限制，或者减少设备数量
docker-compose exec device-emulator morphling_emulator --num_devices 2
```

#### 3. 端口冲突
```bash
# 检查端口使用情况
netstat -tulpn | grep -E ':(6379|8080|443)'

# 修改docker-compose.yml中的端口映射
```

#### 4. 权限问题
```bash
# 确保脚本有执行权限
chmod +x quick-start.sh docker-entrypoint.sh

# 检查Docker权限
sudo usermod -aG docker $USER
# 需要重新登录生效
```

### 调试命令

```bash
# 检查容器状态
docker-compose ps

# 检查容器资源使用
docker stats

# 进入容器调试
docker-compose exec device-emulator bash

# 重启服务
docker-compose restart device-emulator

# 查看详细日志
docker-compose logs --details device-emulator
```

## 清理

### 停止服务
```bash
docker-compose down
```

### 完全清理（包括数据）
```bash
docker-compose down -v
docker rmi device-emulator:latest
```

## 生产环境部署

### 安全考虑
1. 使用专用的网络
2. 配置防火墙规则
3. 使用环境变量管理敏感配置
4. 定期备份数据卷

### 性能优化
1. 使用SSD存储
2. 配置适当的内存和CPU限制
3. 监控资源使用情况
4. 根据负载调整设备数量

### 监控和日志
1. 集成日志聚合系统
2. 设置健康检查和报警
3. 监控GPU和CPU使用率
4. 定期检查容器状态
