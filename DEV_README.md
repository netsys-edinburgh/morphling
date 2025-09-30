# Morphling Development Workflow

这个开发工作流允许你在不重新构建 Docker 镜像的情况下快速测试和开发 Morphling。

## 快速开始

### 1. 构建基础镜像（只需要做一次）
```bash
./dev.sh build
```

### 2. 启动开发容器
```bash
./dev.sh start
```

### 3. 进入容器进行开发
```bash
./dev.sh shell
```

### 4. 在容器内测试你的修改
```bash
# 如果修改了 C++ 代码，重新编译
./scripts/dev_build.sh

# 或者在宿主机上运行
./dev.sh rebuild

# 测试运行
python3 scripts/run_devices.py --num_devices 4 --model_name facebook/opt-125m --backend proxy --seq_length 128 --batch_size 1 --cfg config/proxy/svr.ini
```

## 工作流优势

### 🚀 快速开发
- **源码挂载**: 宿主机的源码直接挂载到容器内，修改立即生效
- **增量编译**: 使用卷缓存构建结果，避免重复编译
- **开发模式**: 自动检测源码变化并重新编译

### 🛠️ 便捷工具
- **一键命令**: `./dev.sh` 脚本提供所有必要操作
- **自动重建**: 容器启动时自动检测并重建必要组件
- **日志跟踪**: 轻松查看容器日志

## 可用命令

| 命令 | 描述 |
|------|------|
| `./dev.sh build` | 构建 Docker 镜像 |
| `./dev.sh start` | 启动开发容器 |
| `./dev.sh stop` | 停止容器 |
| `./dev.sh restart` | 重启容器 |
| `./dev.sh shell` | 进入容器 shell |
| `./dev.sh rebuild` | 在容器内重新编译 |
| `./dev.sh run <cmd>` | 在容器内运行命令 |
| `./dev.sh logs` | 查看容器日志 |
| `./dev.sh clean` | 清理容器和卷 |

## 开发流程示例

### 修改 C++ 代码
1. 在宿主机上修改 `csrc/` 目录下的文件
2. 运行 `./dev.sh rebuild` 重新编译
3. 运行 `./dev.sh run python3 scripts/run_devices.py ...` 测试

### 修改 Python 代码
1. 在宿主机上修改 `morphling/` 目录下的文件
2. 直接运行测试（Python 代码修改立即生效）

### 调试
```bash
# 进入容器
./dev.sh shell

# 设置调试级别日志
export SPDLOG_LEVEL=debug

# 运行并重定向日志到文件
SPDLOG_LEVEL=debug python run_devices.py --num_devices 4 --model_name facebook/opt-125m --backend proxy --seq_length 128 --batch_size 1 --cfg ../config/proxy/svr.ini 2>&1 | tee server.log
```

## 文件挂载

以下目录从宿主机挂载到容器：
- `./csrc` → `/app/csrc` (C++ 源码)
- `./morphling` → `/app/morphling` (Python 源码)
- `./scripts` → `/app/scripts` (脚本)
- `./config` → `/app/config` (配置文件)
- `./logs` → `/app/logs` (日志文件)
- `morphling_build` 卷 → `/app/build` (构建缓存)

## 环境变量

- `MORPHLING_DEV_MODE=1`: 启用开发模式
- `SPDLOG_LEVEL=debug`: 设置日志级别

## 注意事项

1. **首次构建**: 第一次需要构建基础镜像，这可能需要较长时间
2. **构建缓存**: 构建结果缓存在 Docker 卷中，删除卷会丢失缓存
3. **权限问题**: 如果遇到权限问题，确保 Docker 有权限访问项目目录
4. **端口冲突**: 确保端口 8080, 443, 39000, 28516 没有被占用

## 故障排除

### 构建失败
```bash
# 清理并重新构建
./dev.sh clean
./dev.sh build
```

### 权限错误
```bash
# 检查文件权限
ls -la scripts/
chmod +x scripts/*.sh
```

### 端口占用
```bash
# 检查端口使用
lsof -i :28516
# 或修改 docker-compose.yml 中的端口映射
```