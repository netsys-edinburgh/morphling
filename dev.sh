#!/bin/bash
# 便捷的开发工作流脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
IMAGE_NAME="device-emulator:latest"
CONTAINER_NAME="morphling-emulator"

cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Morphling Development Workflow ===${NC}"

# 函数：显示帮助
show_help() {
    echo "Usage: $0 [COMMAND] [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  quickstart     - 启动 Redis、网络和模拟器容器"
    echo "  build          - 重新构建 Docker 镜像 (GPU 模式)"
    echo "  build-cpu      - 重新构建 Docker 镜像 (CPU 模式)"
    echo "  start          - 启动开发容器 (GPU 模式)"
    echo "  start-cpu      - 启动开发容器 (CPU 模式)"
    echo "  stop           - 停止容器"
    echo "  restart        - 重启容器 (GPU 模式)"
    echo "  restart-cpu    - 重启容器 (CPU 模式)"
    echo "  shell          - 进入容器 shell"
    echo "  rebuild        - 在容器内重新编译 morphling"
    echo "  test           - 运行测试"
    echo "  run [cmd]      - 在容器内运行命令"
    echo "  logs           - 查看容器日志"
    echo "  clean          - 清理容器和卷"
    echo ""
    echo "Examples:"
    echo "  $0 start          # 启动GPU模式"
    echo "  $0 start-cpu      # 启动CPU模式"
    echo "  $0 shell"
    echo "  $0 rebuild"
    echo "  $0 run python3 scripts/run_devices.py --num_devices 4 --model_name facebook/opt-125m --backend proxy"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: docker is not installed${NC}"
        exit 1
    fi
    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker daemon is not running${NC}"
        exit 1
    fi
}

# 函数：构建镜像
build_image() {
    echo -e "${YELLOW}Building Docker image (GPU mode)...${NC}"
    docker build -t "$IMAGE_NAME" .
    echo -e "${GREEN}Build completed${NC}"
}

# 函数：构建镜像 (CPU模式)
build_image_cpu() {
    echo -e "${YELLOW}Building Docker image (CPU mode)...${NC}"
    docker build -t "$IMAGE_NAME" .
    echo -e "${GREEN}Build completed${NC}"
}

start_container_common() {
    local use_gpu="$1"
    local gpu_args=()

    if [ "$use_gpu" = "true" ]; then
        gpu_args=(--gpus all)
    fi

    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker run -d \
        --name "$CONTAINER_NAME" \
        "${gpu_args[@]}" \
        -p 39000:39000 \
        "$IMAGE_NAME" \
        tail -f /dev/null
}

quickstart() {
    local redis_container_name="morphling-redis"
    local emulator_container_name="morphling-emulator"
    local network_name="morphling-network"
    local use_gpu=false

    if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
        echo "NVIDIA Docker support detected"
        use_gpu=true
    else
        echo "WARNING: NVIDIA Docker support not detected. Running in CPU-only mode."
    fi

    echo "Building DeviceEmulator Docker image..."
    docker build -t "$IMAGE_NAME" .

    if ! docker network inspect "$network_name" &> /dev/null; then
        docker network create "$network_name" >/dev/null
    fi

    while :; do
        local redis_containers morphling_containers emulator_containers deviceemu_containers morphling_redis_containers
        redis_containers=$(docker ps -aq -f name=redis)
        morphling_containers=$(docker ps -aq -f name=morphling)
        emulator_containers=$(docker ps -aq -f name="$emulator_container_name")
        deviceemu_containers=$(docker ps -aq -f name=deviceemulator-device-emulator)
        morphling_redis_containers=$(docker ps -aq -f name=morphling-redis)

        if [ -n "$redis_containers" ]; then
            echo "Stopping and removing existing redis containers..."
            docker rm -f $redis_containers
        fi
        if [ -n "$morphling_containers" ]; then
            echo "Stopping and removing existing morphling containers..."
            docker rm -f $morphling_containers
        fi
        if [ -n "$emulator_containers" ]; then
            echo "Stopping and removing existing morphling-emulator containers..."
            docker rm -f $emulator_containers
        fi
        if [ -n "$deviceemu_containers" ]; then
            echo "Stopping and removing existing deviceemulator-device-emulator containers..."
            docker rm -f $deviceemu_containers
        fi
        if [ -n "$morphling_redis_containers" ]; then
            echo "Stopping and removing existing morphling-redis containers..."
            docker rm -f $morphling_redis_containers
        fi

        if [ -z "$redis_containers" ] && [ -z "$morphling_containers" ] && [ -z "$deviceemu_containers" ] && [ -z "$morphling_redis_containers" ]; then
            break
        fi

        sleep 1
    done

    echo "Starting services..."
    docker run -d \
        --name "$redis_container_name" \
        --network "$network_name" \
        -p 6379:6379 \
        redis

    if [ "$use_gpu" = true ]; then
        docker run -d \
            --name "$emulator_container_name" \
            --network "$network_name" \
            --gpus all \
            -p 39000:39000 \
            "$IMAGE_NAME" \
            tail -f /dev/null
    else
        echo "Starting in CPU-only mode..."
        docker run -d \
            --name "$emulator_container_name" \
            --network "$network_name" \
            -p 39000:39000 \
            "$IMAGE_NAME" \
            tail -f /dev/null
    fi

    echo "Waiting for services to start..."
    sleep 10

    echo "Service status:"
    docker ps --filter "name=morphling" --filter "name=redis" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "Available commands:"
    echo "  docker exec -it $emulator_container_name bash                    # 进入容器"
    echo "  docker exec -it $emulator_container_name morphling_emulator --help  # 查看帮助"
    echo "  docker logs $emulator_container_name                         # 查看日志"
    echo "  docker stop $emulator_container_name $redis_container_name    # 停止服务"
    echo ""
    echo "Redis is running on: localhost:6379"
    echo "DeviceEmulator is running in container: $emulator_container_name"
}

# 函数：启动容器
start_container() {
    echo -e "${YELLOW}Starting development container (GPU mode)...${NC}"
    start_container_common true
    echo -e "${GREEN}Container started${NC}"
    echo -e "${BLUE}Run '$0 shell' to enter the container${NC}"
}

# 函数：启动容器 (CPU模式)
start_container_cpu() {
    echo -e "${YELLOW}Starting development container (CPU mode)...${NC}"
    start_container_common false
    echo -e "${GREEN}Container started${NC}"
    echo -e "${BLUE}Run '$0 shell' to enter the container${NC}"
}

# 函数：停止容器
stop_container() {
    echo -e "${YELLOW}Stopping containers...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    echo -e "${GREEN}Containers stopped${NC}"
}

# 函数：重启容器
restart_container() {
    echo -e "${YELLOW}Restarting containers (GPU mode)...${NC}"
    docker restart "$CONTAINER_NAME"
    echo -e "${GREEN}Container restarted${NC}"
}

# 函数：重启容器 (CPU模式)
restart_container_cpu() {
    echo -e "${YELLOW}Restarting containers (CPU mode)...${NC}"
    docker restart "$CONTAINER_NAME"
    echo -e "${GREEN}Container restarted${NC}"
}

# 函数：进入容器shell
enter_shell() {
    echo -e "${YELLOW}Entering container shell...${NC}"
    docker exec -it "$CONTAINER_NAME" bash
}

# 函数：在容器内重新编译
rebuild_in_container() {
    echo -e "${YELLOW}Rebuilding morphling in container...${NC}"
    docker exec -it "$CONTAINER_NAME" bash /app/scripts/dev_build.sh
    echo -e "${GREEN}Rebuild completed${NC}"
}

# 函数：运行命令
run_command() {
    shift  # 移除 'run' 参数
    echo -e "${YELLOW}Running command in container: $@${NC}"
    docker exec -it "$CONTAINER_NAME" bash -c "$*"
}

# 函数：查看日志
show_logs() {
    echo -e "${YELLOW}Showing container logs...${NC}"
    docker logs -f "$CONTAINER_NAME"
}

# 函数：清理
clean_up() {
    echo -e "${YELLOW}Cleaning up containers and volumes...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    docker image rm "$IMAGE_NAME" 2>/dev/null || true
    echo -e "${GREEN}Cleanup completed${NC}"
}

# 函数：运行测试
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    docker exec -it "$CONTAINER_NAME" bash -c "cd /app && python3 -m pytest tests/ -v"
}

# 主逻辑
check_docker

case "${1:-help}" in
    "quickstart")
        quickstart
        ;;
    "build")
        build_image
        ;;
    "build-cpu")
        build_image_cpu
        ;;
    "start")
        start_container
        ;;
    "start-cpu")
        start_container_cpu
        ;;
    "stop")
        stop_container
        ;;
    "restart")
        restart_container
        ;;
    "restart-cpu")
        restart_container_cpu
        ;;
    "shell")
        enter_shell
        ;;
    "rebuild")
        rebuild_in_container
        ;;
    "test")
        run_tests
        ;;
    "run")
        run_command "$@"
        ;;
    "logs")
        show_logs
        ;;
    "clean")
        clean_up
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
