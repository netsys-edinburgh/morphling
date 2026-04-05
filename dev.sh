#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CONTAINER_NAME="device-emulator"
IMAGE_GPU="device-emulator:latest"
IMAGE_CPU="device-emulator:cpu"

echo -e "${BLUE}=== Morphling Development Workflow ===${NC}"

show_help() {
    echo "Usage: $0 [COMMAND] [ARGS...]"
    echo ""
    echo "Commands:"
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
    echo "  clean          - 清理容器和镜像"
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
}

build_image() {
    echo -e "${YELLOW}Building Docker image (GPU mode)...${NC}"
    docker build -t "$IMAGE_GPU" .
    echo -e "${GREEN}Build completed${NC}"
}

build_image_cpu() {
    echo -e "${YELLOW}Building Docker image (CPU mode)...${NC}"
    docker build -t "$IMAGE_CPU" .
    echo -e "${GREEN}Build completed${NC}"
}

start_container() {
    echo -e "${YELLOW}Starting development container (GPU mode)...${NC}"
    docker run -d --gpus all --name "$CONTAINER_NAME" "$IMAGE_GPU"
    echo -e "${GREEN}Container started${NC}"
    echo -e "${BLUE}Run '$0 shell' to enter the container${NC}"
}

start_container_cpu() {
    echo -e "${YELLOW}Starting development container (CPU mode)...${NC}"
    docker run -d --name "$CONTAINER_NAME" "$IMAGE_CPU"
    echo -e "${GREEN}Container started${NC}"
    echo -e "${BLUE}Run '$0 shell' to enter the container${NC}"
}

stop_container() {
    echo -e "${YELLOW}Stopping containers...${NC}"
    docker stop "$CONTAINER_NAME" || true
    echo -e "${GREEN}Containers stopped${NC}"
}

restart_container() {
    echo -e "${YELLOW}Restarting containers (GPU mode)...${NC}"
    docker restart "$CONTAINER_NAME"
    echo -e "${GREEN}Container restarted${NC}"
}

restart_container_cpu() {
    echo -e "${YELLOW}Restarting containers (CPU mode)...${NC}"
    docker restart "$CONTAINER_NAME"
    echo -e "${GREEN}Container restarted${NC}"
}

enter_shell() {
    echo -e "${YELLOW}Entering container shell...${NC}"
    docker exec -it "$CONTAINER_NAME" bash
}

rebuild_in_container() {
    echo -e "${YELLOW}Rebuilding morphling in container...${NC}"
    docker exec "$CONTAINER_NAME" bash /app/scripts/dev_build.sh
    echo -e "${GREEN}Rebuild completed${NC}"
}

run_command() {
    shift
    echo -e "${YELLOW}Running command in container: $@${NC}"
    docker exec "$CONTAINER_NAME" bash -c "$*"
}

show_logs() {
    echo -e "${YELLOW}Showing container logs...${NC}"
    docker logs -f "$CONTAINER_NAME"
}

clean_up() {
    echo -e "${YELLOW}Cleaning up containers and images...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    docker rmi "$IMAGE_GPU" "$IMAGE_CPU" 2>/dev/null || true
    echo -e "${GREEN}Cleanup completed${NC}"
}

run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    docker exec "$CONTAINER_NAME" bash -c "cd /app && python3 -m pytest tests/ -v"
}

check_docker

case "${1:-help}" in
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
