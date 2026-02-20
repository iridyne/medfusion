#!/bin/bash

# MedFusion Web UI 停止脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "停止 MedFusion Web UI 服务..."

# 停止后端服务
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        print_info "停止后端服务 (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        print_success "后端服务已停止"
    else
        print_warning "后端服务未运行"
    fi
    rm logs/backend.pid
else
    print_warning "未找到后端 PID 文件"
fi

# 停止前端服务
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        print_info "停止前端服务 (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        print_success "前端服务已停止"
    else
        print_warning "前端服务未运行"
    fi
    rm logs/frontend.pid
else
    print_warning "未找到前端 PID 文件"
fi

# 额外清理：查找并停止所有相关进程
print_info "清理残留进程..."

# 停止 uvicorn 进程
pkill -f "uvicorn app.main:app" 2>/dev/null && print_success "清理了 uvicorn 进程" || true

# 停止 vite 进程
pkill -f "vite" 2>/dev/null && print_success "清理了 vite 进程" || true

print_success "所有服务已停止"
