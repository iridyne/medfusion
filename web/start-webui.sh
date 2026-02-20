#!/bin/bash

# MedFusion Web UI 一键启动脚本
# 
# 功能：
# - 检查依赖
# - 初始化数据库
# - 启动后端服务
# - 启动前端服务

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
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

print_info "MedFusion Web UI 启动脚本"
echo "================================"

# 检查 Python
print_info "检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 未安装，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python 版本: $PYTHON_VERSION"

# 检查 Node.js
print_info "检查 Node.js 环境..."
if ! command -v node &> /dev/null; then
    print_error "Node.js 未安装，请先安装 Node.js 16+"
    exit 1
fi

NODE_VERSION=$(node --version)
print_success "Node.js 版本: $NODE_VERSION"

# 检查 npm
if ! command -v npm &> /dev/null; then
    print_error "npm 未安装"
    exit 1
fi

NPM_VERSION=$(npm --version)
print_success "npm 版本: $NPM_VERSION"

echo ""
print_info "开始安装依赖..."

# 安装后端依赖
print_info "安装后端依赖..."
cd backend

if [ ! -d "venv" ]; then
    print_info "创建 Python 虚拟环境..."
    python3 -m venv venv
fi

print_info "激活虚拟环境并安装依赖..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

print_success "后端依赖安装完成"

# 初始化数据库
print_info "初始化数据库..."
if [ -f "scripts/init_db.py" ]; then
    python scripts/init_db.py
    print_success "数据库初始化完成"
else
    print_warning "数据库初始化脚本不存在，跳过"
fi

cd ..

# 安装前端依赖
print_info "安装前端依赖..."
cd frontend

if [ ! -d "node_modules" ]; then
    print_info "安装 npm 包..."
    npm install
    print_success "前端依赖安装完成"
else
    print_success "前端依赖已存在"
fi

cd ..

echo ""
print_info "启动服务..."

# 创建日志目录
mkdir -p logs

# 启动后端服务
print_info "启动后端服务 (端口 8000)..."
cd backend
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid
print_success "后端服务已启动 (PID: $BACKEND_PID)"
cd ..

# 等待后端启动
print_info "等待后端服务启动..."
sleep 3

# 检查后端是否启动成功
if curl -s http://localhost:8000/health > /dev/null; then
    print_success "后端服务健康检查通过"
else
    print_warning "后端服务可能未完全启动，请检查日志"
fi

# 启动前端服务
print_info "启动前端服务 (端口 5173)..."
cd frontend
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid
print_success "前端服务已启动 (PID: $FRONTEND_PID)"
cd ..

echo ""
print_success "所有服务启动完成！"
echo ""
echo "================================"
echo "服务信息："
echo "  后端 API:  http://localhost:8000"
echo "  API 文档:  http://localhost:8000/docs"
echo "  前端界面:  http://localhost:5173"
echo ""
echo "日志文件："
echo "  后端日志:  logs/backend.log"
echo "  前端日志:  logs/frontend.log"
echo ""
echo "进程 ID："
echo "  后端 PID:  $BACKEND_PID"
echo "  前端 PID:  $FRONTEND_PID"
echo ""
echo "停止服务："
echo "  ./stop.sh"
echo "================================"
echo ""

# 提示用户
print_info "按 Ctrl+C 可以查看实时日志，服务将继续在后台运行"
print_info "或者直接访问 http://localhost:5173 开始使用"

# 可选：显示实时日志
# tail -f logs/backend.log logs/frontend.log
