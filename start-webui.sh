#!/bin/bash

# MedFusion Web UI 快速启动脚本
# 用于快速启动 MedFusion Web UI

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  MedFusion Web UI - 快速启动${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 检查 Python 环境
echo -e "${YELLOW}[1/4]${NC} 检查 Python 环境..."
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗${NC} uv 未安装"
    echo -e "  请先安装 uv: ${BLUE}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} uv 已安装"

# 检查虚拟环境
echo -e "${YELLOW}[2/4]${NC} 检查虚拟环境..."
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}!${NC} 虚拟环境不存在，正在创建..."
    uv venv
    echo -e "${GREEN}✓${NC} 虚拟环境创建完成"
fi

# 安装依赖
echo -e "${YELLOW}[3/4]${NC} 检查依赖..."
if ! uv pip list | grep -q "fastapi"; then
    echo -e "${YELLOW}!${NC} 依赖未安装，正在安装..."
    uv pip install -e ".[web]"
    echo -e "${GREEN}✓${NC} 依赖安装完成"
else
    echo -e "${GREEN}✓${NC} 依赖已安装"
fi

# 检查前端构建
echo -e "${YELLOW}[4/4]${NC} 检查前端资源..."
if [ ! -f "med_core/web/static/index.html" ]; then
    echo -e "${YELLOW}!${NC} 前端资源未构建"

    if [ -d "web/frontend" ]; then
        echo -e "  正在构建前端..."
        cd web/frontend

        # 检查 node_modules
        if [ ! -d "node_modules" ]; then
            echo -e "  安装前端依赖..."
            npm install
        fi

        # 构建前端
        npm run build

        # 复制到后端
        echo -e "  复制构建产物..."
        cp -r dist/* ../../med_core/web/static/
        cd "$SCRIPT_DIR"

        echo -e "${GREEN}✓${NC} 前端构建完成"
    else
        echo -e "${RED}✗${NC} 前端源码不存在"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} 前端资源就绪"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ 所有检查通过，正在启动服务器...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 获取端口（默认 8000）
PORT=${1:-8000}
HOST=${2:-127.0.0.1}

echo -e "${GREEN}🚀 启动 MedFusion Web UI${NC}"
echo -e "   访问地址: ${BLUE}http://${HOST}:${PORT}${NC}"
echo -e "   API 文档: ${BLUE}http://${HOST}:${PORT}/docs${NC}"
echo -e ""
echo -e "   按 ${YELLOW}Ctrl+C${NC} 停止服务器"
echo ""

# 启动服务器
uv run uvicorn med_core.web.app:app --host "$HOST" --port "$PORT"
