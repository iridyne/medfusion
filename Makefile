.PHONY: help install dev test lint format clean docker-build docker-up

help:
	@echo "MedFusion 开发命令"
	@echo ""
	@echo "安装和环境:"
	@echo "  make install       - 安装项目依赖"
	@echo "  make dev           - 安装开发依赖"
	@echo ""
	@echo "开发工具:"
	@echo "  make test          - 运行测试"
	@echo "  make lint          - 代码检查"
	@echo "  make format        - 代码格式化"
	@echo "  make typecheck     - 类型检查"
	@echo ""
	@echo "清理:"
	@echo "  make clean         - 清理缓存和临时文件"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - 构建 Docker 镜像"
	@echo "  make docker-up     - 启动 Docker 服务"

install:
	uv sync

dev:
	uv sync --all-extras
	pre-commit install

test:
	uv run pytest -q

test-cov:
	uv run pytest --cov=med_core --cov-report=html --cov-report=term

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check . --fix

typecheck:
	uv run mypy med_core

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
