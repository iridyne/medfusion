# MedFusion Docker Image - Web UI 一体化容器
# 包含完整的 MedFusion 框架 + FastAPI 后端 + React 前端

# ============================================================================
# Stage 1: Builder - 构建 Python 依赖
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（快速依赖管理）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 复制依赖文件
COPY pyproject.toml README.md ./

# 创建虚拟环境并安装依赖（包含 web 可选依赖）
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -e ".[web]"

# ============================================================================
# Stage 2: Runtime - 运行时环境
# ============================================================================
FROM python:3.11-slim

LABEL maintainer="Medical AI Research Team"
LABEL description="MedFusion - Medical Multimodal Fusion Framework with Web UI"
LABEL version="0.3.0"

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制应用代码
COPY med_core ./med_core
COPY configs ./configs
COPY scripts ./scripts
COPY examples ./examples

# 创建数据目录（用于挂载）
RUN mkdir -p \
    /app/data \
    /app/outputs \
    /app/logs \
    /app/checkpoints \
    && chmod -R 777 /app/data /app/outputs /app/logs /app/checkpoints

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEDCORE_DATA_DIR=/app/data \
    MEDCORE_OUTPUT_DIR=/app/outputs \
    MEDCORE_LOG_DIR=/app/logs \
    MEDCORE_CHECKPOINT_DIR=/app/checkpoints

# 暴露 Web UI 端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认启动 Web UI 服务器
CMD ["uvicorn", "med_core.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
