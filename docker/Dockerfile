# syntax=docker/dockerfile:1

# MedFusion Docker Image - FastAPI BFF + Python worker entry
# v0.4 baseline: PostgreSQL/Redis/Auth ready through MEDFUSION_* envs

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic, fast dependency installation.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml README.md ./
COPY med_core ./med_core

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install ".[web]"

FROM python:3.11-slim

ARG MEDFUSION_VERSION=0.4.0

LABEL maintainer="Medical AI Research Team"
LABEL description="MedFusion - Medical Multimodal Fusion Framework with Web UI"
LABEL version="${MEDFUSION_VERSION}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Keep source code available for debug and static assets.
COPY med_core ./med_core
COPY configs ./configs
COPY scripts ./scripts
COPY examples ./examples

RUN mkdir -p \
    /app/data \
    /app/outputs \
    /app/logs \
    /app/checkpoints

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEDFUSION_DATA_DIR=/app/data \
    MEDFUSION_LOG_LEVEL=INFO \
    MEDFUSION_HOST=0.0.0.0 \
    MEDFUSION_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "med_core.cli", "start", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
