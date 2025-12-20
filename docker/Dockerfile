# Zenith + Triton Inference Server Production Dockerfile
# Multi-stage build for optimized image size
#
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

# =============================================================================
# Stage 1: Build Environment
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml setup.cfg ./
COPY zenith/ ./zenith/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build wheel && \
    pip wheel --no-cache-dir --wheel-dir=/build/wheels -e .

# =============================================================================
# Stage 2: Runtime Environment
# =============================================================================
FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS runtime

LABEL maintainer="Wahyu Ardiansyah <zenith@project.io>"
LABEL description="Zenith ML Optimization Framework with Triton Inference Server"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_REPOSITORY=/models

# Install Zenith from wheels
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy Zenith Python package
COPY zenith/ /opt/zenith/zenith/

# Add Zenith to Python path
ENV PYTHONPATH="${PYTHONPATH}:/opt/zenith"

# Create model repository directory
RUN mkdir -p ${MODEL_REPOSITORY} && \
    chmod 755 ${MODEL_REPOSITORY}

# Create non-root user for security
RUN groupadd -r zenith && \
    useradd -r -g zenith -d /home/zenith -s /sbin/nologin zenith && \
    chown -R zenith:zenith ${MODEL_REPOSITORY}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Expose Triton ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics (Prometheus)
EXPOSE 8000 8001 8002

# Default model repository volume
VOLUME ["/models"]

# Switch to non-root user
USER zenith

# Default command: Start Triton with Zenith backend
CMD ["tritonserver", \
    "--model-repository=${MODEL_REPOSITORY}", \
    "--model-control-mode=poll", \
    "--repository-poll-secs=15", \
    "--strict-model-config=false", \
    "--log-verbose=1"]
