ARG BASE_IMAGE="python:3.11-slim"

# =========================================================================
# Stage 1: 'base'
# Setup OS, system deps, and non-root user.
# =========================================================================
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies. 'python3-tk' resolves to the correct version for the base image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl build-essential python3-pip tk-dev python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID appgroup && \
    useradd -m -s /bin/bash -u $UID -g $GID appuser

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install poetry==1.8.2

# =========================================================================
# Stage 2: 'builder-dev'
# Install Python dependencies and compile local libs.
# =========================================================================
FROM base AS builder-dev

WORKDIR /app

COPY --chown=appuser:appgroup ai-delver-runtime ./ai-delver-runtime
COPY --chown=appuser:appgroup ai-delver-level ./ai-delver-level
COPY --chown=appuser:appgroup pytiling-lib ./pytiling-lib
COPY --chown=appuser:appgroup pyglet-dragonbones-lib ./pyglet-dragonbones-lib
COPY --chown=appuser:appgroup assets ./assets

WORKDIR /app/ai-delver-intelligence
COPY --chown=appuser:appgroup ai-delver-intelligence/pyproject.toml ./
COPY --chown=appuser:appgroup ai-delver-intelligence/poetry.lock ./

# Remove system-level 'blinker' to avoid conflicts with Poetry's version on Debian
RUN apt-get update && apt-get remove -y python3-blinker || true

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

USER appuser

# =========================================================================
# Stage 3: 'production'
# Final runtime image. Copies only installed packages and source code.
# =========================================================================
FROM base AS production

# Dynamic PYTHONPATH to support varying Python versions in TensorFlow images (3.9, 3.10, or 3.11)
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages:/usr/local/lib/python3.10/site-packages:/usr/local/lib/python3.9/site-packages:/usr/local/lib/site-packages

# Copy installed packages from builder using wildcard to match any Python version
COPY --from=builder-dev /usr/local/lib/python3.*/site-packages /usr/local/lib/site-packages

# Ensure the destination is also in path
ENV PYTHONPATH=/usr/local/lib/site-packages:$PYTHONPATH

WORKDIR /app
COPY --chown=appuser:appgroup src ./src
USER appuser

CMD ["python3", "src/main.py"]