ARG BASE_IMAGE="python:3.11-slim"

# =========================================================================
# Stage 1: 'base'
# This stage now contains only the common setup for all environments.
# =========================================================================
FROM ${BASE_IMAGE} AS base

# Install essential system-level dependencies.
RUN apt-get update && \
    apt-get install -y curl build-essential python3-pip tk-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for better security.
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID appgroup && \
    useradd -m -s /bin/bash -u $UID -g $GID appuser

# Install a clean version of pip and then install Poetry.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install poetry==1.8.2

# =========================================================================
# Stage 2: 'builder-dev'
# Builds the final development image with dependencies installed.
# =========================================================================
FROM base AS builder-dev

WORKDIR /app

# Copy all local libraries required for the build.
COPY --chown=appuser:appgroup ai-delver-runtime ./ai-delver-runtime
COPY --chown=appuser:appgroup ai-delver-level ./ai-delver-level
COPY --chown=appuser:appgroup pytiling-lib ./pytiling-lib
COPY --chown=appuser:appgroup pyglet-dragonbones-lib ./pyglet-dragonbones-lib
COPY --chown=appuser:appgroup assets ./assets

WORKDIR /app/ai-delver-intelligence

# Copy the dependency files.
COPY --chown=appuser:appgroup ai-delver-intelligence/pyproject.toml ./
COPY --chown=appuser:appgroup ai-delver-intelligence/poetry.lock ./

# Install all dependencies using Poetry as root.
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Switch to the non-root user for the final image state.
USER appuser

# =========================================================================
# Stage 3: 'production'
# This stage builds the final, lean production image.
# =========================================================================
FROM base AS production

# Set the python path to include the install location
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages

# Copy all installed python packages from the 'builder-dev' stage.
COPY --from=builder-dev /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the application source code into the image.
WORKDIR /app
COPY --chown=appuser:appgroup src ./src

# Set the user for runtime.
USER appuser

# Set the default command to start the server.
CMD ["python3", "src/main.py"]