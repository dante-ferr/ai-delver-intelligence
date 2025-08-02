#FROM paperspace/gradient-base:pt211-tf215-cudatk120-py311-20240202 

ARG BASE_IMAGE="python:3.11.7-slim"
FROM ${BASE_IMAGE}

# Set timezone and install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    curl \
    python3-tk \
    build-essential && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Create user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID appgroup && \
    useradd -m -u $UID -g $GID appuser

# Install Poetry
ENV POETRY_VERSION=1.8.2
ENV PATH="/root/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set workdir and copy project files
WORKDIR /app

# Copy dependency files first for better caching
COPY ai-delver-intelligence/pyproject.docker.toml /app/pyproject.toml
COPY ai-delver-intelligence/poetry.lock* /app/

# Copy library dependencies
COPY assets /app/assets
COPY ai-delver-level /app/ai-delver-level
COPY ai-delver-runtime /app/ai-delver-runtime
COPY pyglet-dragonbones-lib /app/pyglet-dragonbones-lib
COPY pytiling-lib /app/pytiling-lib

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --no-interaction --no-ansi

# Copy source code
COPY ai-delver-intelligence /app

# Set permissions
RUN chown -R $UID:$GID /app

# Switch to non-root user
USER appuser
ENV HOME=/home/appuser
ENV PYTHONUNBUFFERED=1

CMD ["python3", "src/main.py"]