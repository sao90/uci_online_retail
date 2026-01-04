# PLACEHOLDER DOCKERFILE.
# Idea for structuring pipeline container for runtime.

FROM python:3.12-slim

# --- System Dependencies ---
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # Clean up apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Arguments and Environment Variables ---
ARG USERNAME=appuser
ARG VERSION=latest

# --- User and Workspace Setup ---
RUN useradd -m -s /bin/bash ${USERNAME}
USER ${USERNAME}

ARG WORKSPACE=/home/${USERNAME}/app
WORKDIR ${WORKSPACE}

# Create necessary directories
RUN mkdir -p ${WORKSPACE}/data/pipeline_runs ${WORKSPACE}/logs

# --- Python Dependency Installation (Changes when pyproject.toml changes) ---
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml pdm.lock* ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# --- Application Code (Changes Most Frequently) ---
COPY --chown=${USERNAME}:${USERNAME} src ./src
COPY --chown=${USERNAME}:${USERNAME} components ./components
COPY --chown=${USERNAME}:${USERNAME} pipelines ./pipelines

# --- Python Path Configuration ---
ENV PYTHONPATH="${WORKSPACE}:${WORKSPACE}/src:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1

# Default command
CMD ["python", "--version"]
