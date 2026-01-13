# CP-HOSfing Reproducibility Container
# CUDA 12.8 + Python 3.10 for GPU-accelerated experiment execution

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and essential build tools
# dos2unix ensures shell scripts work regardless of host OS line endings
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    dos2unix \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment code
COPY exps/ /app/exps/

# Create workspace directories for volume mounts
RUN mkdir -p /workspace/data /workspace/artifacts /workspace/configs

# Set PYTHONPATH so exps.* imports resolve correctly
ENV PYTHONPATH=/app

# Copy entrypoint script and ensure Unix line endings (works on Windows/Linux hosts)
COPY entrypoint.sh /app/entrypoint.sh
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Default environment variables
ENV CONFIG_FILE=""
ENV OUT_DIR="/workspace/artifacts"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["all"]
