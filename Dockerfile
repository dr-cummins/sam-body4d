FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# Dev Mode requires: bash, curl, wget, procps, git, git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    wget \
    procps \
    git \
    git-lfs \
    ffmpeg \
    python3.11 \
    python3.11-venv \
    python3-pip \
    htop \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create non-root user (uid 1000 required by HF Spaces)
RUN useradd -m -u 1000 user

# Create /data directory for persistent storage
RUN mkdir -p /data/inputs /data/outputs && \
    chown -R user:user /data

# Set up /app directory (Dev Mode requires code in /app, owned by uid 1000)
RUN mkdir -p /app && chown user:user /app

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

# Copy project code into /app
COPY --chown=user:user . /app

# Expose Gradio port
EXPOSE 7860

# Startup command — keeps container alive for Dev Mode SSH
CMD ["python3", "-c", "print('SAM-Body4D container ready. Connect via Dev Mode SSH.'); import time; time.sleep(float('inf'))"]
