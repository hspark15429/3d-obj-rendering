# GPU-enabled Python container for 3D reconstruction
# Phase 3: Added PyTorch ecosystem and model weight infrastructure
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and build essentials (needed for PyTorch3D compilation)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 11.8 support FIRST (before other deps)
# This ensures correct CUDA version matching
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch3D from source (requires CUDA toolkit for compilation)
# Using pre-built wheels from pytorch3d for faster builds
RUN pip install --no-cache-dir \
    "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl"

# Install nvdiffrast (NVIDIA's differentiable rasterizer)
# Requires --no-build-isolation to find PyTorch during CUDA extension build
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/

# Copy and install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model weights directory structure
# Weights will be downloaded and cached here
RUN mkdir -p /app/weights/reconviagen /app/weights/nvdiffrec

# Copy application code
COPY app/ app/

# Create storage directories
RUN mkdir -p /app/storage/jobs

# Expose API port
EXPOSE 8000

# Use exec form CMD for proper signal handling
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
