# GPU-enabled Python container for 3D reconstruction
# Phase 3.1: Upgraded to CUDA 12.1 + PyTorch 2.4.1 for ReconViaGen compatibility
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

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

# Install PyTorch with CUDA 12.1 support FIRST (before other deps)
# This ensures correct CUDA version matching
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# NOTE: PyTorch3D wheel removed - cu118 wheel incompatible with cu121
# PyTorch3D will be addressed in Phase 3.1-02 (alternative: trimesh + nvdiffrast)

# Pin numpy<2 for nvdiffrast/spconv compatibility (NumPy 2.x breaks CUDA extensions ABI)
RUN pip install --no-cache-dir "numpy<2"

# Install nvdiffrast (NVIDIA's differentiable rasterizer)
# - --no-build-isolation: find PyTorch during CUDA extension build
# - TORCH_CUDA_ARCH_LIST: specify GPU architectures (no GPU during Docker build)
#   Updated for newer architectures: Ada Lovelace (8.9) and Hopper (9.0)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/

# Phase 3.1: TRELLIS/ReconViaGen dependencies

# Install spconv-cu120 (sparse convolutions for TRELLIS SLAT)
RUN pip install --no-cache-dir spconv-cu120==2.3.6

# Install xformers (memory-efficient attention, fallback if flash-attn unavailable)
RUN pip install --no-cache-dir xformers==0.0.27.post2

# Install flash-attn from pre-built wheel (CRITICAL: do NOT build from source - takes hours)
# Pre-built for: CUDA 12, PyTorch 2.4, Python 3.10, cxx11abi=FALSE
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install transformers and einops for VGGT backbone
RUN pip install --no-cache-dir transformers==4.46.3 einops==0.8.1

# Install huggingface_hub for model downloads
RUN pip install --no-cache-dir huggingface_hub==0.33.4

# Clone estheryang11/ReconViaGen (TRELLIS-VGGT implementation with app_refine.py)
# Uses --recursive to include TRELLIS submodule
RUN git clone --recursive --depth 1 https://github.com/estheryang11/ReconViaGen.git /app/reconviagen_src && \
    cp -r /app/reconviagen_src/trellis /app/ && \
    rm -rf /app/reconviagen_src

# CRITICAL: Set SPCONV_ALGO for spconv to work with TRELLIS
ENV SPCONV_ALGO=native

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
