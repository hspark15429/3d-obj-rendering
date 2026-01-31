#!/bin/bash
# Verification script for CUDA 12.1 + PyTorch 2.4.1 stack
# Run this after building the Docker image:
#   docker run --gpus all --rm 3d-recon-test:cuda12 bash /app/scripts/verify-cuda-stack.sh

set -e

echo "=== CUDA/PyTorch Stack Verification ==="
echo ""

echo "1. PyTorch and CUDA verification:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    t = torch.tensor([1.0]).cuda()
    print(f'Tensor on GPU: {t.device}')
"

echo ""
echo "2. NumPy version check (must be < 2.0):"
python -c "
import numpy
print(f'NumPy version: {numpy.__version__}')
major_version = int(numpy.__version__.split('.')[0])
if major_version >= 2:
    print('ERROR: NumPy 2.x detected - will break spconv/nvdiffrast')
    exit(1)
print('OK: NumPy version is < 2.0')
"

echo ""
echo "3. nvdiffrast import test:"
python -c "
import nvdiffrast.torch as dr
print('nvdiffrast imported successfully')
print(f'nvdiffrast version: {dr.__version__ if hasattr(dr, \"__version__\") else \"unknown\"}')
"

echo ""
echo "=== All verifications passed! ==="
