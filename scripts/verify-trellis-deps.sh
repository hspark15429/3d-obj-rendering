#!/bin/bash
# Verification script for TRELLIS dependencies
# Run inside Docker container: docker run --gpus all --rm 3d-recon-test:trellis /app/scripts/verify-trellis-deps.sh

set -e

echo "=== TRELLIS Dependencies Verification ==="
echo

echo "1. Testing spconv import..."
python -c "
import spconv.pytorch as spconv
print(f'spconv version: {spconv.__version__}')
# Quick sparse tensor test
import torch
features = torch.randn(100, 32).cuda()
indices = torch.randint(0, 64, (100, 4)).int().cuda()
indices[:, 0] = 0  # batch index
spatial_shape = [64, 64, 64]
batch_size = 1
sp = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
print(f'Sparse tensor created: {sp.features.shape}')
print('spconv: OK')
"
echo

echo "2. Testing flash-attn or xformers..."
python -c "
try:
    import flash_attn
    print(f'flash-attn available: {flash_attn.__version__}')
except ImportError:
    print('flash-attn not available, falling back to xformers')
    import xformers
    print(f'xformers available: {xformers.__version__}')
print('Attention backend: OK')
"
echo

echo "3. Testing TRELLIS module import..."
python -c "
import os
os.environ['SPCONV_ALGO'] = 'native'
from trellis.pipelines import TrellisVGGTTo3DPipeline
print('TRELLIS pipeline importable: OK')
"
echo

echo "4. Testing our wrapper module..."
python -c "
from app.models.trellis import TrellisPipelineWrapper
wrapper = TrellisPipelineWrapper()
print(f'Wrapper created, loaded={wrapper.is_loaded()}')
print('Wrapper module: OK')
"
echo

echo "=== All verifications passed ==="
