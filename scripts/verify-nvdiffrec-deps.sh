#!/bin/bash
# Verify nvdiffrec dependencies
# Run inside Docker container: docker exec <container> /app/scripts/verify-nvdiffrec-deps.sh

set -e

echo "=== nvdiffrec Dependencies Verification ==="
echo ""

# Test tiny-cuda-nn
echo "Testing tiny-cuda-nn..."
python -c "import tinycudann as tcnn; print(f'  tiny-cuda-nn OK (version: {tcnn.__version__})')" 2>/dev/null || \
python -c "import tinycudann as tcnn; print('  tiny-cuda-nn OK')"
echo ""

# Test nvdiffrast
echo "Testing nvdiffrast..."
python -c "import nvdiffrast.torch as dr; print('  nvdiffrast OK')"
echo ""

# Test imageio
echo "Testing imageio..."
python -c "import imageio; print(f'  imageio OK (version: {imageio.__version__})')"
echo ""

# Test camera estimation service
echo "Testing camera estimation service..."
python -c "
from app.services.camera_estimation import look_at_matrix, compute_fov_x, create_nerf_dataset, validate_nerf_dataset
import numpy as np

# Test look_at_matrix
mat = look_at_matrix([0, 0, 2.5], [0, 0, 0], [0, 1, 0])
assert mat.shape == (4, 4), 'Matrix shape incorrect'
print('  look_at_matrix OK')

# Test compute_fov_x
fov = compute_fov_x(512, 1111.0)
assert 0.4 < fov < 0.5, f'FOV outside expected range: {fov}'
print(f'  compute_fov_x OK (FOV: {np.degrees(fov):.1f} degrees)')

print('  camera_estimation OK')
"
echo ""

# Check nvdiffrec reference code is available
echo "Checking nvdiffrec reference code..."
if [ -d "/app/nvdiffrec_src" ]; then
    echo "  /app/nvdiffrec_src exists"
    ls -la /app/nvdiffrec_src/*.py 2>/dev/null | head -5 || echo "  (no .py files in root)"
else
    echo "  WARNING: /app/nvdiffrec_src not found"
fi
echo ""

echo "=== All nvdiffrec dependencies verified! ==="
