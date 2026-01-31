# Phase 4: Quality & Preview - Research

**Researched:** 2026-01-31
**Domain:** Image quality metrics (PSNR/SSIM), 3D mesh rendering with nvdiffrast, depth comparison
**Confidence:** HIGH

## Summary

Phase 4 requires computing quality metrics (PSNR, SSIM, depth error) and generating preview images from reconstructed meshes. The standard approach uses **scikit-image 0.26.0** for PSNR/SSIM metrics, **nvdiffrast** (already available) for differentiable mesh rendering, and **trimesh** for edge extraction. Camera poses for comparison rendering should be extracted from the existing transforms_train.json file created during preprocessing.

The critical insight is to **reuse existing camera poses** rather than creating new ones - the system already has canonical camera transforms in transforms_train.json that were used to create the mesh. Rendering from these exact same poses ensures pixel-perfect comparison between input images and rendered output.

Wireframe rendering is straightforward: extract mesh edges using trimesh.edges_unique, then render them as line primitives. For nvdiffrast, this means rendering edge endpoints and rasterizing lines in a separate pass.

**Primary recommendation:** Use scikit-image.metrics for PSNR/SSIM (HIGH confidence), nvdiffrast for rendering with poses from transforms_train.json, and 6 preview angles (matching the 6 canonical orthogonal views already in the system).

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-image | 0.26.0 | PSNR/SSIM computation | Industry standard for image quality metrics, well-maintained, comprehensive API |
| nvdiffrast | latest (git) | Mesh rendering | Already in project, GPU-accelerated differentiable rendering, NVIDIA official |
| trimesh | 4.0.0+ | Mesh processing & edge extraction | Already in project, comprehensive mesh utilities |
| numpy | <2.0 | Depth comparison metrics | Already constrained for ABI compatibility, standard for numerical operations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pillow | 10.0.0+ | Image I/O for metrics | Already in project, saving preview PNGs |
| torch | 2.4.1 | Tensor operations for rendering | Already in project, nvdiffrast requires PyTorch |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scikit-image | IQA-PyTorch | IQA-PyTorch offers GPU acceleration and more metrics (LPIPS, FID), but adds dependency and complexity. scikit-image is sufficient for PSNR/SSIM. |
| scikit-image | sewar | sewar has more metrics (MS-SSIM, VIF, UQI) but is less maintained. scikit-image is more stable. |
| nvdiffrast | PyTorch3D | PyTorch3D was removed in Phase 3.1 due to CUDA compatibility. nvdiffrast is already available and faster. |

**Installation:**
```bash
# All dependencies already in requirements.txt
pip install scikit-image>=0.26.0
# nvdiffrast already installed from git in Dockerfile
# trimesh>=4.0.0 already in requirements.txt
```

## Architecture Patterns

### Recommended Module Structure
```
app/services/
├── quality_metrics.py       # PSNR/SSIM/depth comparison functions
├── mesh_renderer.py          # nvdiffrast rendering wrapper
└── preview_generator.py      # Orchestrates rendering + saving previews
```

### Pattern 1: Reuse Existing Camera Poses
**What:** Load camera transforms from transforms_train.json created during preprocessing, use identical poses for comparison rendering.

**When to use:** Always - this ensures pixel-perfect alignment between input images and rendered output for accurate PSNR/SSIM computation.

**Example:**
```python
# Source: Existing codebase pattern from app/services/camera_estimation.py
import json
from pathlib import Path

def load_camera_poses(input_dir: Path) -> list:
    """Load camera poses from NeRF dataset format."""
    transforms_path = input_dir / "transforms_train.json"
    with open(transforms_path) as f:
        data = json.load(f)

    return [
        {
            'transform_matrix': frame['transform_matrix'],
            'file_path': frame['file_path']
        }
        for frame in data['frames']
    ]
```

### Pattern 2: Batch Image Quality Metrics
**What:** Compute PSNR/SSIM for all view pairs, then average for overall metrics.

**When to use:** For multi-view reconstruction quality assessment (per user decision for overall metrics only).

**Example:**
```python
# Source: https://scikit-image.org/docs/stable/api/skimage.metrics.html
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def compute_image_quality(img_true, img_test, data_range=1.0):
    """
    Compute PSNR and SSIM between two images.

    Args:
        img_true: Reference image (H, W, C) in [0, 1]
        img_test: Test image (H, W, C) in [0, 1]
        data_range: Max value range (1.0 for float images)

    Returns:
        dict with 'psnr' and 'ssim' keys
    """
    psnr = peak_signal_noise_ratio(img_true, img_test, data_range=data_range)
    ssim = structural_similarity(
        img_true, img_test,
        data_range=data_range,
        channel_axis=-1  # Specify channel axis for multi-channel images
    )

    return {'psnr': psnr, 'ssim': ssim}

def compute_overall_metrics(image_pairs: list) -> dict:
    """Average metrics across all views."""
    psnr_scores = []
    ssim_scores = []

    for img_true, img_test in image_pairs:
        metrics = compute_image_quality(img_true, img_test)
        psnr_scores.append(metrics['psnr'])
        ssim_scores.append(metrics['ssim'])

    return {
        'psnr': float(np.mean(psnr_scores)),
        'ssim': float(np.mean(ssim_scores)),
        'psnr_std': float(np.std(psnr_scores)),
        'ssim_std': float(np.std(ssim_scores))
    }
```

### Pattern 3: nvdiffrast Rendering Pipeline
**What:** Transform mesh to clip space, rasterize with nvdiffrast, interpolate texture/color attributes.

**When to use:** For generating comparison renders and preview images from reconstructed mesh.

**Example:**
```python
# Source: https://github.com/NVlabs/nvdiffrast/blob/main/samples/torch/earth.py
import torch
import nvdiffrast.torch as dr

def render_mesh(glctx, vertices, faces, uvs, texture, mvp_matrix, resolution):
    """
    Render textured mesh with nvdiffrast.

    Args:
        glctx: nvdiffrast GL context
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle indices
        uvs: (V, 2) texture coordinates
        texture: (H, W, 3) texture map
        mvp_matrix: (4, 4) model-view-projection matrix
        resolution: Output image resolution [height, width]

    Returns:
        (H, W, 3) rendered RGB image
    """
    # Transform vertices to clip space
    pos_clip = vertices @ mvp_matrix.T  # (V, 4)

    # Rasterize
    rast_out, rast_out_db = dr.rasterize(
        glctx,
        pos_clip[None, ...],  # Add batch dimension
        faces,
        resolution=resolution
    )

    # Interpolate texture coordinates
    texc, texd = dr.interpolate(
        uvs[None, ...],
        rast_out,
        faces,
        rast_db=rast_out_db,
        diff_attrs='all'
    )

    # Sample texture
    color = dr.texture(
        texture[None, ...],
        texc,
        texd,
        filter_mode='linear-mipmap-linear'
    )

    # Remove batch dimension and alpha channel
    return color[0, :, :, :3]
```

### Pattern 4: Depth Comparison Metrics
**What:** Compute MAE (Mean Absolute Error) or RMSE between rendered depth and input depth maps.

**When to use:** For geometry validation alongside RGB metrics (per user decision to include depth comparison).

**Example:**
```python
# Source: Standard practice from depth estimation literature
import numpy as np

def compute_depth_error(depth_true, depth_pred, valid_mask=None):
    """
    Compute depth error metrics (MAE, RMSE).

    Args:
        depth_true: Reference depth map (H, W)
        depth_pred: Predicted depth map (H, W)
        valid_mask: Boolean mask for valid depth pixels (H, W)

    Returns:
        dict with 'mae', 'rmse' keys
    """
    if valid_mask is None:
        valid_mask = (depth_true > 0) & (depth_pred > 0)

    diff = depth_true[valid_mask] - depth_pred[valid_mask]

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'valid_pixels': int(valid_mask.sum())
    }
```

### Pattern 5: Wireframe Rendering via Edge Extraction
**What:** Extract unique mesh edges using trimesh, render as line primitives.

**When to use:** For wireframe preview images (per user decision for both textured and wireframe renderings).

**Example:**
```python
# Source: https://trimesh.org/trimesh.base.html
import trimesh
import numpy as np

def extract_mesh_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Extract unique edges from mesh for wireframe rendering.

    Args:
        mesh: trimesh Trimesh object

    Returns:
        (E, 2, 3) array of edge endpoints
    """
    # Get unique edges (E, 2) with vertex indices
    edge_indices = mesh.edges_unique

    # Convert to vertex positions (E, 2, 3)
    edge_vertices = mesh.vertices[edge_indices]

    return edge_vertices
```

### Anti-Patterns to Avoid
- **Don't create new camera poses for preview images**: Reuse transforms_train.json poses for 6 canonical views, ensures consistency with training data
- **Don't compute per-view metrics when averaging**: Per user decision, only overall averaged metrics are needed - computing per-view adds complexity without value
- **Don't hand-roll PSNR/SSIM**: scikit-image handles edge cases (division by zero when images identical, data_range auto-detection issues, multichannel support)
- **Don't assume data_range**: Always specify data_range explicitly for PSNR/SSIM (1.0 for float images in [0,1], 255 for uint8)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PSNR computation | Manual MSE + log10(MAX^2/MSE) | `skimage.metrics.peak_signal_noise_ratio` | Handles division by zero (identical images), data_range auto-detection, multi-channel images |
| SSIM computation | Manual luminance/contrast/structure comparison | `skimage.metrics.structural_similarity` | Complex windowing, Gaussian weighting, gradient support, validates image compatibility |
| Camera projection matrix | Manual 4x4 matrix math | Use existing `look_at_matrix()` from camera_estimation.py | Already tested, handles up vector orthogonalization |
| Mesh edge extraction | Loop through faces to find boundary edges | `trimesh.edges_unique` property | Handles degenerate cases, cached for performance |
| Depth error metrics | Manual numpy loops | Vectorized numpy operations with valid masks | Efficient, handles invalid/missing depth pixels |

**Key insight:** Image quality metrics have subtle edge cases that cause incorrect results if hand-rolled. PSNR division-by-zero, SSIM data_range mismatches, and depth invalid-pixel handling all require careful implementation that's already solved in mature libraries.

## Common Pitfalls

### Pitfall 1: PSNR Division by Zero
**What goes wrong:** When comparing identical images, MSE is zero, causing PSNR formula to divide by zero and return infinity or NaN.

**Why it happens:** PSNR = 10 * log10(MAX^2 / MSE). When MSE=0, this becomes log10(infinity).

**How to avoid:** Use `skimage.metrics.peak_signal_noise_ratio` which returns infinity for identical images (documented behavior). If hand-rolling, check for MSE < epsilon before computing.

**Warning signs:** NaN or inf values in quality.json, tests failing when comparing mesh to itself.

### Pitfall 2: SSIM data_range Mismatch
**What goes wrong:** SSIM scores are wildly incorrect (e.g., negative values, >1.0) because data_range parameter doesn't match actual image value range.

**Why it happens:** scikit-image auto-detects data_range from dtype, but this fails for float images. Float images are assumed to be [-1, 1] (range=2.0) when they're actually [0, 1] (range=1.0).

**How to avoid:** **Always specify data_range explicitly**: `data_range=1.0` for float images in [0, 1], `data_range=255` for uint8 images.

**Warning signs:** SSIM values outside [0, 1] range, SSIM lower than expected for visually similar images.

**Source:** https://scikit-image.org/docs/stable/api/skimage.metrics.html - "If data_range is not specified, the range is automatically guessed based on the image data type. However for floating-point image data, this estimate yields a result double the value of the desired range."

### Pitfall 3: nvdiffrast Image Coordinate Convention
**What goes wrong:** Rendered images are vertically flipped compared to input images, causing PSNR/SSIM comparison to fail or give incorrect results.

**Why it happens:** nvdiffrast uses OpenGL bottom-up memory ordering, while PIL/numpy use top-down ordering.

**How to avoid:** Flip rendered images vertically before comparison: `rendered = rendered[::-1, :, :]` or `rendered = np.flipud(rendered)`.

**Warning signs:** PSNR unusually low despite visually similar images, SSIM structural component incorrect.

**Source:** https://nvlabs.github.io/nvdiffrast/ - "The framework follows OpenGL conventions, including bottom-up image memory ordering requiring vertical flipping for standard top-down workflows."

### Pitfall 4: Depth Comparison Without Valid Mask
**What goes wrong:** Depth error metrics include background pixels (depth=0) which skews results, making geometry quality appear worse than it is.

**Why it happens:** Input depth maps have zero values for background, but these shouldn't be included in MAE/RMSE computation.

**How to avoid:** Create valid_mask for pixels with non-zero depth in both reference and predicted: `valid_mask = (depth_true > 0) & (depth_pred > 0)`, then compute metrics only on valid_mask pixels.

**Warning signs:** Depth MAE/RMSE unexpectedly high, metric values dominated by background regions.

### Pitfall 5: Averaging SSIM Per-Pixel vs Per-Image
**What goes wrong:** Computing SSIM on concatenated images gives different results than averaging per-image SSIM scores.

**Why it happens:** SSIM uses local windows for structural comparison. Concatenating images creates artificial boundaries that affect window statistics.

**How to avoid:** Compute SSIM separately for each view pair, then average the scores. Don't concatenate images before calling structural_similarity().

**Warning signs:** SSIM values change when view order changes, inconsistent results between test runs.

### Pitfall 6: Mesh Without Texture Coordinates
**What goes wrong:** Rendering fails or produces black images because mesh lacks UV coordinates required for texture sampling.

**Why it happens:** GLB export from TRELLIS includes texture, but conversion to OBJ/PLY might lose texture coordinates.

**How to avoid:** Verify mesh has UVs before texture rendering: `if mesh.visual.uv is None: use_vertex_colors_instead()`. Load original GLB for textured rendering rather than converted OBJ.

**Warning signs:** Black or solid-color rendered images, nvdiffrast texture() errors, mesh validation shows has_uvs=False.

### Pitfall 7: Hardcoded Image Resolution
**What goes wrong:** Preview images are rendered at wrong resolution (not matching input), causing resize artifacts in comparison metrics.

**Why it happens:** Code assumes 512px images but input may be different resolution.

**How to avoid:** Per user decision, "resolution matches input image resolution" - extract resolution from first input image and use for all rendering. Store in quality.json metadata.

**Warning signs:** Blurry preview images, PSNR lower than expected, SSIM penalizes high-frequency details.

## Code Examples

Verified patterns from official sources:

### Computing PSNR/SSIM with scikit-image
```python
# Source: https://scikit-image.org/docs/stable/api/skimage.metrics.html
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from PIL import Image

def load_image_float(path):
    """Load image as float array in [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0

# Load reference and test images
img_true = load_image_float('input/view_00.png')
img_test = load_image_float('output/render_00.png')

# Compute metrics (CRITICAL: specify data_range for float images)
psnr = peak_signal_noise_ratio(img_true, img_test, data_range=1.0)
ssim = structural_similarity(
    img_true, img_test,
    data_range=1.0,
    channel_axis=-1,  # Last axis is RGB channels
    gaussian_weights=True,  # Default, use Gaussian weighting
    sigma=1.5,  # Default window size
    use_sample_covariance=False  # Default for consistency
)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
```

### Loading Camera Poses from NeRF Dataset
```python
# Source: Existing codebase app/services/camera_estimation.py
import json
import numpy as np
from pathlib import Path

def load_camera_transforms(dataset_dir: Path):
    """Load camera transforms from transforms_train.json."""
    transforms_path = dataset_dir / "transforms_train.json"

    with open(transforms_path) as f:
        data = json.load(f)

    camera_angle_x = data['camera_angle_x']
    frames = []

    for frame in data['frames']:
        frames.append({
            'transform_matrix': np.array(frame['transform_matrix']),
            'image_path': frame['file_path']
        })

    return {
        'camera_angle_x': camera_angle_x,
        'frames': frames
    }
```

### Depth Error Computation
```python
# Source: Standard practice from KITTI benchmark and depth estimation literature
import numpy as np

def compute_depth_metrics(depth_gt, depth_pred, valid_mask=None):
    """
    Compute depth error metrics following standard benchmarks.

    Args:
        depth_gt: Ground truth depth (H, W)
        depth_pred: Predicted depth (H, W)
        valid_mask: Boolean mask (H, W) for valid pixels

    Returns:
        dict with mae, rmse, valid_pixels
    """
    if valid_mask is None:
        # Only compare pixels with non-zero depth in both maps
        valid_mask = (depth_gt > 0) & (depth_pred > 0)

    gt_valid = depth_gt[valid_mask]
    pred_valid = depth_pred[valid_mask]

    # Mean Absolute Error
    mae = np.mean(np.abs(gt_valid - pred_valid))

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'valid_pixels': int(valid_mask.sum()),
        'total_pixels': depth_gt.size
    }
```

### Quality Report JSON Structure
```python
# Example quality.json format
{
    "job_id": "abc123xyz",
    "model": "reconviagen",
    "status": "normal",  # normal/warning/failure
    "summary": "Reconstruction quality: Normal (PSNR 28.3 dB, SSIM 0.89)",

    "metrics": {
        "psnr": 28.34,
        "ssim": 0.8923,
        "depth_mae": 0.045,
        "depth_rmse": 0.068
    },

    "thresholds": {
        "psnr_normal": 25.0,
        "psnr_warning": 20.0,
        "ssim_normal": 0.85,
        "ssim_warning": 0.75
    },

    "metadata": {
        "timestamp": "2026-01-31T10:30:45Z",
        "input_file": "input.zip",
        "processing_duration_sec": 125.3,
        "image_resolution": [512, 512],
        "num_views": 6
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| skimage.measure.compare_psnr | skimage.metrics.peak_signal_noise_ratio | v0.16 (2019) | New module organization, same functionality |
| skimage.measure.compare_ssim | skimage.metrics.structural_similarity | v0.16 (2019) | New module organization, same functionality |
| Manual depth error loops | Vectorized numpy with masks | Standard practice | 10-100x faster, handles invalid pixels correctly |
| PyTorch3D rendering | nvdiffrast rendering | Phase 3.1 (2026) | Faster, CUDA 12.1 compatible, lower memory |

**Deprecated/outdated:**
- `skimage.measure.compare_psnr` and `skimage.measure.compare_ssim`: Moved to `skimage.metrics` in v0.16, old imports will fail in v0.26
- PyTorch3D for rendering: Removed in Phase 3.1 due to CUDA incompatibility, use nvdiffrast instead

## Open Questions

Things that couldn't be fully resolved:

1. **SSIM Threshold Values**
   - What we know: PSNR thresholds are decided (Normal ≥25dB, Warning 20-25dB, Failure <20dB)
   - What's unclear: Exact SSIM thresholds to complement PSNR
   - Recommendation: Based on literature, use SSIM Normal ≥0.85, Warning 0.75-0.85, Failure <0.75. These align with perceptual quality and correlate with PSNR thresholds in reconstruction tasks.

2. **Wireframe Line Width and Color**
   - What we know: User wants wireframe renderings, Claude has discretion on style
   - What's unclear: Optimal line width and color for visibility across different mesh complexities
   - Recommendation: Use line width 1-2 pixels (hardware default), black lines on white background for wireframe-only renders, or colored edges (based on edge length or curvature) for combined wireframe+texture views.

3. **Depth Metric Choice (MAE vs RMSE)**
   - What we know: User wants depth comparison for geometry validation
   - What's unclear: Which metric is more appropriate for this use case
   - Recommendation: Compute both MAE and RMSE. MAE is more interpretable (average pixel-wise error), RMSE penalizes large errors more heavily. Use MAE as primary, RMSE as secondary for detecting outliers.

4. **Number of Preview Angles**
   - What we know: User chose Claude's discretion for 4, 6, or 8 views
   - What's unclear: Optimal balance between comprehensive coverage and file size/processing time
   - Recommendation: Use **6 views** matching the 6 canonical orthogonal views (front, back, left, right, top, bottom) already used in transforms_train.json. This provides complete coverage, reuses existing camera poses, and aligns with industry standard "three orthogonal projections" extended to all six directions.

## Sources

### Primary (HIGH confidence)
- scikit-image 0.26.0 official documentation - https://scikit-image.org/docs/stable/api/skimage.metrics.html
- nvdiffrast official documentation - https://nvlabs.github.io/nvdiffrast/
- nvdiffrast GitHub repository (earth.py sample) - https://github.com/NVlabs/nvdiffrast/blob/main/samples/torch/earth.py
- trimesh official documentation - https://trimesh.org/trimesh.base.html
- Existing codebase: app/services/camera_estimation.py, app/services/mesh_export.py

### Secondary (MEDIUM confidence)
- KITTI depth estimation benchmark - https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion (depth metric standards)
- IEEE "Image Quality Metrics: PSNR vs. SSIM" - https://ieeexplore.ieee.org/document/5596999/ (threshold guidance)
- Monocular Depth Estimation Guide (HuggingFace 2026) - https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide (depth metric practices)
- Nature Scientific Reports "Similarity and quality metrics for MR image-to-image translation" 2025 - https://www.nature.com/articles/s41598-025-87358-0 (SSIM threshold best practices)

### Tertiary (LOW confidence)
- 3D rendering camera angle best practices - https://propertyrender.com/ultimate-guide-to-camera-angles-in-3d-rendering/ (preview angle recommendations, industry practice but not scientific)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - scikit-image and nvdiffrast are well-documented with official sources verified
- Architecture: HIGH - Patterns verified from official examples and existing codebase
- Pitfalls: HIGH - Common issues documented in official scikit-image docs and nvdiffrast GitHub issues
- Preview angles: MEDIUM - Recommendation based on existing system architecture (6 canonical views) and industry practice, but no scientific requirement
- SSIM thresholds: MEDIUM - Recommendation based on literature review but requires validation with real reconstruction data

**Research date:** 2026-01-31
**Valid until:** 2026-03-31 (60 days for stable domain - image quality metrics and rendering are mature fields)
