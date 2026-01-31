# Input Format Reference

This directory contains sample input files you can use to test the 3D reconstruction pipeline.

## Input Requirements

The API expects two sets of multi-view images:

### 1. Color Views (`multi_views/`)

Six PNG images showing the object from orthogonal viewpoints at **2048×2048 resolution**:

- `front.png` - Front view (looking along -Z axis)
- `back.png` - Back view (looking along +Z axis)
- `left.png` - Left view (looking along +X axis)
- `right.png` - Right view (looking along -X axis)
- `top.png` - Top view (looking along -Y axis)
- `bottom.png` - Bottom view (looking along +Y axis)

**Camera Setup:**
- Position: Distance 2.5 from origin
- Projection: Perspective with focal length 1111.0 (~26° FOV)
- Convention: OpenGL camera (looks along -Z, Y-up)
- Format: PNG, RGB or RGBA

### 2. Depth Views (`depth_renders/`)

Six PNG depth maps corresponding to the color views at **1024×1024 resolution**:

- Same file names: `front.png`, `back.png`, `left.png`, `right.png`, `top.png`, `bottom.png`
- Format: Grayscale PNG (depth encoded as intensity)
- Coordinate frame: Matches corresponding color view

## File Naming Convention

**Critical:** File names must exactly match the expected view names. The pipeline uses these names to determine camera positions.

## Using These Samples

### Submit a Reconstruction Job

```bash
curl -X POST http://localhost:8000/jobs \
  -F "multi_views=@examples/input/multi_views/front.png" \
  -F "multi_views=@examples/input/multi_views/back.png" \
  -F "multi_views=@examples/input/multi_views/left.png" \
  -F "multi_views=@examples/input/multi_views/right.png" \
  -F "multi_views=@examples/input/multi_views/top.png" \
  -F "multi_views=@examples/input/multi_views/bottom.png" \
  -F "depth_renders=@examples/input/depth_renders/front.png" \
  -F "depth_renders=@examples/input/depth_renders/back.png" \
  -F "depth_renders=@examples/input/depth_renders/left.png" \
  -F "depth_renders=@examples/input/depth_renders/right.png" \
  -F "depth_renders=@examples/input/depth_renders/top.png" \
  -F "depth_renders=@examples/input/depth_renders/bottom.png" \
  -F "model=reconviagen"
```

The API will return a `job_id` you can use to check status and download results.

See the main [API Documentation](../../docs/API.md) for complete endpoint reference.
