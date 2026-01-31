# Output Format Reference

**Note:** Real outputs require running the pipeline with GPU. This document describes what to expect when downloading completed job results.

## Output Directory Structure

When you download results via `GET /jobs/{job_id}/download`, you receive a ZIP file with the following structure:

```
{job_id}/
  output/
    reconviagen/           # If ReconViaGen model selected
      mesh.glb             # Primary output: GLB format (includes textures)
      mesh.obj             # OBJ format (if conversion successful)
      mesh.mtl             # Material definition for OBJ
      texture_kd.png       # Diffuse texture map (if available)

    nvdiffrec/             # If nvdiffrec model selected
      mesh.glb             # Primary output: GLB format
      mesh.obj             # OBJ format (if conversion successful)
      mesh.mtl             # Material definition for OBJ
      texture_kd.png       # Diffuse texture map (if available)

    preview/               # Preview renders from all models
      front.png            # Front view preview (512×512)
      back.png             # Back view preview
      left.png             # Left view preview
      right.png            # Right view preview
      top.png              # Top view preview
      bottom.png           # Bottom view preview
      wireframe_front.png  # Wireframe overlay - front
      wireframe_back.png   # Wireframe overlay - back
      wireframe_left.png   # Wireframe overlay - left
      wireframe_right.png  # Wireframe overlay - right
      wireframe_top.png    # Wireframe overlay - top
      wireframe_bottom.png # Wireframe overlay - bottom

    quality.json           # Quality metrics and assessment
```

## File Formats

### 3D Mesh Formats

**GLB (Primary Format)**
- Industry-standard 3D format (glTF 2.0 binary)
- Embeds textures and materials in single file
- Widely supported (Blender, Unity, Three.js, etc.)
- Always available in output

**OBJ (Optional Format)**
- Classic 3D mesh format
- Separate files: `.obj` (geometry), `.mtl` (materials), `.png` (textures)
- Available if trimesh conversion succeeds
- May not be present if conversion fails (GLB still available)

### Preview Images

All preview images are **512×512 PNG** files showing the reconstructed mesh from 6 orthogonal viewpoints.

**Standard Previews:** Rendered mesh with materials and lighting
**Wireframe Previews:** Edge overlay on top of rendered mesh (green lines)

## Quality Assessment

The `quality.json` file contains metrics comparing the output preview renders against input images:

### Format

```json
{
  "status": "normal",
  "metrics": {
    "psnr": 28.5,
    "ssim": 0.89
  },
  "thresholds": {
    "psnr": {
      "normal": 25,
      "warning": 20
    },
    "ssim": {
      "normal": 0.85,
      "warning": 0.75
    }
  }
}
```

### Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio)**
- Measures reconstruction quality in decibels (dB)
- Higher is better
- **Normal:** ≥ 25 dB
- **Warning:** 20-25 dB
- **Failure:** < 20 dB

**SSIM (Structural Similarity Index)**
- Measures perceptual similarity (0.0 to 1.0)
- Higher is better
- **Normal:** ≥ 0.85
- **Warning:** 0.75-0.85
- **Failure:** < 0.75

### Status Classification

The overall status is determined by **BOTH** metrics (AND logic):

| Status    | Requirements                            |
|-----------|-----------------------------------------|
| `normal`  | PSNR ≥ 25 **AND** SSIM ≥ 0.85          |
| `warning` | PSNR ≥ 20 **AND** SSIM ≥ 0.75 (but not normal) |
| `failure` | Either metric below warning threshold   |

**Note:** If quality assessment returns `failure` status, the entire job is marked as FAILED and returns an error.

## Downloading Results

### Check Job Status

```bash
curl http://localhost:8000/jobs/{job_id}/status
```

Wait until `state` is `SUCCESS`.

### Download Output ZIP

```bash
curl -o output.zip http://localhost:8000/jobs/{job_id}/download
unzip output.zip
```

### What You Get

- **If model="reconviagen"**: Output in `{job_id}/output/reconviagen/`
- **If model="nvdiffrec"**: Output in `{job_id}/output/nvdiffrec/`
- **If model="both"**: Both model directories with their respective outputs
- **Always**: `preview/` directory with renders and `quality.json`

## Using the Meshes

### Blender

```bash
# Import GLB
File → Import → glTF 2.0 (.glb/.gltf)
```

### Three.js (Web)

```javascript
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('mesh.glb', (gltf) => {
  scene.add(gltf.scene);
});
```

### Python (trimesh)

```python
import trimesh

mesh = trimesh.load('mesh.glb')
# or
mesh = trimesh.load('mesh.obj')
```

## Troubleshooting

**Missing OBJ files?**
- GLB is always the primary output
- OBJ conversion is optional and may fail
- Use GLB format for maximum compatibility

**Quality warnings?**
- Check PSNR and SSIM values in quality.json
- Low scores may indicate input image issues
- Try different camera angles or better lighting

**Empty preview directory?**
- Preview generation failed (rare)
- Check job status for error messages
- Mesh files may still be valid

See the [API Documentation](../../docs/API.md) for complete endpoint reference and error codes.
