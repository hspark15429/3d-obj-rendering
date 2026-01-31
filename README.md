# 3D Object Reconstruction API

Transform multi-view images into high-quality 3D meshes using state-of-the-art AI models. Upload 6 RGB images and depth renders, get back textured meshes with quality validation.

## What It Does

This system runs two AI reconstruction models (ReconViaGen and nvdiffrec) to generate textured 3D meshes from multi-view images. It handles long-running inference tasks asynchronously, validates output quality with PSNR and SSIM metrics, and provides preview renders from multiple angles. Submit a job via REST API, poll for status, and download complete results including meshes, textures, previews, and quality reports.

## Prerequisites

Before running this system, ensure you have:

- **Docker** and **Docker Compose** installed
- **NVIDIA GPU** with 16GB+ VRAM (RTX 3090 or better recommended)
- **NVIDIA Container Toolkit** (nvidia-docker) for GPU passthrough
- **~30GB disk space** for model weights and outputs
- **8GB+ system RAM** for PyTorch data loading

## Quick Start

```bash
git clone https://github.com/hspark15429/3d-obj-rendering.git
cd 3d-obj-rendering
docker-compose up --build
```

**Note:** First build downloads ~10GB of model weights (TRELLIS, nvdiffrec checkpoints). This may take 15-30 minutes depending on network speed.

The API will be available at `http://localhost:8000` once the health check passes.

## Usage

### 1. Check System Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "vram_total": "24GB",
  "vram_free": "22GB"
}
```

### 2. Submit a Reconstruction Job

```bash
curl -X POST http://localhost:8000/jobs/ \
  -F "views=@input/multi_views/front.png" \
  -F "views=@input/multi_views/back.png" \
  -F "views=@input/multi_views/left.png" \
  -F "views=@input/multi_views/right.png" \
  -F "views=@input/multi_views/top.png" \
  -F "views=@input/multi_views/bottom.png" \
  -F "depth_renders=@input/depth_renders/front.png" \
  -F "depth_renders=@input/depth_renders/back.png" \
  -F "depth_renders=@input/depth_renders/left.png" \
  -F "depth_renders=@input/depth_renders/right.png" \
  -F "depth_renders=@input/depth_renders/top.png" \
  -F "depth_renders=@input/depth_renders/bottom.png" \
  -F "model=reconviagen"
```

Response:
```json
{
  "job_id": "abc123def456",
  "status": "PENDING",
  "message": "Job submitted successfully"
}
```

**Model options:**
- `reconviagen` - TRELLIS-VGGT model (faster, good for general objects)
- `nvdiffrec` - nvdiffrec differentiable renderer (slower, higher quality)

### 3. Check Job Status

```bash
curl http://localhost:8000/jobs/abc123def456
```

Response (in progress):
```json
{
  "job_id": "abc123def456",
  "status": "PROGRESS",
  "progress": 45,
  "step": "Mesh generation",
  "model": "reconviagen"
}
```

Response (completed):
```json
{
  "job_id": "abc123def456",
  "status": "SUCCESS",
  "progress": 100,
  "result": {
    "mesh_path": "storage/jobs/abc123def456/output/mesh.obj",
    "preview_paths": [
      "storage/jobs/abc123def456/output/preview_0.png",
      "storage/jobs/abc123def456/output/preview_1.png",
      "storage/jobs/abc123def456/output/preview_2.png"
    ],
    "quality": {
      "status": "normal",
      "psnr": 28.5,
      "ssim": 0.92
    }
  }
}
```

### 4. Download Results

```bash
curl http://localhost:8000/jobs/abc123def456/download -o results.zip
```

The ZIP file contains:
- `mesh.obj` - 3D mesh geometry
- `material.mtl` - Material definitions
- `texture_*.png` - Texture images
- `preview_*.png` - Multi-angle preview renders
- `quality.json` - Quality metrics and status

### Cancelling Jobs

```bash
# Request cancellation (two-step process)
curl -X POST http://localhost:8000/jobs/abc123def456/cancel

# Confirm cancellation
curl -X POST http://localhost:8000/jobs/abc123def456/cancel \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'
```

## API Reference

For complete endpoint documentation, see **[docs/API.md](docs/API.md)**.

Key endpoints:
- `GET /health` - System health and GPU status
- `POST /jobs` - Submit reconstruction job
- `GET /jobs/{job_id}` - Poll job status
- `POST /jobs/{job_id}/cancel` - Cancel running job
- `GET /jobs/{job_id}/download` - Download results ZIP

Full error code reference (17 codes) available in API documentation.

## Architecture

This system uses:
- **FastAPI** for the REST API
- **Celery** for async task queue (handles long-running model inference)
- **Redis** as message broker and state store
- **ReconViaGen** (TRELLIS-VGGT) for neural reconstruction
- **nvdiffrec** for differentiable rendering refinement

For detailed architecture diagrams and design decisions, see **[docs/architecture.md](docs/architecture.md)**.

## Input Format

The system expects:

**Multi-view RGB images** (`multi_views/`):
- 6 PNG images: `front.png`, `back.png`, `left.png`, `right.png`, `top.png`, `bottom.png`
- Resolution: 2048x2048 pixels
- Format: RGB PNG
- Camera: Perspective projection from orthogonal axes

**Depth renders** (`depth_renders/`):
- 6 grayscale PNG images matching multi-view names
- Resolution: 1024x1024 pixels
- Format: Grayscale PNG (depth encoded as intensity)

Example input files are provided in `input/multi_views/` and `input/depth_renders/`.

## Output Format

Each completed job produces:

**3D Mesh:**
- `mesh.obj` - Wavefront OBJ geometry file
- `material.mtl` - Material definitions with texture references
- Optional: `mesh.glb` (GLB format for web viewing)

**Textures:**
- `texture_kd.png` - Diffuse color texture
- `texture_ks.png` - Specular texture (if available)
- `texture_normal.png` - Normal map (if available)

**Preview Images:**
- `preview_0.png` through `preview_7.png` - 8 rendered views at 45-degree intervals
- Resolution: 512x512 pixels
- Shows mesh with textures and wireframe overlay

**Quality Report:**
- `quality.json` - Contains:
  - `status`: "normal", "warning", or "failure"
  - `psnr`: Peak Signal-to-Noise Ratio (dB)
  - `ssim`: Structural Similarity Index (0-1)
  - `psnr_expected`: Threshold for current status
  - `ssim_expected`: Threshold for current status

**Quality Thresholds:**
- **Normal**: PSNR ≥ 25dB AND SSIM ≥ 0.85
- **Warning**: PSNR ≥ 20dB AND SSIM ≥ 0.75
- **Failure**: PSNR < 20dB OR SSIM < 0.75

## Development

### Running Tests

```bash
# Run all tests
docker-compose exec api pytest

# Run with verbose output
docker-compose exec api pytest -v

# Run specific test file
docker-compose exec api pytest app/tests/test_validation.py
```

### Viewing Logs

```bash
# Follow all service logs
docker-compose logs -f

# API logs only
docker-compose logs -f api

# Worker logs only
docker-compose logs -f worker
```

### Troubleshooting

**GPU not detected:**
```bash
# Verify nvidia-docker installation
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Check GPU allocation
docker-compose exec api nvidia-smi
```

**Out of memory errors:**
- Reduce concurrent workers in `docker-compose.yml` (default: 1)
- Ensure no other GPU processes are running
- Verify 16GB+ VRAM available

**Slow first run:**
- Model weights (~10GB) download on first build
- CUDA compilation for nvdiffrast takes 5-10 minutes
- Subsequent starts are fast (<30 seconds)

## License

This project was developed as a technical assessment for a job application. Use for evaluation and reference purposes.

## Citation

If you use this system, please cite the underlying models:

**TRELLIS (ReconViaGen):**
```
@article{trellis2024,
  title={TRELLIS: Scalable Text-to-3D with Triplane Latent Diffusion},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

**nvdiffrec:**
```
@article{munkberg2022nvdiffrec,
  title={Extracting Triangular 3D Models, Materials, and Lighting From Images},
  author={Munkberg, Jacob and Hasselgren, Jon and Shen, Tianchang and Gao, Jun and Chen, Wenzheng and Evans, Alex and M{\"u}ller, Thomas and Fidler, Sanja},
  journal={CVPR},
  year={2022}
}
```
