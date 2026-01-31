---
phase: 04
plan: 03
subsystem: quality-preview
tags: [preview, quality-metrics, mesh-rendering, png]

dependency-graph:
  requires: ["04-01", "04-02"]
  provides: ["preview-generator", "quality-report", "quality-pipeline"]
  affects: ["04-04"]

tech-stack:
  added: []
  patterns: ["preview-orchestration", "quality-pipeline"]

key-files:
  created:
    - app/services/preview_generator.py
  modified:
    - app/services/__init__.py

decisions:
  - id: "04-03-01"
    description: "Lazy renderer initialization - defer MeshRenderer creation to first access"

metrics:
  duration: 2min
  completed: 2026-01-31
---

# Phase 04 Plan 03: Preview Generator Summary

Preview generation service that renders preview images and computes quality reports by orchestrating mesh rendering for preview images (textured + wireframe from 6 angles) and quality metric computation.

## One-liner

Preview generator service with PreviewGenerator class for textured/wireframe renders and generate_quality_report() for quality.json output.

## Changes Made

### app/services/preview_generator.py (created - 556 lines)

Main preview generation service:

**PreviewGenerator class:**
- `__init__(renderer)` - stores or lazy-creates MeshRenderer instance
- `generate_previews(mesh_path, output_dir, camera_poses, resolution)` - renders textured and wireframe images for up to 6 camera poses, saves as PNG files
- `compute_quality_metrics(mesh_path, input_dir, camera_poses, resolution)` - computes PSNR/SSIM by comparing renders to input images, computes depth MAE/RMSE if depth maps available
- `generate_all(job_id, model, mesh_path, input_dir, output_dir, metadata)` - main entry point that orchestrates full quality pipeline
- `_get_input_resolution(input_dir, camera_poses)` - extracts resolution from first input image

**Standalone functions:**
- `generate_quality_report(job_id, model, metrics, metadata, output_path)` - generates quality.json with status, metrics, thresholds, metadata
- `generate_previews(mesh_path, output_dir, input_dir, resolution)` - convenience function for preview generation

**Quality report structure:**
```json
{
  "job_id": "abc123",
  "model": "reconviagen",
  "status": "normal",
  "summary": "Reconstruction quality: Normal (PSNR 28.3 dB, SSIM 0.89)",
  "metrics": {"psnr": 28.3, "ssim": 0.89, "depth_mae": 0.04, "depth_rmse": 0.06},
  "thresholds": {"psnr_normal": 25.0, "psnr_warning": 20.0, ...},
  "metadata": {"timestamp": "...", "input_file": "...", "resolution": [...]}
}
```

### app/services/__init__.py (modified)

Added exports for all quality/preview services:
- Quality metrics: `compute_psnr`, `compute_ssim`, `compute_depth_error`, `classify_quality_status`
- Mesh renderer: `MeshRenderer`, `load_camera_poses`, `build_mvp_matrix`
- Preview generator: `PreviewGenerator`, `generate_quality_report`, `generate_previews`

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Lazy renderer initialization | Defer heavy nvdiffrast context creation until actually needed |
| Resolution from input images | Match input resolution per CONTEXT.md requirement |
| Up to 6 preview angles | Use available camera poses, limit to 6 for consistency |
| Normalize depth for comparison | Input depth and rendered depth may have different scales |

## Artifacts Delivered

| Artifact | Path | Purpose |
|----------|------|---------|
| Preview generator service | app/services/preview_generator.py | Preview generation and quality report orchestration |
| Service exports | app/services/__init__.py | Clean import interface |

## Key Links Created

| From | To | Via | Purpose |
|------|-----|-----|---------|
| preview_generator.py | quality_metrics.py | import | compute_psnr, compute_ssim, classify_quality_status |
| preview_generator.py | mesh_renderer.py | import | MeshRenderer, build_mvp_matrix, load_camera_poses |

## Commits

| Hash | Type | Description |
|------|------|-------------|
| d7f793d | feat | Create preview generator service (556 lines) |
| d6c2140 | feat | Export quality services from app.services |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

All verification criteria passed:
- [x] preview_generator.py exists with all required functions
- [x] Python imports work from app.services
- [x] PreviewGenerator has generate_previews, compute_quality_metrics methods
- [x] generate_quality_report produces quality.json structure
- [x] generate_all orchestrates full pipeline
- [x] File exceeds minimum 150 lines (556 lines)

## Next Phase Readiness

Ready for 04-04 (Quality Pipeline Integration). The preview generator service provides all necessary components for integrating quality assessment into the job processing pipeline.
