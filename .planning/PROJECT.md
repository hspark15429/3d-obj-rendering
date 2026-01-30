# 3D Object Reconstruction API

## What This Is

A Docker-based API system that generates textured 3D meshes from multi-view RGB images and depth data. It runs two AI reconstruction models (ReconViaGen and nvdiffrec), validates output quality with quantitative metrics, and provides async job management for the long-running inference tasks.

## Core Value

**End-to-end working pipeline**: Upload images, get back validated 3D meshes with quality reports. Everything runs with a single Docker command.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Accept ZIP upload with multi-view images (6 views) and depth renders
- [ ] Run ReconViaGen model to generate textured mesh
- [ ] Run nvdiffrec model to generate textured mesh
- [ ] Allow user to select which model(s) to run
- [ ] Generate static preview images of results (multiple angles)
- [ ] Compute quality metrics (PSNR, SSIM, or mesh quality)
- [ ] Classify result status (normal/warning/failure) based on metrics
- [ ] Save quality validation as JSON
- [ ] POST /jobs — upload data, select model, return job ID
- [ ] GET /jobs/{job_id} — return processing status
- [ ] GET /jobs/{job_id}/result — download mesh, textures, previews, quality report
- [ ] Async job processing (queue-based, polling for status)
- [ ] Docker deployment with GPU support
- [ ] Single command to start entire system
- [ ] Handle input validation errors gracefully
- [ ] Handle model failures (OOM, convergence) gracefully
- [ ] README.md with execution and API usage examples
- [ ] architecture.md explaining design decisions

### Out of Scope

- Turntable video rendering — static images sufficient for requirements
- Web UI — API-only as specified
- Model training — inference only
- Multiple GPU support — single RTX 3090 target
- Production hardening (auth, rate limiting) — demo/assessment scope

## Context

**Purpose:** Job application technical assessment demonstrating 3D reconstruction pipeline engineering.

**Input data format:**
- `multi_views/`: 6 RGB images (front, back, left, right, top, bottom) at 2048x2048
- `depth_renders/`: 6 grayscale depth images at 1024x1024
- Camera positions: orthogonal axes (+x, -x, +y, -y, +z, -z), perspective projection

**Models:**
- ReconViaGen: https://github.com/estheryang11/ReconViaGen
- nvdiffrec: https://github.com/NVlabs/nvdiffrec

**Hardware:** RTX 3090 (24GB VRAM) — sufficient for both models.

**Timeline:** Days, not weeks. Priority is working end-to-end over polish.

## Constraints

- **Docker**: Must run in Docker with GPU passthrough (nvidia-docker)
- **Two models**: Both ReconViaGen and nvdiffrec must be functional
- **Async**: Long inference times require async job handling
- **Deliverables**: Source, README.md, architecture.md, example outputs required

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Async job queue over sync API | Model inference takes minutes; sync would timeout | — Pending |
| Static images over video preview | Simpler, meets requirements, saves time | — Pending |
| Single Docker compose setup | "Single command" requirement, easier deployment | — Pending |

---
*Last updated: 2026-01-30 after initialization*
