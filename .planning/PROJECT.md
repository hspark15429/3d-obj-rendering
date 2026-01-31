# 3D Object Reconstruction API

## What This Is

A Docker-based API system that generates textured 3D meshes from multi-view RGB images and depth data. It runs two AI reconstruction models (ReconViaGen via TRELLIS-VGGT and nvdiffrec), validates output quality with PSNR/SSIM metrics, and provides async job management for long-running inference tasks.

## Core Value

**End-to-end working pipeline**: Upload images, get back validated 3D meshes with quality reports. Everything runs with a single Docker command.

## Requirements

### Validated

- Accept ZIP upload with multi-view images (6 views) and depth renders — v1.0
- Run ReconViaGen model to generate textured mesh — v1.0
- Run nvdiffrec model to generate textured mesh — v1.0
- Allow user to select which model(s) to run — v1.0
- Generate static preview images of results (multiple angles) — v1.0
- Compute quality metrics (PSNR, SSIM) — v1.0
- Classify result status (normal/warning/failure) based on metrics — v1.0
- Save quality validation as JSON — v1.0
- POST /jobs — upload data, select model, return job ID — v1.0
- GET /jobs/{job_id} — return processing status — v1.0
- GET /jobs/{job_id}/download — download mesh, textures, previews, quality report — v1.0
- POST /jobs/{job_id}/cancel — cancel running job with two-step confirmation — v1.0
- Async job processing (queue-based, polling for status) — v1.0
- Docker deployment with GPU support — v1.0
- Single command to start entire system — v1.0
- Handle input validation errors gracefully — v1.0
- Handle model failures (OOM, convergence) gracefully — v1.0
- README.md with execution and API usage examples — v1.0
- architecture.md explaining design decisions — v1.0

### Active

(None — v1.0 milestone complete, awaiting v2.0 planning)

### Out of Scope

- Turntable video rendering — static images sufficient for requirements
- Web UI — API-only as specified
- Model training — inference only
- Multiple GPU support — single RTX 3090 target
- Production hardening (auth, rate limiting) — demo/assessment scope
- Webhook notifications — deferred to v2
- Batch job submission — deferred to v2

## Context

**Purpose:** Job application technical assessment demonstrating 3D reconstruction pipeline engineering.

**Current State (v1.0 shipped):**
- 7,601 lines of Python code
- 78 tests (1,944 lines of test code)
- 133 files in repository
- CUDA 12.1 + PyTorch 2.4.1 stack
- Real model implementations (TRELLIS-VGGT, nvdiffrec)

**Input data format:**
- `multi_views/`: 6 RGB images (front, back, left, right, top, bottom) at 2048x2048
- `depth_renders/`: 6 grayscale depth images at 1024x1024
- Camera positions: orthogonal axes (+x, -x, +y, -y, +z, -z), perspective projection

**Models:**
- ReconViaGen: Uses TRELLIS-VGGT pipeline from estheryang11/ReconViaGen
- nvdiffrec: NVIDIA differentiable rendering with optimization loop

**Hardware:** RTX 3090 (24GB VRAM) — sufficient for both models (16GB needed for TRELLIS).

## Constraints

- **Docker**: Must run in Docker with GPU passthrough (nvidia-docker)
- **Two models**: Both ReconViaGen and nvdiffrec must be functional
- **Async**: Long inference times require async job handling
- **Deliverables**: Source, README.md, architecture.md, example outputs required

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Async job queue over sync API | Model inference takes minutes; sync would timeout | Good |
| Static images over video preview | Simpler, meets requirements, saves time | Good |
| Single Docker compose setup | "Single command" requirement, easier deployment | Good |
| CUDA 12.1 over 12.4 | Better spconv-cu120 compatibility | Good |
| Pre-built flash-attn wheel | Source build takes hours and fails | Good |
| Two-step cancellation flow | Prevents accidental cancellations | Good |
| Quality failure = job failure | Clear failure semantics, per requirements | Good |
| Lazy imports via __getattr__ | Enables tests to run without torch/CUDA | Good |
| 17 structured error codes | Actionable errors improve debugging | Good |
| PSNR >= 25dB for "normal" quality | Industry standard threshold | Pending validation |
| SSIM >= 0.85 for "normal" quality | Industry standard threshold | Pending validation |

---
*Last updated: 2026-01-31 after v1.0 milestone*
