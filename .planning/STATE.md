# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-30)

**Core value:** End-to-end working pipeline: Upload images, get back validated 3D meshes with quality reports. Everything runs with a single Docker command.
**Current focus:** Phase 4 - Quality & Preview (In progress)

## Current Position

Phase: 4 of 6 (Quality & Preview)
Plan: 1 of TBD in current phase
Status: In progress
Last activity: 2026-01-31 - Completed 04-01-PLAN.md (quality metrics service)

Progress: [████████░░] 76% (16/21 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: 5 min
- Total execution time: 1.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 1 | 12min | 12min |
| 2. Job Pipeline | 4 | 18min | 5min |
| 3. Model Integration | 4 | 16min | 4min |
| 3.1 CUDA 12 Upgrade | 6 | 26min | 4min |
| 4. Quality & Preview | 1 | 3min | 3min |

**Recent Trend:**
- Last 5 plans: 4min, 3min, 3min, 8min, 3min
- Trend: Consistent velocity, Phase 4 started

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Phase 1 (Foundation): Async job queue over sync API - Model inference takes minutes; sync would timeout
- Phase 1 (Foundation): Static images over video preview - Simpler, meets requirements, saves time
- Phase 1 (Foundation): Single Docker compose setup - "Single command" requirement, easier deployment
- 01-01: CUDA 11.8 over CUDA 12 - Broader compatibility with existing model checkpoints
- 01-01: Lifespan context manager over deprecated @app.on_event - Future-proof FastAPI code
- 01-01: Fail-fast GPU validation (12GB minimum) - ReconViaGen/nvdiffrec require substantial VRAM
- 01-01: Modern docker-compose GPU syntax - Forward-compatible with Compose V2
- 02-01: Dual Redis DB architecture - DB 0 for Celery broker/results, DB 1 for app state (cancellation flags)
- 02-01: Worker prefetch_multiplier=1 - Fair task distribution for long-running jobs
- 02-01: Visibility timeout 14400s - 4-hour timeout for model inference without premature requeue
- 02-01: Factory pattern for Celery app - Avoids circular imports with shared_task decorator
- 02-02: Pydantic v2 syntax - model_config instead of deprecated Config class
- 02-02: PNG magic byte validation - Validate by reading first 8 bytes instead of trusting extension
- 02-02: Two-step cancellation - CancelRequest.confirm defaults to False
- 02-02: Field-level error tracking - FileValidationError includes field parameter
- 02-02: File pointer reset - Critical seek(0) after validation for later reads
- 02-02: 20MB/200MB limits - 20MB per file, 200MB total prevents DoS
- 02-03: Two-step cancellation implementation - request_cancellation() sets pending flag, confirm_cancellation() activates it
- 02-03: 1-hour TTL on cancellation flags - Automatic cleanup of abandoned requests
- 02-03: Cancellation before each step - Check is_job_cancelled() before each of 6 processing steps for responsive cancellation
- 02-03: Progress tracking via update_state - PROGRESS state with meta={'progress': %, 'step': name}
- 02-03: File cleanup on cancel - delete_job_files() before returning cancelled status
- 02-04: apply_async with task_id=job_id - Match our job_id with Celery task ID for status lookup
- 02-04: Read content for size check - FastAPI UploadFile.seek() doesn't support whence parameter
- 02-04: Explicit task import in __init__.py - Required for Celery autodiscovery
- 03-01: Lazy pytorch3d import - Defer heavy import until function call
- 03-01: Deferred vram_manager import in base.py - Avoids circular dependency
- 03-01: Graceful CUDA absence - VRAM functions return error dict when CUDA unavailable
- 03-02: PyTorch CUDA installed first via --index-url wheels - Ensures correct CUDA version matching
- 03-02: PyTorch3D from pre-built wheel - Faster builds than source compilation
- 03-02: nvdiffrast from git - No stable PyPI release
- 03-02: 8GB shared memory for PyTorch - DataLoader multiprocessing needs /dev/shm
- 03-02: Separate model-weights volume - Checkpoints persist across image rebuilds
- 03-03: STUB implementations for both models - Official ReconViaGen code not released; enables integration testing
- 03-03: Different placeholder meshes (cube vs sphere) - Distinguish model outputs visually during testing
- 03-03: Progress reporting at multiple stages - Fine-grained feedback for long-running tasks
- 03-04: Sequential execution for 'both' mode - Run ReconViaGen first, then nvdiffrec with VRAM cleanup between
- 03-04: 2-hour soft timeout for 'both' mode - Extended timeout (7200s) for sequential model execution
- 03-04: Progress tracking includes model info - current_model field shows which model is running
- 03.1-01: CUDA 12.1.1 over 12.4 - better spconv-cu120 compatibility
- 03.1-01: PyTorch 2.4.1 with cu121 wheels - latest stable version
- 03.1-01: Removed PyTorch3D wheel - cu118 incompatible, use trimesh+nvdiffrast
- 03.1-01: TORCH_CUDA_ARCH_LIST 7.0-9.0 - includes Ada Lovelace and Hopper GPUs
- 03.1-02: Pre-built flash-attn wheel - source build takes hours and fails; use cu12+torch2.4 wheel
- 03.1-02: SPCONV_ALGO=native in Dockerfile and Python - critical for spconv to work with TRELLIS
- 03.1-02: Clone estheryang11/ReconViaGen with --recursive - includes TRELLIS submodule
- 03.1-02: Lazy TrellisPipelineWrapper loading - defers heavy imports to load() method
- 03.1-03: --no-build-isolation for tiny-cuda-nn - required for CUDA extension compilation
- 03.1-03: OpenGL camera convention - camera looks along -Z with Y-up for NeRF/nvdiffrec
- 03.1-03: Canonical 6-view camera layout - front/back/right/left/top/bottom at distance 2.5
- 03.1-03: Default focal length 1111.0 - produces ~26 degree FOV for object rendering
- 03.1-04: postprocessing_utils.to_glb() for mesh export - handles texture baking internally
- 03.1-04: GLB as primary output - trimesh converts to OBJ/PLY for compatibility
- 03.1-04: 16GB VRAM requirement for TRELLIS - needs ~14-16GB for full pipeline
- 03.1-04: Non-fatal OBJ/PLY conversion - GLB always available even if trimesh fails
- 03.1-05: 500 default iterations for nvdiffrec - balance between quality and speed
- 03.1-05: SimpleGeometry class - deformable sphere with learnable displacement when DMTet unavailable
- 03.1-05: nvdiffrast fallback - point splatting renderer when nvdiffrast unavailable
- 03.1-05: Spherical UV mapping - simple texture generation for sphere-based geometry
- 04-01: PSNR thresholds Normal>=25dB, Warning>=20dB, Failure<20dB
- 04-01: SSIM thresholds Normal>=0.85, Warning>=0.75, Failure<0.75
- 04-01: Both PSNR and SSIM must pass threshold for status level (AND logic)
- 04-01: data_range=1.0 always for float images, channel_axis=-1 for RGB

### Pending Todos

None yet.

### Roadmap Evolution

- Phase 3.1 inserted after Phase 3: CUDA 12 upgrade + real model integration (URGENT) - Prioritizing real model output over quality metrics

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-01-31T10:17:05Z
Stopped at: Completed 04-01-PLAN.md
Resume file: None
Next: Continue Phase 4 (04-02 preview generation)
