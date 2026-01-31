# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-30)

**Core value:** End-to-end working pipeline: Upload images, get back validated 3D meshes with quality reports. Everything runs with a single Docker command.
**Current focus:** Phase 3 - Model Integration (In Progress)

## Current Position

Phase: 3 of 6 (Model Integration)
Plan: 2 of 4 in current phase
Status: In progress
Last activity: 2026-01-31 - Completed 03-02-PLAN.md

Progress: [███░░░░░░░] 35% (7/20 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 5 min
- Total execution time: 0.6 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 1 | 12min | 12min |
| 2. Job Pipeline | 4 | 18min | 5min |
| 3. Model Integration | 2 | 5min | 3min |

**Recent Trend:**
- Last 5 plans: 3min, 3min, 8min, 3min, 2min
- Trend: Fast infrastructure plans, Docker config updates efficient

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

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-01-31
Stopped at: Completed 03-02-PLAN.md (Docker PyTorch Infrastructure)
Resume file: None
