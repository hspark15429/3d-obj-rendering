---
phase: 02-job-pipeline
plan: 02
subsystem: api
tags: [pydantic, fastapi, file-validation, png, async-io, aiofiles, pytest]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: FastAPI app structure, Docker setup
provides:
  - Pydantic schemas for job API (submit, status, cancel, error responses)
  - File validation service with PNG magic byte checking
  - Upload file handler with async storage
  - Unit tests for file validation edge cases
affects: [02-03-endpoints, 02-04-celery-integration]

# Tech tracking
tech-stack:
  added: [aiofiles, pytest, pytest-asyncio]
  patterns: [Pydantic v2 model_config, async file I/O, magic byte validation, field-level error tracking]

key-files:
  created:
    - app/api/schemas.py
    - app/services/file_handler.py
    - tests/test_file_handler.py
  modified:
    - requirements.txt

key-decisions:
  - "Use Pydantic v2 syntax (model_config instead of Config class)"
  - "Validate PNG by magic bytes (not just file extension)"
  - "Two-step cancellation (confirm flag defaults to False)"
  - "Field-level error tracking in FileValidationError"
  - "Reset file pointers after validation for later reads"

patterns-established:
  - "JobStatus enum with 5 states: queued, processing, completed, failed, cancelled"
  - "ISO 8601 timestamps via datetime (automatic serialization)"
  - "File size limits: 20MB per file, 200MB total"
  - "Job directory structure: storage/jobs/{job_id}/{views,depth}/"
  - "Async file operations with aiofiles"

# Metrics
duration: 3min
completed: 2026-01-31
---

# Phase 2 Plan 2: API Schemas & File Validation Summary

**Pydantic v2 schemas with 5-state job lifecycle, PNG magic byte validation, and async file storage with 20MB/200MB limits**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-31T01:25:38Z
- **Completed:** 2026-01-31T01:28:31Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Type-safe API contracts with Pydantic v2 for all job endpoints
- Strict PNG validation by magic bytes (not just extension)
- File size enforcement (20MB per file, 200MB total upload)
- Async file storage with proper directory structure
- Comprehensive unit tests covering all validation edge cases
- File pointer reset after validation ensures files readable later

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Pydantic schemas for job API** - `da5ad25` (feat)
2. **Task 2: Create file handling service with PNG validation** - `a3ac63f` (feat)
3. **Task 3: Write unit tests for file validation** - `9ed3695` (test)

## Files Created/Modified

- `app/api/__init__.py` - API package marker
- `app/api/schemas.py` - All job-related Pydantic models (JobStatus enum, submit/status/cancel/error responses)
- `app/services/__init__.py` - Services package marker
- `app/services/file_handler.py` - File validation and storage with PNG magic byte checking, async I/O
- `tests/__init__.py` - Tests package marker
- `tests/test_file_handler.py` - 6 unit tests for file validation (wrong counts, non-PNG, oversized, success, pointer reset)
- `requirements.txt` - Added aiofiles, pytest, pytest-asyncio

## Decisions Made

1. **Pydantic v2 syntax** - Used model_config instead of deprecated Config class for future compatibility
2. **PNG magic byte validation** - Validate by reading first 8 bytes (`\x89PNG\r\n\x1a\n`) instead of trusting file extension
3. **Two-step cancellation** - CancelRequest.confirm defaults to False, requires explicit confirmation to prevent accidental cancellations
4. **Field-level error tracking** - FileValidationError includes field parameter to identify which specific file failed
5. **File pointer reset** - Critical: seek(0) after validation so files can be read again during save
6. **20MB/200MB limits** - 20MB per file handles large 2048x2048 PNGs, 200MB total prevents DoS

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed without errors.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for 02-03 (API endpoints):**
- Schemas defined for all job endpoints (submit, status, cancel)
- File validation ready for integration into upload endpoint
- Error response format standardized

**Ready for 02-04 (Celery integration):**
- JobStatus enum matches planned Celery task states
- File storage structure defined (storage/jobs/{job_id}/)
- Async file I/O compatible with Celery worker

**No blockers.** All validation logic tested and verified.

---
*Phase: 02-job-pipeline*
*Completed: 2026-01-31*
