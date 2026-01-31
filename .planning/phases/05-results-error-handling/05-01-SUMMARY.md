---
phase: 05-results-error-handling
plan: 01
subsystem: api
tags: [fastapi, error-handling, pydantic, exception-handlers]

# Dependency graph
requires:
  - phase: 02-job-pipeline
    provides: FileValidationError exception and jobs router
provides:
  - ErrorCode enum with 17 error codes
  - ERROR_SUGGESTIONS mapping for actionable hints
  - make_error_detail helper function
  - ErrorDetail and ErrorResponse schemas
  - Global exception handlers for consistent API errors
affects: [05-02, 05-03, all-future-api-work]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Structured error responses with code, message, details, suggestion
    - Global exception handlers for consistent formatting
    - HTTP status code semantics (422 validation, 409 conflict, 410 gone)

key-files:
  created:
    - app/api/error_codes.py
  modified:
    - app/api/schemas.py
    - app/main.py

key-decisions:
  - "17 error codes covering validation, not found, conflict, gone, model, and resource errors"
  - "Specific error code detection from message patterns (FILE_TOO_LARGE, INVALID_FILE_FORMAT, etc.)"
  - "422 status for validation errors (was 400, now semantically correct)"
  - "Generic Exception handler logs errors but returns safe message (no internal details exposed)"

patterns-established:
  - "make_error_detail(code, message, details) for building error responses"
  - "HTTPException with structured detail dict for API errors"
  - "Error suggestions lookup from ERROR_SUGGESTIONS mapping"

# Metrics
duration: 2min
completed: 2026-01-31
---

# Phase 5 Plan 1: Error Code Infrastructure Summary

**Centralized error taxonomy with 17 error codes, structured ErrorDetail schema, and 4 global exception handlers for consistent API error responses**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-31T11:01:07Z
- **Completed:** 2026-01-31T11:03:28Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments
- ErrorCode enum with 17 error codes covering all failure scenarios (validation, not found, conflict, gone, model failures, resource errors)
- ERROR_SUGGESTIONS dict providing actionable hints for each error code (e.g., FILE_TOO_LARGE tells users to "Reduce image resolution or compress PNGs before uploading")
- make_error_detail helper that builds structured error dicts with automatic suggestion lookup
- ErrorDetail schema with code, message, details, suggestion fields per CONTEXT.md requirements
- 4 global exception handlers ensuring all API errors return consistent structured JSON

## Task Commits

Each task was committed atomically:

1. **Task 1: Create error code taxonomy** - `1ca49fb` (feat)
2. **Task 2: Add error schemas and global exception handlers** - `ca3dd74` (feat)

## Files Created/Modified
- `app/api/error_codes.py` - ErrorCode enum, ERROR_SUGGESTIONS dict, make_error_detail helper
- `app/api/schemas.py` - ErrorDetail and ErrorResponse Pydantic schemas
- `app/main.py` - 4 global exception handlers (FileValidationError, HTTPException, RequestValidationError, Exception)

## Decisions Made
- **17 error codes defined:** Comprehensive coverage from RESEARCH.md taxonomy including VALIDATION_FAILED, FILE_TOO_LARGE, INVALID_FILE_FORMAT, INVALID_FILE_COUNT, JOB_NOT_FOUND, JOB_NOT_READY, JOB_EXPIRED, MODEL_FAILED, MODEL_OOM, MODEL_VRAM_OOM, MODEL_CONVERGENCE_FAILED, QUALITY_THRESHOLD_FAILED, INCOMPLETE_RESULTS, GPU_UNAVAILABLE, DISK_FULL, MEMORY_EXHAUSTED, UNKNOWN_ERROR
- **Smart error code detection:** FileValidationError handler examines message patterns to select specific codes (FILE_TOO_LARGE if "too large" in message, INVALID_FILE_FORMAT if "format" or "png" in message)
- **422 status for validation:** Changed from 400 to 422 for semantic correctness per HTTP standards
- **Safe generic handler:** Unhandled exceptions log full details but return safe message without exposing internals

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Error infrastructure ready for use in download endpoint (05-02)
- All HTTPException handlers can now use structured detail dicts
- make_error_detail helper available for consistent error creation across endpoints

---
*Phase: 05-results-error-handling*
*Completed: 2026-01-31*
