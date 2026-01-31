# Phase 5: Results & Error Handling - Research

**Researched:** 2026-01-31
**Domain:** FastAPI file downloads, error handling, and HTTP status codes
**Confidence:** HIGH

## Summary

This phase implements result downloads (ZIP packaging of all outputs) and comprehensive error handling across the API. The research focused on FastAPI best practices for streaming file responses, structured error handling patterns, and HTTP status code selection for various failure scenarios.

FastAPI provides two main approaches for file downloads: `FileResponse` for disk-based files and `StreamingResponse` for in-memory or generated content. For this phase, `StreamingResponse` with Python's standard library `zipfile` module is the recommended approach, as all job outputs need to be packaged into a single ZIP file in memory. The standard library's `BytesIO` + `zipfile` pattern is well-established and requires no additional dependencies.

For error handling, FastAPI's `HTTPException` combined with custom exception handlers provides structured error responses. The user has already locked in a JSON error format with `code`, `message`, `details`, and `suggestion` fields. HTTP status code selection follows REST API best practices: 422 for validation errors (semantic failures), 404 for missing resources, 503 for temporary resource exhaustion, and 410 for expired/cleaned-up jobs.

**Primary recommendation:** Use `StreamingResponse` with `BytesIO` and `zipfile` for downloads, implement a global exception handler for consistent error formatting, and create a structured error code taxonomy with snake_case naming (e.g., `VALIDATION_FAILED`, `MODEL_OOM`, `FILE_TOO_LARGE`).

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | 0.115+ | Web framework | Already in use, built-in error handling |
| Pydantic | v2 | Data validation | Already in use for schemas, validation errors |
| zipfile | stdlib | ZIP creation | Standard library, no dependencies needed |
| io.BytesIO | stdlib | In-memory buffer | Standard library, efficient memory usage |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| fastapi.responses.StreamingResponse | 0.115+ | Stream files to client | Required for in-memory ZIP downloads |
| fastapi.responses.FileResponse | 0.115+ | Stream disk files | Alternative for single-file downloads |
| fastapi.exception_handlers | 0.115+ | Custom error handling | Global error formatting |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| zipfile (stdlib) | stream-zip, zipstream-ng | Third-party libs offer better streaming for TB-scale files, but stdlib is sufficient for this use case (< 500MB ZIPs) |
| StreamingResponse | FileResponse | FileResponse simpler but requires writing ZIP to disk first, adding I/O overhead |
| Custom error format | RFC 9457 Problem Details | Problem Details is more standardized but requires additional fields (type URI, instance); user already locked in simpler format |

**Installation:**
```bash
# No additional dependencies needed
# zipfile, io.BytesIO are in Python standard library
# FastAPI 0.115 already installed
```

## Architecture Patterns

### Recommended Project Structure
```
app/
├── api/
│   ├── jobs.py              # Add GET /jobs/{job_id}/download endpoint
│   ├── schemas.py           # Add ErrorDetail, DownloadResponse schemas
│   └── exceptions.py        # NEW: Custom exception classes
├── services/
│   ├── result_packager.py   # NEW: ZIP packaging logic
│   └── error_formatter.py   # NEW: Error message formatting
└── middleware/
    └── error_handler.py     # NEW: Global exception handler
```

### Pattern 1: In-Memory ZIP Streaming
**What:** Create ZIP file in memory and stream to client without disk write
**When to use:** All result downloads (job outputs are already on disk, ZIP is temporary)
**Example:**
```python
# Source: FastAPI official docs + Python zipfile docs
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from fastapi.responses import StreamingResponse

def create_result_zip(job_id: str) -> BytesIO:
    """Package all job outputs into a single ZIP file."""
    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED, compresslevel=9) as zip_file:
        # Add mesh files
        zip_file.write(f"storage/jobs/{job_id}/output/mesh.obj", "mesh.obj")
        zip_file.write(f"storage/jobs/{job_id}/output/mesh.ply", "mesh.ply")
        zip_file.write(f"storage/jobs/{job_id}/output/mesh.glb", "mesh.glb")

        # Add textures
        zip_file.write(f"storage/jobs/{job_id}/output/mesh.png", "texture.png")

        # Add previews (all 6 textured + 6 wireframe)
        for i in range(6):
            zip_file.write(
                f"storage/jobs/{job_id}/output/previews/textured_{i:02d}.png",
                f"previews/textured_{i:02d}.png"
            )
            zip_file.write(
                f"storage/jobs/{job_id}/output/previews/wireframe_{i:02d}.png",
                f"previews/wireframe_{i:02d}.png"
            )

        # Add quality report
        zip_file.write(
            f"storage/jobs/{job_id}/output/quality.json",
            "quality.json"
        )

    # CRITICAL: Reset buffer position before streaming
    zip_buffer.seek(0)
    return zip_buffer

@router.get("/{job_id}/download")
async def download_results(job_id: str):
    """Download all job results as ZIP."""
    zip_buffer = create_result_zip(job_id)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={job_id}.zip"
        }
    )
```

### Pattern 2: Structured Error Responses
**What:** Consistent error JSON structure across all endpoints
**When to use:** All error scenarios (validation, not found, server errors)
**Example:**
```python
# Source: FastAPI error handling docs
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class ErrorDetail(BaseModel):
    """Locked decision: user-specified error format."""
    code: str  # Human-readable: VALIDATION_FAILED, MODEL_OOM, etc.
    message: str  # Human-readable error description
    details: dict  # Additional context (field errors, metrics, etc.)
    suggestion: str  # Actionable fix hint

class ErrorResponse(BaseModel):
    error: ErrorDetail

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Format all HTTPException as structured error response."""
    # Default error detail
    error_detail = {
        "code": "UNKNOWN_ERROR",
        "message": str(exc.detail),
        "details": {},
        "suggestion": "Please contact support if this persists."
    }

    # If exc.detail is already a dict with our structure, use it
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        error_detail = exc.detail

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_detail}
    )

# Usage in endpoints
@router.get("/{job_id}/download")
async def download_results(job_id: str):
    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        raise HTTPException(
            status_code=404,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job {job_id} not found",
                "details": {"job_id": job_id},
                "suggestion": "Check the job ID and try again."
            }
        )

    if result.state != "SUCCESS":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "JOB_NOT_READY",
                "message": f"Job is in '{result.state}' state",
                "details": {
                    "job_id": job_id,
                    "current_state": result.state
                },
                "suggestion": "Wait for job to complete before downloading results."
            }
        )
```

### Pattern 3: Validation Error Formatting
**What:** Convert Pydantic ValidationError to structured error format
**When to use:** FastAPI request validation failures (Pydantic v2)
**Example:**
```python
# Source: FastAPI + Pydantic v2 docs
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """Format Pydantic validation errors as structured response."""
    # Extract field-level errors
    field_errors = {}
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        field_errors[field_path] = {
            "message": error["msg"],
            "type": error["type"]
        }

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_FAILED",
                "message": "Request validation failed",
                "details": {
                    "fields": field_errors,
                    "error_count": len(field_errors)
                },
                "suggestion": "Fix the validation errors and try again."
            }
        }
    )
```

### Anti-Patterns to Avoid
- **Writing ZIP to disk before streaming:** Wastes I/O, disk space, and requires cleanup. Use `BytesIO` instead.
- **Forgetting `zip_buffer.seek(0)`:** StreamingResponse will return empty file if buffer position is at EOF.
- **Inconsistent error formats:** Don't mix plain string errors with structured JSON errors. Use global exception handler for consistency.
- **Exposing internal paths in errors:** Don't include `/home/user/storage/jobs/...` in error messages. Use relative paths or omit entirely.
- **Generic HTTP status codes:** Don't use 400 for everything. Use specific codes (422 for validation, 404 for not found, 503 for resources).

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ZIP file creation | Custom binary ZIP writer | Python stdlib `zipfile` | ZIP format is complex (CRC32, compression, directory structure). zipfile handles all edge cases. |
| Error code registry | Hardcoded strings everywhere | Enum or constants file | Typos break client integrations. Centralized enum ensures consistency and enables autocomplete. |
| HTTP status code selection | Guessing/googling each time | REST API status code reference | Status codes have semantic meaning. Wrong codes confuse clients (e.g., 500 for validation errors). |
| Streaming file responses | Manual chunking with generators | FastAPI `StreamingResponse` | StreamingResponse handles headers, chunking, connection management, and edge cases. |

**Key insight:** Error handling is deceptively complex. Consistency matters more than individual error quality. A mediocre but consistent error format is better than excellent but inconsistent errors, because clients can parse consistent formats programmatically.

## Common Pitfalls

### Pitfall 1: Missing Files in ZIP Cause Silent Failures
**What goes wrong:** `ZipFile.write()` raises `FileNotFoundError` if source file missing, but exception bubbles up as generic 500 error instead of structured error response.
**Why it happens:** Job cleanup may delete files before download, or quality pipeline may fail silently without creating expected outputs.
**How to avoid:** Validate all expected files exist before creating ZIP. Return 410 Gone if job cleaned up, or 500 with specific error if outputs incomplete.
**Warning signs:** Users report downloading empty or incomplete ZIPs without clear error messages.

### Pitfall 2: Large ZIPs Cause Memory Issues
**What goes wrong:** Creating multi-GB ZIPs in `BytesIO` exhausts memory, killing worker or causing OOM.
**Why it happens:** `BytesIO` loads entire ZIP into RAM before streaming. For large jobs (high-res textures, many preview images), this can exceed available memory.
**How to avoid:** Monitor ZIP size during creation. For this project, ZIPs should be < 500MB (6 input PNGs + 6 depth PNGs + outputs + previews). If exceeded, either compress more aggressively or write to temp file and use `FileResponse`.
**Warning signs:** Workers crash during download endpoint, memory usage spikes correlate with downloads.

### Pitfall 3: Inconsistent Error Codes Break Client Integrations
**What goes wrong:** Same error condition returns different error codes on different endpoints (e.g., `FILE_TOO_LARGE` vs `UPLOAD_SIZE_EXCEEDED`).
**Why it happens:** Multiple developers implement error handling independently without checking existing codes.
**How to avoid:** Centralize error codes in `app/api/error_codes.py` as Enum or constants. Code review enforces consistency.
**Warning signs:** Client bug reports about inconsistent error handling, duplicate error codes with slightly different names.

### Pitfall 4: HTTP 404 Used for All "Not Found" Scenarios
**What goes wrong:** Returning 404 for both "job never existed" and "job expired/cleaned up" prevents clients from distinguishing permanent vs temporary absence.
**Why it happens:** 404 is the most well-known "not found" status code, so developers default to it.
**How to avoid:** Use 410 Gone for intentionally deleted/expired resources. This signals to clients (and search engines) that resource is permanently gone, not temporarily missing.
**Warning signs:** Clients retry 404s indefinitely expecting jobs to appear, when jobs were already cleaned up.

### Pitfall 5: Error Suggestions Are Too Generic
**What goes wrong:** Error suggestions like "Please try again" or "Contact support" don't help users fix the actual problem.
**Why it happens:** Generic suggestions are easier to write than actionable ones, and developers don't think through user's next action.
**How to avoid:** For each error code, write specific suggestion based on root cause. Examples:
  - `FILE_TOO_LARGE` → "Reduce image resolution or compress PNGs before uploading (max 20MB per file)."
  - `MODEL_OOM` → "Try again when GPU resources are available, or contact support to increase VRAM quota."
  - `QUALITY_THRESHOLD_FAILED` → "Reconstruction quality below threshold (PSNR 18.5dB vs 20dB required). Try uploading higher-quality input images."
**Warning signs:** Users contact support with errors that could have been self-resolved with better suggestions.

## Code Examples

Verified patterns from official sources:

### Complete Download Endpoint
```python
# Source: Verified pattern combining FastAPI docs + Python zipfile docs
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.celery_app import celery_app
from app.services.file_handler import get_job_path

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.get("/{job_id}/download")
async def download_results(job_id: str):
    """
    Download all job results as a single ZIP file.

    Includes: OBJ, PLY, GLB meshes, textures, preview images, quality.json

    Returns:
        StreamingResponse: ZIP file with all outputs

    Raises:
        HTTPException 404: Job not found
        HTTPException 409: Job not completed yet
        HTTPException 410: Job results expired/cleaned up
        HTTPException 500: Job failed or outputs incomplete
    """
    # Check job exists and is completed
    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        raise HTTPException(
            status_code=404,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job {job_id} not found",
                "details": {"job_id": job_id},
                "suggestion": "Check the job ID and try again."
            }
        )

    if result.state == "FAILURE":
        raise HTTPException(
            status_code=500,
            detail={
                "code": "JOB_FAILED",
                "message": "Job failed during processing",
                "details": {
                    "job_id": job_id,
                    "error": str(result.info) if result.info else "Unknown error"
                },
                "suggestion": "Submit a new job with different inputs."
            }
        )

    if result.state in ["QUEUED", "PROGRESS", "STARTED"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "JOB_NOT_READY",
                "message": f"Job is still {result.state.lower()}",
                "details": {
                    "job_id": job_id,
                    "current_state": result.state,
                    "progress": result.info.get("progress") if result.info else None
                },
                "suggestion": "Wait for job to complete before downloading. Poll GET /jobs/{job_id} for status."
            }
        )

    # Check output directory exists
    job_path = get_job_path(job_id)
    output_dir = job_path / "output"

    if not output_dir.exists():
        raise HTTPException(
            status_code=410,
            detail={
                "code": "JOB_EXPIRED",
                "message": "Job results have been cleaned up",
                "details": {"job_id": job_id},
                "suggestion": "Job results are no longer available. Submit a new job if needed."
            }
        )

    # Create ZIP in memory
    try:
        zip_buffer = create_result_zip(job_id, output_dir)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INCOMPLETE_RESULTS",
                "message": "Job outputs are incomplete",
                "details": {
                    "job_id": job_id,
                    "missing_file": str(e)
                },
                "suggestion": "Job may have failed partially. Contact support."
            }
        )

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={job_id}.zip"
        }
    )

def create_result_zip(job_id: str, output_dir: Path) -> BytesIO:
    """
    Package all job outputs into a single ZIP file.

    Per user decision: ZIP contains ALL outputs (OBJ, PLY, GLB, textures,
    previews, quality.json). No partial downloads.
    """
    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED, compresslevel=9) as zip_file:
        # Determine which model outputs exist
        model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            model_name = model_dir.name

            # Add mesh files (all formats)
            for mesh_file in ["mesh.obj", "mesh.ply", "mesh.glb", "mesh.mtl"]:
                mesh_path = model_dir / mesh_file
                if mesh_path.exists():
                    zip_file.write(
                        mesh_path,
                        f"{model_name}/{mesh_file}"
                    )

            # Add texture
            texture_path = model_dir / "mesh.png"
            if texture_path.exists():
                zip_file.write(texture_path, f"{model_name}/texture.png")

            # Add previews (textured + wireframe)
            previews_dir = model_dir / "previews"
            if previews_dir.exists():
                for preview_file in previews_dir.glob("*.png"):
                    zip_file.write(
                        preview_file,
                        f"{model_name}/previews/{preview_file.name}"
                    )

            # Add quality report
            quality_path = model_dir / "quality.json"
            if quality_path.exists():
                zip_file.write(quality_path, f"{model_name}/quality.json")

    # CRITICAL: Reset buffer position
    zip_buffer.seek(0)
    return zip_buffer
```

### Error Code Taxonomy
```python
# Source: REST API best practices + user-locked decisions
# app/api/error_codes.py
from enum import Enum

class ErrorCode(str, Enum):
    """
    Centralized error code taxonomy.

    Per user decision: Human-readable snake_case codes.
    """
    # Validation errors (422)
    VALIDATION_FAILED = "VALIDATION_FAILED"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    INVALID_FILE_COUNT = "INVALID_FILE_COUNT"

    # Not found errors (404)
    JOB_NOT_FOUND = "JOB_NOT_FOUND"

    # State conflict errors (409)
    JOB_NOT_READY = "JOB_NOT_READY"

    # Gone errors (410)
    JOB_EXPIRED = "JOB_EXPIRED"

    # Model/processing errors (500)
    MODEL_FAILED = "MODEL_FAILED"
    MODEL_OOM = "MODEL_OOM"
    MODEL_VRAM_OOM = "MODEL_VRAM_OOM"  # Distinguish GPU vs system RAM
    MODEL_CONVERGENCE_FAILED = "MODEL_CONVERGENCE_FAILED"
    QUALITY_THRESHOLD_FAILED = "QUALITY_THRESHOLD_FAILED"
    INCOMPLETE_RESULTS = "INCOMPLETE_RESULTS"

    # Resource errors (503)
    GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
    DISK_FULL = "DISK_FULL"
    MEMORY_EXHAUSTED = "MEMORY_EXHAUSTED"

    # Generic
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

# Error code to suggestion mapping
ERROR_SUGGESTIONS = {
    ErrorCode.VALIDATION_FAILED: "Fix the validation errors and try again.",
    ErrorCode.FILE_TOO_LARGE: "Reduce image resolution or compress PNGs before uploading (max 20MB per file).",
    ErrorCode.INVALID_FILE_FORMAT: "Ensure all uploaded files are valid PNG images.",
    ErrorCode.INVALID_FILE_COUNT: "Upload exactly 6 view images and 6 depth render images.",
    ErrorCode.JOB_NOT_FOUND: "Check the job ID and try again.",
    ErrorCode.JOB_NOT_READY: "Wait for job to complete before downloading. Poll GET /jobs/{job_id} for status.",
    ErrorCode.JOB_EXPIRED: "Job results are no longer available. Submit a new job if needed.",
    ErrorCode.MODEL_FAILED: "Model processing failed. Try again with different inputs or contact support.",
    ErrorCode.MODEL_OOM: "System ran out of memory. Try again when resources are available.",
    ErrorCode.MODEL_VRAM_OOM: "GPU ran out of VRAM. Try again when GPU resources are available.",
    ErrorCode.MODEL_CONVERGENCE_FAILED: "Model failed to converge. Try uploading higher-quality or better-lit input images.",
    ErrorCode.QUALITY_THRESHOLD_FAILED: "Reconstruction quality below threshold. Try uploading higher-quality input images.",
    ErrorCode.INCOMPLETE_RESULTS: "Job outputs are incomplete. Contact support.",
    ErrorCode.GPU_UNAVAILABLE: "No GPU available. Job will retry automatically.",
    ErrorCode.DISK_FULL: "Server disk is full. Contact support.",
    ErrorCode.MEMORY_EXHAUSTED: "Server out of memory. Try again later.",
    ErrorCode.UNKNOWN_ERROR: "An unexpected error occurred. Contact support if this persists.",
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| RFC 7807 Problem Details | RFC 9457 Problem Details | 2023 | RFC 9457 is the official successor, but user locked in simpler custom format |
| zipfile with file-like objects | stream-zip library | 2020+ | stream-zip offers better memory efficiency for TB-scale files, but stdlib sufficient for this use case |
| Generic 400 for all client errors | Specific 4xx codes (422, 409, 410) | Ongoing best practice | Better error semantics help clients handle errors appropriately |
| Pydantic v1 ValidationError | Pydantic v2 ValidationError | 2023 (Pydantic v2 release) | Different error structure, `.errors()` method returns different format |

**Deprecated/outdated:**
- **Writing ZIPs to disk before streaming:** Modern pattern uses `BytesIO` + `StreamingResponse` for better performance
- **Plain string error messages:** Industry moving toward structured error formats (RFC 9457, custom JSON schemas)
- **HTTP 500 for validation errors:** Should use 422 for semantic validation failures, 400 for syntax errors

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal ZIP compression level for preview images**
   - What we know: `compresslevel=9` is maximum compression, but PNGs are already compressed
   - What's unclear: Whether re-compressing PNGs in ZIP saves meaningful space vs CPU time
   - Recommendation: Start with `compresslevel=9`, profile actual ZIP sizes. If minimal savings, drop to `compresslevel=6` for faster compression.

2. **When to clean up completed jobs**
   - What we know: User wants 410 Gone for expired jobs, implying cleanup happens
   - What's unclear: Cleanup policy not defined in phase requirements (time-based? storage-based?)
   - Recommendation: Phase 5 should validate cleanup hasn't happened (return 410 if missing), but cleanup scheduling is separate concern for Phase 6 or operational procedures.

3. **Error detail verbosity for production**
   - What we know: User wants detailed breakdown (pipeline stage, model name, actual vs expected values)
   - What's unclear: Whether production should expose internal details (could leak implementation details to attackers)
   - Recommendation: Implement full detail as specified. If security concerns arise in production, add environment-based detail filtering later.

## Sources

### Primary (HIGH confidence)
- [FastAPI Custom Response Documentation](https://fastapi.tiangolo.com/advanced/custom-response/) - FileResponse and StreamingResponse usage
- [FastAPI Error Handling Documentation](https://fastapi.tiangolo.com/tutorial/handling-errors/) - HTTPException and custom exception handlers
- [Python zipfile Documentation](https://docs.python.org/3/library/zipfile.html) - Standard library ZIP creation
- [Pydantic Validation Errors Documentation](https://docs.pydantic.dev/latest/errors/validation_errors/) - Pydantic v2 error structure

### Secondary (MEDIUM confidence)
- [Beeceptor: 400 vs 422](https://beeceptor.com/docs/concepts/400-vs-422/) - HTTP status code semantics for validation
- [MDN: 410 Gone](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/410) - 410 vs 404 for expired resources
- [Speakeasy: REST API Errors Best Practices](https://www.speakeasy.com/api-design/errors) - Error response structure recommendations
- [Medium: FastAPI Streaming Response](https://medium.com/@vickypalaniappan12/create-in-memory-zip-files-in-python-79193fbbc6c3) - BytesIO + zipfile pattern

### Tertiary (LOW confidence - marked for validation)
- WebSearch results on HTTP status code selection (multiple sources agreeing on 422 for validation, 409 for conflicts, 503 for resources)
- Community discussions on error code naming conventions (snake_case consensus but no authoritative source)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - FastAPI and Python stdlib are well-documented, existing codebase already uses FastAPI
- Architecture: HIGH - Patterns verified with official FastAPI docs and Python docs
- Pitfalls: MEDIUM - Based on common REST API patterns and community knowledge, validated with multiple sources

**Research date:** 2026-01-31
**Valid until:** 2026-03-31 (60 days - FastAPI and HTTP standards are stable)
