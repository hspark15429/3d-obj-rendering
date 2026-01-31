"""
Centralized error code taxonomy for API responses.

Per CONTEXT.md: All errors use structured JSON with code, message, details, and suggestion fields.
Error suggestions must be actionable and specific to each error type.
"""
from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    """
    Centralized error code taxonomy.

    Human-readable snake_case codes per user decision.
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
    MODEL_VRAM_OOM = "MODEL_VRAM_OOM"  # Distinguish GPU VRAM vs system RAM
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
# Per CONTEXT.md: Suggestions must be specific and actionable
ERROR_SUGGESTIONS: dict[ErrorCode, str] = {
    ErrorCode.VALIDATION_FAILED: "Fix the validation errors and try again.",
    ErrorCode.FILE_TOO_LARGE: "Reduce image resolution or compress PNGs before uploading (max 20MB per file).",
    ErrorCode.INVALID_FILE_FORMAT: "Ensure all uploaded files are valid PNG images.",
    ErrorCode.INVALID_FILE_COUNT: "Upload exactly 6 view images and 6 depth render images.",
    ErrorCode.JOB_NOT_FOUND: "Check the job ID and try again.",
    ErrorCode.JOB_NOT_READY: "Wait for job to complete before downloading. Poll GET /jobs/{job_id} for status.",
    ErrorCode.JOB_EXPIRED: "Job results are no longer available. Submit a new job if needed.",
    ErrorCode.MODEL_FAILED: "Model processing failed. Try again with different inputs or contact support.",
    ErrorCode.MODEL_OOM: "System ran out of memory. Try again when resources are available.",
    ErrorCode.MODEL_VRAM_OOM: "GPU ran out of VRAM (16GB limit). Try again when GPU resources are available.",
    ErrorCode.MODEL_CONVERGENCE_FAILED: "Model failed to converge. Try uploading higher-quality or better-lit input images.",
    ErrorCode.QUALITY_THRESHOLD_FAILED: "Reconstruction quality below threshold. Try uploading higher-quality input images.",
    ErrorCode.INCOMPLETE_RESULTS: "Job outputs are incomplete. Contact support.",
    ErrorCode.GPU_UNAVAILABLE: "No GPU available. Job will retry automatically.",
    ErrorCode.DISK_FULL: "Server disk is full. Contact support.",
    ErrorCode.MEMORY_EXHAUSTED: "Server out of memory. Try again later.",
    ErrorCode.UNKNOWN_ERROR: "An unexpected error occurred. Contact support if this persists.",
}


def make_error_detail(
    code: ErrorCode,
    message: str,
    details: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Build structured error detail dict with automatic suggestion lookup.

    Args:
        code: Error code from ErrorCode enum
        message: Human-readable error message
        details: Additional context dict (optional)

    Returns:
        Dict with code, message, details, and suggestion fields
    """
    return {
        "code": code.value,
        "message": message,
        "details": details or {},
        "suggestion": ERROR_SUGGESTIONS.get(code, ERROR_SUGGESTIONS[ErrorCode.UNKNOWN_ERROR]),
    }
