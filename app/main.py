"""3D Reconstruction API - Foundation"""
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetMemoryInfo,
    nvmlSystemGetDriverVersion,
)

from app.api.jobs import router as jobs_router
from app.api.error_codes import ErrorCode, make_error_detail
from app.services.file_handler import FileValidationError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MINIMUM_VRAM_GB = 12

# Shared state for GPU info (populated at startup)
gpu_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate GPU at startup, cleanup on shutdown."""
    logger.info("Starting GPU validation...")

    # Ensure storage directory exists
    Path("/app/storage/jobs").mkdir(parents=True, exist_ok=True)
    logger.info("Storage directory initialized")

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        memory = nvmlDeviceGetMemoryInfo(handle)

        total_gb = memory.total / (1024**3)
        if total_gb < MINIMUM_VRAM_GB:
            raise RuntimeError(
                f"Insufficient VRAM: {total_gb:.1f}GB < {MINIMUM_VRAM_GB}GB required"
            )

        # Cache GPU info for health checks
        gpu_state["name"] = nvmlDeviceGetName(handle)
        gpu_state["driver_version"] = nvmlSystemGetDriverVersion()
        gpu_state["memory_total_gb"] = round(total_gb, 2)
        gpu_state["initialized"] = True

        logger.info(f"GPU validated: {gpu_state['name']} with {total_gb:.1f}GB VRAM")
        nvmlShutdown()

    except Exception as e:
        logger.error(f"GPU validation failed: {e}")
        raise RuntimeError(f"GPU unavailable or insufficient: {e}")

    yield  # Application runs

    # Shutdown cleanup
    gpu_state.clear()
    logger.info("Shutdown complete")


app = FastAPI(
    title="3D Reconstruction API",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(jobs_router)


# Exception handlers
@app.exception_handler(FileValidationError)
async def file_validation_handler(request: Request, exc: FileValidationError):
    """Handle file validation errors with structured format (422 status)."""
    # Map common validation errors to specific error codes
    code = ErrorCode.VALIDATION_FAILED
    if "too large" in exc.message.lower() or "exceeds" in exc.message.lower():
        code = ErrorCode.FILE_TOO_LARGE
    elif "format" in exc.message.lower() or "png" in exc.message.lower():
        code = ErrorCode.INVALID_FILE_FORMAT
    elif "expected" in exc.message.lower() and "files" in exc.message.lower():
        code = ErrorCode.INVALID_FILE_COUNT

    error_detail = make_error_detail(
        code=code,
        message=exc.message,
        details={"field": exc.field} if exc.field else {}
    )
    return JSONResponse(
        status_code=422,
        content={"error": error_detail}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Format all HTTPException as structured error response."""
    # If exc.detail is already a dict with our structure, use it
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        error_detail = exc.detail
    else:
        # Wrap in UNKNOWN_ERROR structure
        error_detail = make_error_detail(
            code=ErrorCode.UNKNOWN_ERROR,
            message=str(exc.detail),
            details={}
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format Pydantic validation errors as structured response."""
    # Extract field-level errors
    field_errors = {}
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        field_errors[field_path] = {
            "message": error["msg"],
            "type": error["type"]
        }

    error_detail = make_error_detail(
        code=ErrorCode.VALIDATION_FAILED,
        message="Request validation failed",
        details={"fields": field_errors, "error_count": len(field_errors)}
    )
    return JSONResponse(
        status_code=422,
        content={"error": error_detail}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions with structured format (500 status)."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    error_detail = make_error_detail(
        code=ErrorCode.UNKNOWN_ERROR,
        message="An internal server error occurred",
        details={}  # Don't expose internal details for security
    )
    return JSONResponse(
        status_code=500,
        content={"error": error_detail}
    )


@app.get("/health")
async def health_check():
    """Return health status with GPU information."""
    if not gpu_state.get("initialized"):
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "GPU not initialized"}
        )

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        memory = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()

        return {
            "status": "healthy",
            "gpu": {
                "name": gpu_state["name"],
                "driver_version": gpu_state["driver_version"],
                "memory_total_gb": gpu_state["memory_total_gb"],
                "memory_free_gb": round(memory.free / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )
