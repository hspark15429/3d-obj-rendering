"""3D Reconstruction API - Foundation"""
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, Request
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
    """Handle file validation errors with 400 status."""
    return JSONResponse(
        status_code=400,
        content={"error": exc.message, "field": exc.field}
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
