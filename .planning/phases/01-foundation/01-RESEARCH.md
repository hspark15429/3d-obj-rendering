# Phase 1: Foundation - Research

**Researched:** 2026-01-31
**Domain:** Docker GPU infrastructure + FastAPI health endpoints
**Confidence:** HIGH

## Summary

This phase establishes the GPU-enabled Docker environment and basic FastAPI server with health check endpoint. The research focused on three core areas: (1) NVIDIA CUDA Docker image selection for CUDA 11.8 compatibility, (2) Docker Compose GPU configuration using the modern `deploy.resources.reservations.devices` syntax, and (3) FastAPI health endpoint patterns with GPU monitoring.

The standard approach uses `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` as the base image, Docker Compose with the modern GPU reservation syntax, and FastAPI's lifespan events for fail-fast GPU validation at startup. The health endpoint exposes GPU information via either PyTorch's CUDA functions or the `nvidia-ml-py` (pynvml) library.

**Primary recommendation:** Use a multi-stage Dockerfile with devel base for build stage (needed for nvdiffrast/PyTorch3D later), and a simpler runtime for now. Fail fast at startup using lifespan events if GPU is unavailable.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nvidia/cuda base image | 11.8.0-cudnn8-devel-ubuntu22.04 | CUDA runtime + build tools | Official NVIDIA image, devel variant includes headers for building CUDA extensions (nvdiffrast/PyTorch3D), cuDNN8 for ML workloads |
| FastAPI | 0.115.8 | Web framework | Native async, auto-generated OpenAPI docs, lifespan events for startup validation |
| Uvicorn | 0.34.x | ASGI server | High performance, development-friendly, works with FastAPI's exec CMD form for graceful shutdown |
| Python | 3.10.x | Runtime | Required by ReconViaGen, maximum ML library compatibility |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| nvidia-ml-py | latest | GPU monitoring via NVML | Health endpoint GPU info (driver version, memory, GPU name) |
| structlog | 24.x | Structured logging | JSON logs for production, human-readable for dev |
| pydantic | 2.x | Data validation | Health response schema, request/response models |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nvidia-ml-py for GPU info | torch.cuda functions | torch.cuda requires full PyTorch import; nvidia-ml-py is lightweight. Use nvidia-ml-py for health endpoint if PyTorch not already loaded |
| cudnn8-devel image | runtime image | Runtime is smaller but lacks headers for building CUDA extensions needed in later phases |
| Single-stage Dockerfile | Multi-stage | Multi-stage reduces image size but adds complexity; defer to Phase 2+ when build dependencies grow |

**Installation:**
```bash
# Base requirements for Phase 1
pip install fastapi==0.115.8 uvicorn[standard]==0.34.0 pydantic>=2.0 nvidia-ml-py structlog
```

## Architecture Patterns

### Recommended Project Structure
```
project/
├── docker-compose.yml      # Single service for Phase 1
├── Dockerfile              # CUDA-based image
├── requirements.txt        # Python dependencies
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI app with lifespan
│   ├── config.py           # Settings/configuration
│   └── health.py           # Health endpoint logic
└── scripts/
    └── entrypoint.sh       # Optional: GPU check before Python
```

### Pattern 1: Lifespan Events for Fail-Fast GPU Validation

**What:** Use FastAPI's lifespan context manager to verify GPU access at startup, preventing the server from accepting requests if GPU is unavailable.

**When to use:** Always for GPU-dependent services.

**Example:**
```python
# Source: FastAPI official docs - https://fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI
import torch

gpu_info = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Verify GPU access
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available - refusing to start")

    # Cache GPU info for health checks
    gpu_info["name"] = torch.cuda.get_device_name(0)
    gpu_info["device_count"] = torch.cuda.device_count()

    yield  # Application runs here

    # Shutdown: Cleanup
    gpu_info.clear()

app = FastAPI(lifespan=lifespan)
```

### Pattern 2: Docker Compose GPU Configuration

**What:** Modern GPU reservation syntax using `deploy.resources.reservations.devices`.

**When to use:** Any Docker Compose service requiring GPU access.

**Example:**
```yaml
# Source: Docker Compose GPU docs - https://docs.docker.com/compose/how-tos/gpu-support/
services:
  api:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # Specific GPU
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
```

### Pattern 3: Health Endpoint with GPU Metrics

**What:** Structured health response including GPU availability, memory, and device info.

**When to use:** All production deployments for monitoring and load balancer health checks.

**Example:**
```python
# Source: nvidia-ml-py docs - https://pypi.org/project/nvidia-ml-py/
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName,
    nvmlDeviceGetMemoryInfo, nvmlSystemGetDriverVersion, nvmlShutdown
)

@app.get("/health")
async def health_check():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        memory = nvmlDeviceGetMemoryInfo(handle)

        response = {
            "status": "healthy",
            "gpu": {
                "name": nvmlDeviceGetName(handle),
                "driver_version": nvmlSystemGetDriverVersion(),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_free_gb": round(memory.free / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
            }
        }
        nvmlShutdown()
        return response
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )
```

### Anti-Patterns to Avoid

- **Starting without GPU validation:** Never assume GPU is available. Always validate at startup via lifespan events, fail immediately if unavailable.

- **Using deprecated `runtime: nvidia`:** Use modern `deploy.resources.reservations.devices` syntax instead.

- **Heavy imports in health endpoint:** Don't import torch just for health checks if nvidia-ml-py suffices. Keep health endpoint lightweight.

- **Shell form CMD in Dockerfile:** Always use exec form `CMD ["fastapi", "run", ...]` to enable graceful shutdown and lifespan events.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU memory monitoring | Custom nvidia-smi parsing | nvidia-ml-py (pynvml) | NVML is the underlying library for nvidia-smi; Python bindings are official |
| Health check structure | Ad-hoc JSON responses | Pydantic models | Type safety, documentation, validation |
| Structured logging | Custom formatters | structlog | Battle-tested, JSON/pretty switching, context propagation |
| Container GPU access | Manual device mounts | NVIDIA Container Toolkit + deploy.resources | Handles driver libraries, device nodes, capabilities correctly |

**Key insight:** GPU container setup has many subtle requirements (driver library injection, device node creation, cgroup permissions). The NVIDIA Container Toolkit handles all of this; don't try to manually mount /dev/nvidia* devices.

## Common Pitfalls

### Pitfall 1: "Failed to initialize NVML: Unknown Error"
**What goes wrong:** nvidia-smi and GPU access work initially, then fail after hours/days.
**Why it happens:** systemd cgroup management can detach containers from GPUs when daemon-reload occurs.
**How to avoid:**
1. Create device node symlinks: `sudo nvidia-ctk system create-dev-char-symlinks --create-all` (run at boot)
2. Or use cgroupfs as Docker's cgroup driver instead of systemd
**Warning signs:** GPU access works on container start but fails later without restart.

### Pitfall 2: Missing `capabilities: [gpu]` in Compose
**What goes wrong:** Container starts but no GPU access; torch.cuda.is_available() returns False.
**Why it happens:** The `capabilities` field is required in Docker Compose GPU config, but easy to forget.
**How to avoid:** Always include `capabilities: [gpu]` in the devices configuration.
**Warning signs:** Container runs but nvidia-smi shows no devices.

### Pitfall 3: Using `base` or `runtime` image for build
**What goes wrong:** Later phases fail to build nvdiffrast/PyTorch3D because CUDA headers are missing.
**Why it happens:** Only `devel` images include CUDA development headers and toolchain.
**How to avoid:** Use `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` from the start.
**Warning signs:** Build errors like "cuda.h not found" or "nvcc not found".

### Pitfall 4: Shell form CMD preventing graceful shutdown
**What goes wrong:** Lifespan shutdown events never fire; resources not cleaned up.
**Why it happens:** Shell form `CMD command args` runs via /bin/sh, which doesn't forward signals.
**How to avoid:** Always use exec form: `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]`
**Warning signs:** Shutdown logs missing; connections not properly closed.

### Pitfall 5: VRAM check but not actual GPU compute test
**What goes wrong:** Health says "healthy" but inference fails with CUDA errors.
**Why it happens:** NVML can query GPU info even if CUDA runtime has issues.
**How to avoid:** During startup (lifespan), do a minimal CUDA operation like `torch.cuda.current_device()`.
**Warning signs:** Health endpoint returns healthy but actual workloads fail with CUDA errors.

## Code Examples

### Complete Dockerfile for Phase 1
```dockerfile
# Source: NVIDIA CUDA images - https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and build essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Use exec form for graceful shutdown support
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Complete docker-compose.yml for Phase 1
```yaml
# Source: Docker Compose GPU docs - https://docs.docker.com/compose/how-tos/gpu-support/
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app  # Mount for development
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Complete FastAPI App with Lifespan
```python
# app/main.py
# Source: FastAPI lifespan docs - https://fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlSystemGetDriverVersion
)
import logging

logger = logging.getLogger(__name__)

# Shared state for GPU info
gpu_state = {}

MINIMUM_VRAM_GB = 12  # From user requirements

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate GPU at startup, cleanup on shutdown."""
    logger.info("Starting GPU validation...")

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
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `runtime: nvidia` in compose | `deploy.resources.reservations.devices` | Docker Compose 3.8+ | More granular control, consistent with Docker Swarm |
| `@app.on_event("startup")` | `lifespan` context manager | FastAPI 0.100+ | Cleaner, supports shared state between startup/shutdown |
| tiangolo/uvicorn-gunicorn-fastapi image | Build from scratch | 2024 | Deprecated; official recommendation is custom Dockerfile |
| nvidia-docker2 package | nvidia-container-toolkit | 2022 | Unified toolkit, CDI support |

**Deprecated/outdated:**
- `@app.on_event("startup")` and `@app.on_event("shutdown")`: Deprecated in favor of lifespan context manager
- `runtime: nvidia` in docker-compose: Still works but modern syntax is `deploy.resources.reservations.devices`
- tiangolo/uvicorn-gunicorn-fastapi Docker image: Officially deprecated; build custom images

## Open Questions

1. **Multi-stage build timing**
   - What we know: Multi-stage can reduce image size by 45%+
   - What's unclear: Whether complexity is worth it for Phase 1 when only simple dependencies exist
   - Recommendation: Start with single-stage; refactor to multi-stage in Phase 2 when build dependencies grow

2. **Structlog vs standard logging**
   - What we know: Structlog provides JSON output, context propagation, better production debugging
   - What's unclear: Whether additional complexity is warranted for MVP
   - Recommendation: Use structlog from start; configuration is minimal and benefits are significant

3. **Port selection**
   - What we know: 8000 is conventional for uvicorn/FastAPI
   - What's unclear: Whether any conflicts exist in target environment
   - Recommendation: Use 8000, make configurable via environment variable

## Sources

### Primary (HIGH confidence)
- [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda) - Image tags, variants (base/runtime/devel)
- [Docker Compose GPU Support](https://docs.docker.com/compose/how-tos/gpu-support/) - Modern GPU configuration syntax
- [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) - Startup/shutdown pattern
- [FastAPI Docker Deployment](https://fastapi.tiangolo.com/deployment/docker/) - Dockerfile best practices
- [nvidia-ml-py PyPI](https://pypi.org/project/nvidia-ml-py/) - GPU monitoring library

### Secondary (MEDIUM confidence)
- [NVIDIA Container Toolkit Troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html) - NVML error debugging
- [PyTorch CUDA Memory](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) - Memory monitoring functions
- [Structlog FastAPI Integration](https://gist.github.com/nymous/f138c7f06062b7c43c060bf03759c29e) - Logging setup patterns

### Tertiary (LOW confidence)
- Forum discussions on "Failed to initialize NVML" errors - Solutions vary by environment

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official NVIDIA images, FastAPI docs verified
- Architecture: HIGH - Docker Compose GPU syntax verified from official docs
- Pitfalls: MEDIUM - Based on GitHub issues and forum reports, may vary by environment

**Research date:** 2026-01-31
**Valid until:** 60 days (stable technologies, CUDA 11.8 is mature)
