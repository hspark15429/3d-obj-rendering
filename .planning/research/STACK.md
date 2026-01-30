# Technology Stack

**Project:** 3D Reconstruction API (ReconViaGen + nvdiffrec)
**Researched:** 2026-01-30
**Context:** Docker-based API for multi-view 3D reconstruction with GPU acceleration (RTX 3090, 24GB VRAM)

## Executive Summary

This stack prioritizes **compatibility** over bleeding-edge versions due to ML dependencies. ReconViaGen requires PyTorch 2.4.0, while nvdiffrec requires PyTorch 1.10+. The stack uses PyTorch 2.4.0 as the common denominator, Python 3.10 for maximum compatibility, and FastAPI for async job handling. Redis + Celery provides battle-tested job queuing for long-running ML inference tasks.

## Recommended Stack

### Core Python Runtime

| Technology | Version | Purpose | Rationale | Confidence |
|------------|---------|---------|-----------|------------|
| Python | **3.10.x** | Runtime environment | Required by ReconViaGen; maximum compatibility with PyTorch 2.4.0 and all ML dependencies. Python 3.11/3.12 now supported by PyTorch 2.10+ but untested with legacy dependencies like nvdiffrec. | **HIGH** |
| PyTorch | **2.4.0** (NOT 2.4.1) | ML framework | Explicitly required by ReconViaGen. PyTorch 2.4.1 may work but not tested. nvdiffrec requires 1.10+ (compatible). Latest PyTorch 2.10+ drops CUDA 11.8 support which is needed for RTX 3090. | **HIGH** |
| torchvision | **0.19.0** | Vision utilities | Matches PyTorch 2.4.0 compatibility matrix per ReconViaGen requirements. | **HIGH** |
| CUDA | **11.8** | GPU acceleration | PyTorch 2.4.0 supports CUDA 11.8; RTX 3090 (Ampere architecture) compatible. Latest PyTorch 2.10+ only supports CUDA 12.6+, breaking compatibility with existing setup. | **HIGH** |

**Why NOT newer versions:**
- PyTorch 2.5+: Drops CUDA 11.8 support (requires CUDA 12.6+)
- Python 3.11/3.12: While now supported by PyTorch 2.10+, untested with nvdiffrec and spconv dependencies
- PyTorch 2.4.1: Not explicitly tested by ReconViaGen (stick to exact version)

### Web Framework & API

| Technology | Version | Purpose | Rationale | Confidence |
|------------|---------|---------|-----------|------------|
| FastAPI | **0.115.8** | Web framework | Native async support critical for ML inference APIs. Handles 15K-20K req/sec vs Flask's 2K-3K. Growing 35-40% YoY in 2025. Supports async endpoints but delegates compute-heavy inference to Celery. | **HIGH** |
| Pydantic | **2.x** | Data validation | Required by FastAPI 0.115.x; handles request/response validation for job API. | **HIGH** |
| Uvicorn | **0.34.x** | ASGI server | Development server; 45K req/sec vs Gunicorn's 10K. Use with Gunicorn worker for production. | **HIGH** |
| Gunicorn | **23.x** | Process manager | Production deployment: `gunicorn -k uvicorn.workers.UvicornWorker --workers 4` for multi-core scaling and fault tolerance. Industry standard for ASGI production. | **HIGH** |

**Why FastAPI over Flask:**
- Async I/O: Handles concurrent job status queries while Celery processes inference
- Auto-generated OpenAPI docs: Critical for API usability
- Type safety: Pydantic validation catches errors before hitting Celery
- Performance: 7.5x throughput vs Flask (though bottleneck will be GPU inference)

**Why NOT other options:**
- Flask: No native async support; poor fit for job queue API pattern
- BentoML/Ray Serve: Over-engineered for 2-model use case; adds deployment complexity
- Pure Uvicorn: No process management; single point of failure

### Job Queue & Task Management

| Technology | Version | Purpose | Rationale | Confidence |
|------------|---------|---------|-----------|------------|
| Celery | **5.6.2** | Task queue | Battle-tested for async ML jobs. Handles retries, timeouts, result storage. Redis broker stability fixed in v5.4.0. Supports scheduling and concurrency control. | **HIGH** |
| Redis | **7.2.x** | Message broker & result backend | Fast, simple, widely deployed. In-memory = low latency for job status checks. Use 7.2.x (BSD licensed) NOT 7.4+ (restrictive licensing). Celery docs confirm Redis as "feature complete" broker. | **HIGH** |
| redis-py | **5.2.x** | Redis client | Official Python client for Redis; required by Celery. | **MEDIUM** |

**Why Celery + Redis:**
- Proven pattern: "Celery with Redis provides a battle-tested solution for reliable job processing in Python" (2025 production guidance)
- Simplicity: Single Redis instance handles both broker and result backend
- Performance: In-memory operations, negligible overhead vs 3D reconstruction compute time
- Features: Built-in retries, task timeouts, result expiration, monitoring

**Why NOT alternatives:**
- RabbitMQ: Overkill for single-server deployment; adds complexity without benefits
- RQ: Simpler than Celery but lacks advanced features (canvas, chord, task routing)
- Database queue: Orders of magnitude slower than Redis for status polling

**Redis licensing consideration:**
- Redis 7.0-7.2.4: BSD-3-Clause (permissive)
- Redis 7.4-7.8: Dual RSALv2/SSPLv1 (restrictive, avoid)
- Redis 8.0+: Tri-license with AGPLv3 option (restrictive, avoid)
- **Recommendation:** Pin to `redis:7.2.6-alpine` Docker image for licensing safety

### 3D Reconstruction Dependencies

| Library | Version | Purpose | Installation Notes | Confidence |
|---------|---------|---------|-------------------|------------|
| nvdiffrast | **latest from git** | Differentiable rendering | Install from GitHub: `pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation`. Requires cudatoolkit-dev, ninja, wheel, setuptools. Compiles CUDA kernels at build time. | **MEDIUM** |
| spconv | **2.3.6** | Sparse convolution (ReconViaGen) | CUDA-specific build: Install with `--extra-index-url` pointing to PyTorch CUDA index. Must match CUDA 11.8. | **MEDIUM** |
| xformers | **0.0.27.post2** | Attention mechanisms (ReconViaGen) | Matches PyTorch 2.4.0. Check compatibility matrix on GitHub releases. | **MEDIUM** |
| PyTorch3D | **0.7.8** | 3D utilities | Supports PyTorch 2.1-2.4.1. CUDA 11.7+ required (11.8 compatible). **Build from source** to match PyTorch/CUDA: `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`. Pre-compiled wheels often incompatible. | **MEDIUM** |

**Critical installation order:**
1. PyTorch 2.4.0 + torchvision 0.19.0 (with CUDA 11.8)
2. spconv 2.3.6 (CUDA-specific wheel)
3. xformers 0.0.27.post2
4. nvdiffrast (from git, compiles against installed PyTorch)
5. PyTorch3D (from git, compiles against installed PyTorch/CUDA)

**Why build from source (nvdiffrast, PyTorch3D):**
- Pre-compiled wheels rarely match exact PyTorch + CUDA combination
- Building ensures binary compatibility with your CUDA toolkit
- Prevents cryptic runtime errors from ABI mismatches

### Mesh Processing & Validation

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| trimesh | **4.5.x** | Mesh I/O, validation, export | Primary mesh library. Handles OBJ/PLY/glTF I/O, watertight checks, repair. Lightweight, no GPU dependencies. | **HIGH** |
| Open3D | **0.19.x** | Advanced mesh operations | Self-intersection detection (`is_self_intersecting`), visualization, ICP registration. Use for quality validation. Python 3.12 support added Jan 2025. | **HIGH** |
| neatmesh | **latest** | Mesh quality metrics | Computes quality metrics: aspect ratio, volume, face area, neighbor ratios. Use for quantitative validation. Reads `neatmesh.toml` for quality thresholds. | **MEDIUM** |
| PyVista | **0.46.3** | 3D visualization | Optional: mesh visualization, VTK integration. Heavy dependency, use only if visualization needed. v0.47 adds improved trimesh interop. | **LOW** |

**Recommended mesh validation pipeline:**
1. **trimesh**: Load mesh, basic checks (watertight, manifold)
2. **Open3D**: Self-intersection detection
3. **neatmesh**: Quality metrics (aspect ratio, volume)
4. **Result**: Pass/fail with quality scores

**Why trimesh as primary:**
- Lightweight: No GPU/VTK dependencies
- Fast: Pure Python + NumPy
- Complete: Handles all common mesh formats
- Maintained: Active development, large community

**Why NOT PyVista for validation:**
- Heavy: VTK dependency adds 100MB+ to Docker image
- Overkill: Visualization features not needed for API
- Use case: Only add if you need VTK algorithms

### Docker Infrastructure

| Technology | Version | Purpose | Rationale | Confidence |
|------------|---------|---------|-----------|------------|
| Base Image | **nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04** | CUDA foundation | Official NVIDIA image with CUDA 11.8 + cuDNN 8. `devel` variant includes headers for compiling nvdiffrast/PyTorch3D. Ubuntu 22.04 LTS for stability. | **HIGH** |
| Python Install | **3.10.x via deadsnakes PPA** | Runtime | System Python in Ubuntu 22.04 is 3.10.6; add deadsnakes for patch versions. Avoid conda in Docker (bloated). | **HIGH** |
| NVIDIA Container Toolkit | **latest** | GPU access | Installed on host, not in container. Required for `--gpus` flag. Configure with `nvidia-ctk runtime configure --runtime=docker`. | **HIGH** |
| Docker Compose | **2.x with `deploy.resources`** | Orchestration | Modern GPU syntax: `deploy.resources.reservations.devices` with `driver: nvidia, capabilities: [gpu]`. Replaces deprecated `runtime: nvidia`. | **HIGH** |

**Docker build strategy:**
1. Start with NVIDIA CUDA base image (not PyTorch official image - lacks build tools)
2. Install Python 3.10 + build essentials (gcc, g++, ninja, cmake)
3. Install PyTorch 2.4.0 from wheel index (faster than pip default)
4. Build nvdiffrast, PyTorch3D from source (compiled against base image CUDA)
5. Install app requirements (FastAPI, Celery, trimesh, etc.)
6. Multi-stage build to reduce final image size (builder stage for compilation)

**Why NOT pytorch/pytorch base image:**
- Missing build tools (gcc, cmake) needed for nvdiffrast compilation
- Harder to control exact CUDA/Python versions
- NVIDIA CUDA images are canonical source

**Docker Compose best practices:**
```yaml
services:
  api:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # Single RTX 3090
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0

  redis:
    image: redis:7.2.6-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru

  celery:
    build: .
    command: celery -A app.tasks worker --loglevel=info --concurrency=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
```

**Why `--concurrency=1` for Celery:**
- RTX 3090 has 24GB VRAM
- ReconViaGen + nvdiffrec likely use 10-20GB per job
- Running concurrent jobs risks OOM errors
- Better to queue jobs than crash GPU

### Supporting Libraries

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| numpy | **1.26.x** | Array operations | Pin <2.0 for ML compatibility (PyTorch/Open3D) |
| Pillow | **10.x** | Image I/O | Handle 2048x2048 RGB, 1024x1024 depth |
| imageio | **2.x** | Image utilities | Used by nvdiffrec for texture I/O |
| ninja | **1.11.x** | Build system | Required for nvdiffrast, PyTorch3D compilation |
| httpx | **0.28.x** | HTTP client | For async health checks, result downloads |
| python-multipart | **0.0.x** | File uploads | FastAPI multipart form support |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|-------------------|
| **Python** | 3.10 | 3.11/3.12 | Untested with spconv, nvdiffrec; compatibility risk |
| **PyTorch** | 2.4.0 | 2.10 (latest) | Drops CUDA 11.8; requires CUDA 12.6+ (incompatible with RTX 3090 setup) |
| **Web Framework** | FastAPI | Flask | No native async; poor job API pattern fit |
| **Web Framework** | FastAPI | BentoML/Ray Serve | Over-engineered for 2 models; complex deployment |
| **Job Queue** | Celery | RQ | Lacks advanced features (canvas, retries, monitoring) |
| **Message Broker** | Redis 7.2 | RabbitMQ | Overkill for single-server; unnecessary complexity |
| **Message Broker** | Redis 7.2 | Redis 8.0 | Restrictive tri-licensing (RSALv2/SSPLv1/AGPLv3) |
| **ASGI Server** | Uvicorn + Gunicorn | Pure Uvicorn | No process management; single point of failure |
| **Base Image** | nvidia/cuda | pytorch/pytorch | Missing build tools; harder to control versions |
| **Mesh Library** | trimesh | PyMeshLab | Heavier; MeshLab server deprecated |

## Installation Script

```bash
#!/bin/bash
# Build environment for Docker

# System dependencies (in Dockerfile)
apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    build-essential cmake ninja-build \
    git wget

# Python dependencies (install order matters!)
pip3 install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118

# CUDA-specific ML libraries
pip3 install --no-cache-dir \
    spconv-cu118==2.3.6 \
    xformers==0.0.27.post2

# Build from source (requires CUDA toolkit)
pip3 install --no-cache-dir \
    ninja setuptools wheel
pip3 install --no-cache-dir \
    "git+https://github.com/NVlabs/nvdiffrast.git" \
    --no-build-isolation
pip3 install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git"

# Web framework & job queue
pip3 install --no-cache-dir \
    fastapi==0.115.8 \
    uvicorn[standard]==0.34.0 \
    gunicorn==23.0.0 \
    celery[redis]==5.6.2 \
    redis==5.2.0

# Mesh processing
pip3 install --no-cache-dir \
    trimesh==4.5.2 \
    open3d==0.19.0 \
    neatmesh

# Supporting libraries
pip3 install --no-cache-dir \
    numpy==1.26.4 \
    Pillow==10.4.0 \
    imageio==2.36.0 \
    httpx==0.28.1 \
    python-multipart==0.0.20
```

## Version Pinning Strategy

| Category | Strategy | Rationale |
|----------|----------|-----------|
| **ML Core** (PyTorch, CUDA) | **Exact pin** (`==`) | Breaking changes common; must match ReconViaGen requirements |
| **ML Dependencies** (spconv, xformers) | **Exact pin** (`==`) | Binary compatibility with PyTorch; patch versions can break |
| **Web Framework** (FastAPI, Celery) | **Minor pin** (`~=0.115.0`) | Patch releases safe; security fixes important |
| **Utilities** (trimesh, httpx) | **Major pin** (`>=4.0,<5.0`) | Semantic versioning respected; allow patches |
| **Build Tools** (ninja, setuptools) | **No pin** | Latest compatible; only used at build time |

**Pin file recommendation:** Use `requirements.txt` with exact versions for reproducibility:
```
# requirements.txt (generated from pip freeze after successful build)
torch==2.4.0
torchvision==0.19.0
# ... (all exact versions)
```

**Why exact pins for ML:**
- PyTorch 2.4.0 vs 2.4.1: Binary ABI differences can break compiled extensions
- CUDA version drift: Patch updates can break kernel compatibility
- spconv/xformers: Tightly coupled to PyTorch version

## Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| Python/PyTorch versions | **HIGH** | Verified from ReconViaGen repo (WebFetch), nvdiffrec repo (WebFetch), PyTorch official docs (WebFetch) |
| FastAPI/Celery stack | **HIGH** | Cross-verified from multiple 2025 production guides (WebSearch), official Celery docs (WebFetch) |
| Docker GPU setup | **HIGH** | Verified from official Docker Compose docs (WebSearch), NVIDIA Container Toolkit docs (WebSearch) |
| Mesh libraries | **MEDIUM** | trimesh/Open3D well-documented; neatmesh less adoption (WebSearch only) |
| nvdiffrast/PyTorch3D builds | **MEDIUM** | Installation notes verified from GitHub (WebFetch), but build-from-source has environment-specific risks |
| Redis licensing | **HIGH** | Verified from official Redis GitHub releases (WebSearch), Redis docs (WebSearch) |

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **nvdiffrast compilation failure** | Can't run nvdiffrec model | Dockerfile multi-stage build with error handling; fallback to CPU-only mode |
| **PyTorch3D build issues** | Missing 3D utilities | Build with verbose logging; pin CUB library version; use pre-compiled wheel as last resort |
| **CUDA OOM errors** | Job failures | Celery concurrency=1; add VRAM monitoring; implement job queueing |
| **Redis memory limits** | Lost job results | Configure `maxmemory 2gb` + `allkeys-lru` eviction; store results on disk after completion |
| **Dependency version drift** | Build failures on rebuild | Exact version pins in requirements.txt; Docker layer caching; lock file (pip freeze) |

## Sources

### High Confidence (Official Documentation)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) - PyTorch 2.10.0 versions and CUDA support
- [ReconViaGen GitHub Repository](https://github.com/estheryang11/ReconViaGen) - Python 3.10, PyTorch 2.4.0, dependencies
- [nvdiffrec GitHub Repository](https://github.com/NVlabs/nvdiffrec) - PyTorch 1.10+, CUDA 11.3+, installation
- [Celery Documentation - Using Redis](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html) - Celery 5.6.2 with Redis
- [Docker Compose GPU Support](https://docs.docker.com/compose/how-tos/gpu-support/) - Modern `deploy.resources` syntax
- [FastAPI Release Notes](https://github.com/fastapi/fastapi/releases) - Version 0.115.8 (Jan 30, 2025)
- [Celery Releases](https://github.com/celery/celery/releases) - Version 5.6.2
- [Redis Official Docker Hub](https://hub.docker.com/_/redis) - Redis 7.2.x, 8.x licensing

### Medium Confidence (Cross-Verified WebSearch)
- [FastAPI vs Flask 2025 Comparison](https://strapi.io/blog/fastapi-vs-flask-python-framework-comparison) - Performance benchmarks
- [Celery + Redis + FastAPI 2025 Guide](https://medium.com/@dewasheesh.rana/celery-redis-fastapi-the-ultimate-2025-production-guide-broker-vs-backend-explained-5b84ef508fa7) - Production patterns
- [Python ASGI Servers 2025](https://www.deployhq.com/blog/python-application-servers-in-2025-from-wsgi-to-modern-asgi-solutions) - Uvicorn + Gunicorn recommendations
- [Reliable Python Queues 2025](https://medium.com/@Nexumo_/reliable-python-queues-7-celery-dramatiq-rq-choices-266ac544a4a5) - Celery vs RQ comparison
- [Redis vs RabbitMQ for ML Workloads](https://medium.com/@omsaikrishnamadhav.lella/choosing-between-redis-kafka-and-rabbitmq-for-ml-workloads-9284341f83f7) - Broker selection
- [Best Python 3D Geometry Libraries 2025](https://meshlib.io/blog/top-python-libraries-for-3d-geometry/) - trimesh, Open3D, neatmesh
- [PyTorch3D Installation Guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) - CUDA compatibility, build from source
- [nvdiffrast Installation](https://github.com/NVlabs/nvdiffrast) - Build requirements

### Low Confidence (Single Source)
- neatmesh quality metrics - Limited documentation, small community adoption
- PyVista v0.47 trimesh interop - Upcoming release, not yet verified in production
