# Research Summary

**Project:** 3D Reconstruction API (ReconViaGen + nvdiffrec)
**Synthesized:** 2026-01-30

## Key Stack Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Python** | 3.10.x | Required by ReconViaGen; max compatibility with ML dependencies |
| **PyTorch** | 2.4.0 (NOT 2.4.1) | Explicitly required by ReconViaGen; nvdiffrec needs 1.10+ (compatible) |
| **CUDA** | 11.8 | PyTorch 2.4.0 supports it; RTX 3090 compatible |
| **Web Framework** | FastAPI 0.115.x | Native async, auto OpenAPI docs, 7x throughput vs Flask |
| **Job Queue** | Celery 5.6.x + Redis 7.2.x | Battle-tested for async ML; Redis 7.2.x for BSD license |
| **Docker Base** | nvidia/cuda:11.8.0-devel-ubuntu22.04 | `-devel` required for nvdiffrast/spconv compilation |

**Critical installation order:** PyTorch → spconv → xformers → nvdiffrast → PyTorch3D (build from source)

## Table Stakes Features (Must Have)

- Async job management (POST /jobs → 202 + job ID)
- Job status polling (GET /jobs/{id})
- Multi-view input support (6 RGB views)
- Result download (GET /jobs/{id}/result)
- Basic error reporting with clear codes
- Quality metrics (PSNR, SSIM)
- Thumbnail/preview generation

## Differentiators (Nice to Have)

- Model selection (ReconViaGen vs nvdiffrec)
- Webhook notifications
- Multiple output formats (GLB, OBJ)
- Mesh quality validation (topology checks)

## Recommended Architecture

```
Client → FastAPI → Redis Queue → Celery Workers (GPU) → Storage
                       ↓
              Job Status Tracking
```

**Components:**
1. **FastAPI Server** - REST API, file uploads, job submission
2. **Redis** - Message broker + result backend (single instance)
3. **Celery Workers** - GPU inference, one worker per model type
4. **Storage** - Docker volumes for inputs/outputs (MinIO optional)

**Data Flow:** Upload → Queue → Worker picks up → Inference → Write results → Update status

## Top 5 Pitfalls to Avoid

| # | Pitfall | Prevention | Phase |
|---|---------|------------|-------|
| 1 | **CUDA/PyTorch version mismatch** | Use `-devel` base image, strict install order, test GPU in build | Phase 1 |
| 2 | **Docker shared memory exhaustion** | Always `--shm-size=16G` or mount `/dev/shm` | Phase 1 |
| 3 | **VRAM overflow during concurrent jobs** | Celery `concurrency=1`, queue jobs sequentially | Phase 2 |
| 4 | **Model weight download during inference** | Pre-download weights in Docker build, not runtime | Phase 3 |
| 5 | **nvdiffrec two-pass training incomplete** | Verify both passes complete, check output file count | Phase 3 |

## Build Order Recommendations

**Phase 1: Docker + Basic API**
- Get GPU working in Docker container first (biggest risk)
- Simple FastAPI with health endpoint
- Redis + Celery infrastructure (no models yet)

**Phase 2: Job Pipeline**
- File upload handling
- Job queue submission
- Status tracking
- Result retrieval endpoints

**Phase 3: Model Integration**
- ReconViaGen integration (start with one model)
- nvdiffrec integration
- Model selection in API

**Phase 4: Quality & Preview**
- Quality metrics computation
- Preview image generation
- Status classification (normal/warning/failure)

**Phase 5: Polish & Documentation**
- Error handling refinement
- README.md with examples
- architecture.md

## Open Questions

1. **Model weights location** - Need to identify where ReconViaGen/nvdiffrec store weights and how to pre-download
2. **Camera pose format** - Do both models expect same input format or need conversion?
3. **VRAM profiling** - Actual usage unknown; may need adjustment based on testing

---
*Research complete. Ready for requirements definition.*
