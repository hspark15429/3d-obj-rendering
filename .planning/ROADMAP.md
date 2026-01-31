# Roadmap: 3D Object Reconstruction API

## Overview

This roadmap delivers a Docker-based 3D reconstruction pipeline in 6 phases, progressing from GPU-enabled infrastructure through model integration to a production-ready API. Each phase builds toward the core value: upload images, get validated 3D meshes with a single Docker command.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Docker environment with GPU support and basic API
- [ ] **Phase 2: Job Pipeline** - Async job queue infrastructure for long-running inference
- [ ] **Phase 3: Model Integration** - ReconViaGen and nvdiffrec model execution
- [ ] **Phase 4: Quality & Preview** - Metrics computation and result validation
- [ ] **Phase 5: Results & Error Handling** - Complete API with robust error handling
- [ ] **Phase 6: Documentation** - README, architecture docs, and examples

## Phase Details

### Phase 1: Foundation
**Goal**: GPU-enabled Docker environment runs with single command and responds to health checks
**Depends on**: Nothing (first phase)
**Requirements**: DEPLOY-01, DEPLOY-02, API-05
**Success Criteria** (what must be TRUE):
  1. User can start entire system with `docker-compose up`
  2. Container has working GPU access (nvidia-docker)
  3. User can verify API is running via GET /health endpoint
  4. FastAPI server accepts requests and returns responses
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — Docker infrastructure + FastAPI health endpoint with GPU validation

### Phase 2: Job Pipeline
**Goal**: Users can submit jobs, check status, and cancel jobs through async queue
**Depends on**: Phase 1
**Requirements**: DEPLOY-03, API-01, API-02, API-04
**Success Criteria** (what must be TRUE):
  1. User can upload multi-view images + depth data and receive job ID
  2. User can check job status (queued/processing/completed/failed) by ID
  3. User can cancel a queued or running job
  4. Jobs process asynchronously via Celery + Redis queue
  5. File uploads validate input structure (6 views, depth renders)
**Plans**: 4 plans

Plans:
- [ ] 02-01-PLAN.md — Celery infrastructure with Redis broker
- [ ] 02-02-PLAN.md — Pydantic schemas and file validation service
- [ ] 02-03-PLAN.md — Job manager and Celery reconstruction task
- [ ] 02-04-PLAN.md — Job API endpoints and integration

### Phase 3: Model Integration
**Goal**: System runs both reconstruction models and produces textured mesh outputs
**Depends on**: Phase 2
**Requirements**: MODEL-01, MODEL-02, MODEL-03, OUT-01
**Success Criteria** (what must be TRUE):
  1. User can select ReconViaGen model and receive textured mesh output
  2. User can select nvdiffrec model and receive textured mesh output
  3. Jobs complete successfully with OBJ/PLY and texture files
  4. Model weights are pre-downloaded in Docker image (no runtime downloads)
  5. Both models run sequentially without VRAM overflow
**Plans**: TBD

Plans:
- [ ] TBD (to be planned)

### Phase 4: Quality & Preview
**Goal**: Results include quality metrics, status classification, and preview images
**Depends on**: Phase 3
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04, OUT-02
**Success Criteria** (what must be TRUE):
  1. System computes PSNR metric for reconstruction quality
  2. System computes SSIM metric for reconstruction quality
  3. System classifies results as normal/warning/failure based on thresholds
  4. Quality metrics and status saved as JSON with results
  5. System generates static preview images from multiple angles
**Plans**: TBD

Plans:
- [ ] TBD (to be planned)

### Phase 5: Results & Error Handling
**Goal**: Users can download complete results and receive clear errors for failures
**Depends on**: Phase 4
**Requirements**: API-03, ERR-01, ERR-02, ERR-03
**Success Criteria** (what must be TRUE):
  1. User can download mesh, textures, previews, and quality report by job ID
  2. Invalid uploads return clear error messages (format, size, structure)
  3. Model failures (OOM, convergence) return error status with details
  4. System resource issues (disk, memory) handled gracefully
  5. All error scenarios return appropriate HTTP codes and messages
**Plans**: TBD

Plans:
- [ ] TBD (to be planned)

### Phase 6: Documentation
**Goal**: Repository includes complete documentation and example outputs
**Depends on**: Phase 5
**Requirements**: DOC-01, DOC-02, DOC-03
**Success Criteria** (what must be TRUE):
  1. README.md contains execution instructions and API usage examples
  2. architecture.md explains system design and key decisions
  3. Example outputs included (3D meshes, previews, quality JSON)
  4. User can run system from documentation alone (no external knowledge)
**Plans**: TBD

Plans:
- [ ] TBD (to be planned)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 1/1 | Complete | 2026-01-31 |
| 2. Job Pipeline | 0/4 | Planned | - |
| 3. Model Integration | 0/TBD | Not started | - |
| 4. Quality & Preview | 0/TBD | Not started | - |
| 5. Results & Error Handling | 0/TBD | Not started | - |
| 6. Documentation | 0/TBD | Not started | - |

---
*Last updated: 2026-01-31 - Phase 2 planned (4 plans)*
