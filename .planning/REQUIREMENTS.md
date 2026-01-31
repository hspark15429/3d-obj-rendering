# Requirements

**Project:** 3D Object Reconstruction API
**Version:** v1
**Last updated:** 2026-01-31

## v1 Requirements

### API Endpoints

- [x] **API-01**: User can submit a job with multi-view images and depth renders, selecting model type, and receive a job ID
- [x] **API-02**: User can check job status (queued/processing/completed/failed) by job ID
- [x] **API-03**: User can download job results (mesh, textures, preview images, quality report) by job ID
- [x] **API-04**: User can cancel a running or queued job by job ID
- [x] **API-05**: User can check API health via GET /health endpoint

### 3D Reconstruction Models

- [x] **MODEL-01**: System can run ReconViaGen model on multi-view RGB + depth input to produce textured mesh
- [x] **MODEL-02**: System can run nvdiffrec model on multi-view RGB + depth input to produce textured mesh
- [x] **MODEL-03**: User can select which model to run when submitting a job

### Quality Validation

- [x] **QUAL-01**: System computes PSNR metric for reconstruction quality
- [x] **QUAL-02**: System computes SSIM metric for reconstruction quality
- [x] **QUAL-03**: System classifies result status as normal/warning/failure based on quality thresholds
- [x] **QUAL-04**: System saves quality metrics and status to JSON file with results

### Output & Preview

- [x] **OUT-01**: System generates textured mesh output (OBJ/PLY with texture files)
- [x] **OUT-02**: System generates static preview images from multiple angles

### Deployment

- [x] **DEPLOY-01**: System runs in Docker container with GPU support (nvidia-docker)
- [x] **DEPLOY-02**: Entire system starts with single command (docker-compose up)
- [x] **DEPLOY-03**: System uses async job queue for long-running inference tasks

### Error Handling

- [x] **ERR-01**: System validates input and returns clear error messages for invalid uploads
- [x] **ERR-02**: System handles model failures (OOM, convergence) gracefully with error status
- [x] **ERR-03**: System handles system resource issues (disk, memory) gracefully

### Documentation

- [ ] **DOC-01**: README.md with execution instructions and API usage examples
- [ ] **DOC-02**: architecture.md explaining system design and key decisions
- [ ] **DOC-03**: Example outputs included (3D results, preview images, quality JSON)

## v2 Requirements (Deferred)

- Webhook notifications for job completion
- Multiple output formats (GLB, GLTF in addition to OBJ)
- Mesh topology validation (watertight, non-manifold detection)
- Turntable video preview (.mp4)
- Batch job submission

## Out of Scope

- Web UI — API-only as specified in assignment
- Model training — inference only
- Multiple GPU support — single RTX 3090 target
- Production hardening (auth, rate limiting) — demo/assessment scope
- Real-time streaming of partial results

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| API-01 | Phase 2 | Complete |
| API-02 | Phase 2 | Complete |
| API-03 | Phase 5 | Complete |
| API-04 | Phase 2 | Complete |
| API-05 | Phase 1 | Complete |
| MODEL-01 | Phase 3.1 | Complete |
| MODEL-02 | Phase 3.1 | Complete |
| MODEL-03 | Phase 3 | Complete |
| QUAL-01 | Phase 4 | Complete |
| QUAL-02 | Phase 4 | Complete |
| QUAL-03 | Phase 4 | Complete |
| QUAL-04 | Phase 4 | Complete |
| OUT-01 | Phase 3 | Complete |
| OUT-02 | Phase 4 | Complete |
| DEPLOY-01 | Phase 1 | Complete |
| DEPLOY-02 | Phase 1 | Complete |
| DEPLOY-03 | Phase 2 | Complete |
| ERR-01 | Phase 5 | Complete |
| ERR-02 | Phase 5 | Complete |
| ERR-03 | Phase 5 | Complete |
| DOC-01 | Phase 6 | Pending |
| DOC-02 | Phase 6 | Pending |
| DOC-03 | Phase 6 | Pending |

---
*Generated from requirements scoping session*
