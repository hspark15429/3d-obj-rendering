# Requirements

**Project:** 3D Object Reconstruction API
**Version:** v1
**Last updated:** 2026-01-30

## v1 Requirements

### API Endpoints

- [ ] **API-01**: User can submit a job with multi-view images and depth renders, selecting model type, and receive a job ID
- [ ] **API-02**: User can check job status (queued/processing/completed/failed) by job ID
- [ ] **API-03**: User can download job results (mesh, textures, preview images, quality report) by job ID
- [ ] **API-04**: User can cancel a running or queued job by job ID
- [x] **API-05**: User can check API health via GET /health endpoint

### 3D Reconstruction Models

- [ ] **MODEL-01**: System can run ReconViaGen model on multi-view RGB + depth input to produce textured mesh
- [ ] **MODEL-02**: System can run nvdiffrec model on multi-view RGB + depth input to produce textured mesh
- [ ] **MODEL-03**: User can select which model to run when submitting a job

### Quality Validation

- [ ] **QUAL-01**: System computes PSNR metric for reconstruction quality
- [ ] **QUAL-02**: System computes SSIM metric for reconstruction quality
- [ ] **QUAL-03**: System classifies result status as normal/warning/failure based on quality thresholds
- [ ] **QUAL-04**: System saves quality metrics and status to JSON file with results

### Output & Preview

- [ ] **OUT-01**: System generates textured mesh output (OBJ/PLY with texture files)
- [ ] **OUT-02**: System generates static preview images from multiple angles

### Deployment

- [x] **DEPLOY-01**: System runs in Docker container with GPU support (nvidia-docker)
- [x] **DEPLOY-02**: Entire system starts with single command (docker-compose up)
- [ ] **DEPLOY-03**: System uses async job queue for long-running inference tasks

### Error Handling

- [ ] **ERR-01**: System validates input and returns clear error messages for invalid uploads
- [ ] **ERR-02**: System handles model failures (OOM, convergence) gracefully with error status
- [ ] **ERR-03**: System handles system resource issues (disk, memory) gracefully

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
| API-01 | Phase 2 | Pending |
| API-02 | Phase 2 | Pending |
| API-03 | Phase 5 | Pending |
| API-04 | Phase 2 | Pending |
| API-05 | Phase 1 | Complete |
| MODEL-01 | Phase 3 | Pending |
| MODEL-02 | Phase 3 | Pending |
| MODEL-03 | Phase 3 | Pending |
| QUAL-01 | Phase 4 | Pending |
| QUAL-02 | Phase 4 | Pending |
| QUAL-03 | Phase 4 | Pending |
| QUAL-04 | Phase 4 | Pending |
| OUT-01 | Phase 3 | Pending |
| OUT-02 | Phase 4 | Pending |
| DEPLOY-01 | Phase 1 | Complete |
| DEPLOY-02 | Phase 1 | Complete |
| DEPLOY-03 | Phase 2 | Pending |
| ERR-01 | Phase 5 | Pending |
| ERR-02 | Phase 5 | Pending |
| ERR-03 | Phase 5 | Pending |
| DOC-01 | Phase 6 | Pending |
| DOC-02 | Phase 6 | Pending |
| DOC-03 | Phase 6 | Pending |

---
*Generated from requirements scoping session*
