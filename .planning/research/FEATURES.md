# Feature Landscape

**Domain:** 3D Reconstruction API
**Researched:** 2026-01-30
**Confidence:** MEDIUM

## Table Stakes

Features users expect. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Async Job Management** | 3D reconstruction is compute-intensive (minutes to hours). Synchronous APIs are non-starters. | Medium | POST /jobs returns 202 Accepted + job ID. Standard pattern across industry. |
| **Job Status Polling** | Users need to check job progress. GET /jobs/{id} with status enum. | Low | Status values: queued, processing, completed, failed. Include progress percentage if possible. |
| **Multi-format Output** | Different consumers need different formats. GLB for web/AR, OBJ for compatibility, GLTF for editing. | Medium | GLB is primary (compact, single file). OBJ for legacy tools. GLTF for team workflows. |
| **Result Download** | GET /jobs/{id}/result must deliver mesh + textures. | Low | Single endpoint returns ZIP or primary format (GLB self-contained). |
| **Basic Error Reporting** | When jobs fail, users need to know why. | Low | Enum: invalid_input, processing_error, timeout. Human-readable message field. |
| **Multi-view Input Support** | Reconstruction requires multiple views. Must accept image sets (RGB). | Medium | Batch upload or multi-part form. Include camera poses/intrinsics if available. |
| **Thumbnail Generation** | Users need visual preview without downloading full mesh. | Medium | Small PNG/JPG preview of reconstructed model. Generated during processing. |
| **Quality Metrics** | Users need objective measures of reconstruction quality. | Medium | PSNR and SSIM for texture quality. Standard in 3D reconstruction research. |

## Differentiators

Features that set product apart. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Webhook Notifications** | Real-time job completion alerts instead of polling. More scalable. | Medium | Optional webhook_url parameter. POST to client URL on completion. Reduces API traffic. |
| **Model Selection** | Multiple reconstruction algorithms for different use cases. Let users choose quality vs speed tradeoff. | High | ReconViaGen (quality), nvdiffrec (speed/differentiable). Emerging: Gaussian Splatting. |
| **LOD Generation** | Generate multiple detail levels automatically. Critical for real-time apps (games, AR). | High | 3-5 LOD levels with configurable polygon targets. Requires mesh decimation pipeline. |
| **Mesh Quality Validation** | Detect topology issues (non-manifold, holes, self-intersections) before delivery. | High | Automated checks: watertightness, triangle quality (aspect ratio), normal consistency. |
| **Depth Data Integration** | Accept depth maps alongside RGB for higher quality reconstruction. | High | Depth sensors (LiDAR, structured light) significantly improve geometry accuracy. |
| **Preview LOD** | Deliver fast low-poly preview before full-resolution mesh completes. | Medium | Progressive delivery: coarse mesh in 10% of time, full quality later. Better UX. |
| **Result Caching** | Store results for retrieval after initial download. | Low | GET /jobs/{id}/result works indefinitely (or with TTL). Avoids reprocessing. |
| **Custom Quality Targets** | Let users specify polygon count, texture resolution. | Medium | Parameters: max_polygons (default 50k), texture_size (default 2048). |
| **Batch Job Management** | Submit multiple reconstruction jobs atomically. | Medium | POST /jobs/batch with array of configs. Single tracking endpoint. |

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Synchronous Processing** | 3D reconstruction takes minutes. Synchronous would timeout or block clients unnecessarily. | Always async with job ID pattern. Return 202 immediately. |
| **Automatic Re-upload on Failure** | Wastes bandwidth. Users should inspect failure reason first. | Return clear error codes. Let client decide whether to retry. |
| **Single Uber-Format** | No format serves all needs. GLB for web, OBJ for tools, GLTF for editing. | Support 2-3 key formats. Let users choose at download time. |
| **Black Box Processing** | Users can't debug or optimize if they don't understand what's happening. | Expose model selection, parameters, quality metrics. Log warnings for suboptimal input. |
| **Unlimited Job TTL** | Storage costs grow unbounded. | TTL on completed jobs (7 days default, configurable). Clear in API docs. |
| **Real-time Streaming** | Partial mesh streaming sounds cool but adds massive complexity. Most clients want final result. | Provide preview thumbnail + progress percentage. That's sufficient. |
| **Built-in Viewer** | Scope creep. 3D viewers are complex (WebGL, controls, materials). Clients have their own or can integrate existing libraries. | Return standard formats (GLB). Link to three.js or Babylon.js examples in docs. |
| **Auto-correction of Bad Input** | Silently fixing bad data hides problems. Users need to know their input is suboptimal. | Validate strictly. Return warnings for suboptimal input (low overlap, poor lighting). |

## Feature Dependencies

```
Job Submission → Job Status → Result Download
                    ↓
              Progress Updates
                    ↓
            Quality Metrics (available on completion)

Multi-view Input → Model Selection → Reconstruction → Output Format Selection
                                            ↓
                                    Thumbnail Generation
                                            ↓
                                    Quality Validation
                                            ↓
                                    LOD Generation (optional)

Depth Data (optional) → Reconstruction Quality Enhancement

Webhook Configuration → Job Completion → Webhook Delivery
```

**Key Dependencies:**
- **Thumbnail generation** requires successful reconstruction (depends on completion)
- **Quality metrics** computed during reconstruction, available with results
- **LOD generation** depends on successful mesh generation
- **Webhooks** depend on job completion detection
- **Depth data integration** affects reconstruction quality but is optional input

## MVP Recommendation

For MVP, prioritize:

### Phase 1: Core Job Pipeline
1. **Async job management** (POST /jobs, job ID return)
2. **Job status polling** (GET /jobs/{id} with status enum)
3. **Multi-view RGB input** (image upload, basic validation)
4. **Single model** (start with one algorithm, expand later)
5. **GLB output** (primary format, self-contained)
6. **Result download** (GET /jobs/{id}/result)
7. **Basic error reporting** (failure reasons, human-readable messages)

### Phase 2: Quality & Observability
8. **Thumbnail generation** (visual preview)
9. **Quality metrics** (PSNR, SSIM)
10. **Mesh validation** (topology checks, warnings)

### Phase 3: Advanced Features
11. **Model selection** (multiple algorithms)
12. **Additional formats** (OBJ, GLTF)
13. **Webhook notifications** (async completion alerts)

### Defer to post-MVP:
- **LOD generation**: High complexity, needed for real-time apps but not core reconstruction
- **Depth data integration**: Nice-to-have, requires sensor hardware
- **Preview LOD**: Progressive delivery is UX polish, not core functionality
- **Batch jobs**: Useful at scale but adds complexity
- **Custom quality targets**: Power user feature, reasonable defaults work for most

**Rationale:**
- Phase 1 establishes the fundamental async job pattern and proves reconstruction works
- Phase 2 adds visibility into quality without changing core pipeline
- Phase 3 differentiates with advanced features once basics are solid
- Deferred features are valuable but non-essential for proving product-market fit

## MVP Feature Sizing

| Feature | Estimated Effort | Justification |
|---------|-----------------|---------------|
| Async job management | 2-3 days | Standard pattern, use Redis/Postgres for queue |
| Job status API | 1 day | Simple CRUD on job table |
| Multi-view upload | 2-3 days | Multi-part form parsing, file validation, storage |
| Model integration | 3-5 days | Wrap ReconViaGen/nvdiffrec, containerize |
| GLB output | 2 days | Format conversion, texture packing |
| Result download | 1 day | Serve file from storage with correct headers |
| Error reporting | 1 day | Status classification, message templates |
| Thumbnail generation | 2-3 days | Render preview image from mesh |
| Quality metrics | 2-3 days | Integrate PSNR/SSIM computation |
| Mesh validation | 3-4 days | Topology checks (manifold, holes, self-intersection) |

**Total MVP (Phases 1-2): 19-27 days**

## Sources

### Async API Patterns
- [Asynchronous Request-Reply pattern - Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/patterns/async-request-reply)
- [Building Asynchronous APIs - Beeceptor](https://beeceptor.com/docs/building-asynchronous-apis/)
- [Managing Asynchronous Workflows with a REST API - AWS](https://aws.amazon.com/blogs/architecture/managing-asynchronous-workflows-with-a-rest-api/)
- [Design asynchronous API - REST API cookbook](https://octo-woapi.github.io/cookbook/asynchronous-api.html)

### 3D Format Standards
- [GLTF vs GLB: Which Format is Right for Your 3D Projects?](https://resources.imagine.io/blog/gltf-vs-glb-which-format-is-right-for-your-3d-projects)
- [3D File Formats: OBJ, FBX, GLB, USDZ, and Delivery](https://threedium.io/3d-model/file-formats)
- [glTF - Runtime 3D Asset Delivery](https://www.khronos.org/gltf/)

### Quality Metrics
- [PSNR vs SSIM: Complete Video Quality Metrics Analysis Guide](https://www.probe.dev/resources/psnr-ssim-quality-analysis)
- [What are the NeRF Metrics? - Radiance Fields](https://radiancefields.com/what-are-the-nerf-metrics)
- [MeshFormer: High-Quality Mesh Generation with 3D-Guided Reconstruction Model](https://arxiv.org/html/2408.10198v1)

### Mesh Quality Validation
- [Geometry Validation in FEA: How to Prepare Clean Models](https://sdcverifier.com/structural-engineering-101/geometry-validation-fea-mesh-quality/)
- [A Survey of Indicators for Mesh Quality Assessment](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14779)

### 3D Reconstruction Technologies
- [3D Gaussian Splatting vs NeRF: The End Game of 3D Reconstruction?](https://pyimagesearch.com/2024/12/09/3d-gaussian-splatting-vs-nerf-the-end-game-of-3d-reconstruction/)
- [Gaussian splatting vs. photogrammetry vs. NeRFs](https://teleport.varjo.com/blog/photogrammetry-vs-nerfs-gaussian-splatting-pros-and-cons)
- [3D Gaussian Splatting: Complete Guide (2026)](https://www.utsubo.com/blog/gaussian-splatting-guide)

### Photogrammetry APIs
- [OpenScan Cloud - Photogrammetry Web API](https://github.com/OpenScan-org/OpenScanCloud)
- [Reality Capture API - Autodesk Platform Services](https://aps.autodesk.com/developer/overview/reality-capture-api)
- [Meshy API Documentation](https://docs.meshy.ai/en/api/image-to-3d)

### Preview & LOD Generation
- [Generate Thumbnails - Model Derivative API](https://aps.autodesk.com/en/docs/model-derivative/v2/developers_guide/basics/thumbnail_generation)
- [LOD generation guidelines - Unity Asset Transformer SDK](https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/sdktips/lod-guidelines)
