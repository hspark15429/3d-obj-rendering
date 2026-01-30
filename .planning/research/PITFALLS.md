# Domain Pitfalls: 3D Reconstruction API with ReconViaGen and nvdiffrec

**Domain:** Docker-based 3D reconstruction ML inference API
**Target Hardware:** RTX 3090 (24GB VRAM)
**Models:** ReconViaGen, nvdiffrec
**Researched:** 2026-01-30
**Confidence:** HIGH (based on official documentation, GitHub issues, and recent 2026 research)

---

## Critical Pitfalls

These mistakes cause rewrites, complete project failures, or days of debugging. Address in Phase 1-2.

### Pitfall 1: CUDA/PyTorch/Library Version Mismatches in Docker

**What goes wrong:**
Docker container builds successfully but PyTorch can't detect GPU (`torch.cuda.is_available()` returns `False`), or imports fail with "CUDA version mismatch" errors. This is the #1 blocker for ML Docker projects.

**Why it happens:**
- **Root cause:** Three-way compatibility matrix between host NVIDIA driver, container CUDA toolkit, and PyTorch CUDA version
- ReconViaGen requires PyTorch 2.4.0 with specific CUDA version
- spconv 2.3.6 requires exact CUDA version match (hardcoded `/usr/local/cuda` path)
- xformers 0.0.27.post2 must match PyTorch version exactly
- nvdiffrast compiles at runtime requiring nvcc compiler + matching CUDA toolkit

**Consequences:**
- 1-3 days lost debugging "why doesn't GPU work in Docker"
- Requires complete container rebuild for each attempt
- Tight timeline blown on environment issues instead of API development

**Prevention:**
1. **Verify compatibility chain BEFORE Docker build:**
   ```bash
   # On host
   nvidia-smi  # Check driver version (e.g., 520+)

   # In Dockerfile, align all versions:
   # - Base image CUDA version (e.g., nvidia/cuda:11.8.0-devel-ubuntu22.04)
   # - PyTorch CUDA version (torch 2.4.0+cu118)
   # - spconv CUDA version (spconv-cu118==2.3.6)
   # - xformers CUDA version (install from matching index)
   ```

2. **Use `-devel` base images, NOT `-runtime`:**
   - nvdiffrast and spconv need nvcc compiler
   - `nvidia/cuda:11.8.0-devel-ubuntu22.04` NOT `nvidia/cuda:11.8.0-runtime-ubuntu22.04`

3. **Install in strict order:**
   ```dockerfile
   # 1. PyTorch FIRST (before everything)
   RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

   # 2. CUDA-specific libraries matching PyTorch CUDA version
   RUN pip install spconv-cu118==2.3.6

   # 3. xformers from matching index
   RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118

   # 4. Everything else
   RUN pip install -r requirements.txt
   ```

4. **Test GPU detection in build:**
   ```dockerfile
   RUN python -c "import torch; assert torch.cuda.is_available(), 'GPU not detected'"
   ```

**Detection:**
- `nvidia-smi` works but `torch.cuda.is_available()` is False
- Import errors: "CUDA version X.Y does not match compiled version Z.W"
- spconv errors: "cannot find /usr/local/cuda"
- xformers errors: "torch version mismatch"

**Phase mapping:** Phase 1 (Docker Environment Setup)

---

### Pitfall 2: Docker Shared Memory Exhaustion Causing Silent Training Failures

**What goes wrong:**
nvdiffrec training starts but crashes mid-training with cryptic "std::runtime_error: Attempted to free arena memory that was not allocated" or silently produces incomplete results (only first pass, missing second optimization pass).

**Why it happens:**
- **Root cause:** Docker containers default to 64MB `/dev/shm` (shared memory)
- PyTorch DataLoader with `num_workers > 0` uses shared memory for inter-process communication
- nvdiffrec's two-pass training (dmtet_mesh + mesh refinement) loads large batches into shared memory
- RTX 3090 with 24GB VRAM encourages large batch sizes, amplifying the problem

**Consequences:**
- Training appears to work but produces corrupted/incomplete outputs
- Second optimization pass silently fails
- 2-4 hours wasted debugging "why is my mesh quality bad" when it's actually incomplete
- Difficult to diagnose: error occurs deep in training loop, not at startup

**Prevention:**
1. **ALWAYS set shared memory in docker run:**
   ```bash
   docker run --gpus all --shm-size=16G ...
   # OR
   docker run --gpus all --ipc=host ...
   ```

2. **For docker-compose:**
   ```yaml
   services:
     api:
       shm_size: '16gb'
       # OR
       ipc: host
   ```

3. **Add to Dockerfile comments/documentation:**
   ```dockerfile
   # REQUIRED: Run with --shm-size=16G or --ipc=host
   # nvdiffrec training will FAIL SILENTLY without adequate shared memory
   ```

4. **Validate in startup script:**
   ```python
   import os
   shm_size = os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize
   assert shm_size > 8e9, f"Shared memory too low: {shm_size/1e9:.1f}GB. Run with --shm-size=16G"
   ```

**Detection:**
- Training completes first pass but not second
- Errors mentioning "arena memory" or segmentation faults
- DataLoader hanging or crashing with `num_workers > 0`
- `df -h` shows `/dev/shm` is 64M

**Phase mapping:** Phase 1 (Docker Environment Setup) - must be in docker run documentation

**Source:** [nvdiffrec Issue #52](https://github.com/NVlabs/nvdiffrec/issues/52)

---

### Pitfall 3: Blocking Event Loop with Synchronous ML Inference

**What goes wrong:**
FastAPI appears to hang under load. Simple health checks take 30+ seconds. API becomes unresponsive when processing a single reconstruction job.

**Why it happens:**
- **Root cause:** ML inference is CPU/GPU-bound synchronous work
- FastAPI async endpoints run on single event loop thread
- Synchronous model inference (forward pass) blocks the entire event loop
- ReconViaGen + nvdiffrec can take 2-10 minutes per job
- During that time, ZERO other requests can be processed (including health checks)

**Consequences:**
- API appears "broken" or "crashed" to users
- Load balancer health checks fail, killing healthy containers
- Cannot handle concurrent requests
- Development appears to work (single request) but production fails immediately

**Prevention:**
1. **NEVER run inference in FastAPI route handler directly:**
   ```python
   # WRONG - blocks event loop for minutes
   @app.post("/reconstruct")
   async def reconstruct(images: List[UploadFile]):
       mesh = model.reconstruct(images)  # 5 min blocking call
       return mesh
   ```

2. **Use async task queue (Celery/RQ) for inference:**
   ```python
   # CORRECT - returns immediately, work happens in background
   @app.post("/reconstruct")
   async def reconstruct(images: List[UploadFile]):
       task = celery_app.send_task('reconstruct_job', args=[images])
       return {"job_id": task.id, "status": "queued"}

   @app.get("/jobs/{job_id}")
   async def get_job_status(job_id: str):
       task = AsyncResult(job_id)
       return {"status": task.state, "result": task.result}
   ```

3. **Return 202 Accepted immediately:**
   - API accepts request, returns job ID instantly (< 100ms)
   - Client polls `/jobs/{id}` for status
   - Prevents timeout issues at every layer (nginx, load balancer, client)

4. **Keep API layer purely async I/O:**
   - Validation, upload to S3, queue submission = async
   - Actual reconstruction = background worker (separate process/container)

**Detection:**
- API response times in minutes instead of milliseconds
- Health check endpoints timing out
- Single request blocks all others
- Load testing shows concurrency = 1 regardless of workers

**Phase mapping:** Phase 2 (API Architecture) - core architectural decision

**Sources:**
- [Async ML Inference API Pitfalls](https://shiladityamajumder.medium.com/async-apis-with-fastapi-patterns-pitfalls-best-practices-2d72b2b66f25)
- [Breaking Up With FastAPI for ML Serving](https://bentoml.com/blog/breaking-up-with-flask-amp-fastapi-why-ml-model-serving-requires-a-specialized-framework)

---

### Pitfall 4: GPU Memory Fragmentation Causing OOM Despite Available VRAM

**What goes wrong:**
PyTorch throws "CUDA out of memory" errors when nvidia-smi shows 18GB free on RTX 3090. First job succeeds, second job fails. Memory appears "leaked" but `torch.cuda.empty_cache()` doesn't help.

**Why it happens:**
- **Root cause:** GPU memory fragmentation prevents large contiguous allocations
- PyTorch allocates/frees tensors in irregular patterns during reconstruction
- Small unusable gaps accumulate between allocated blocks
- Model needs 8GB contiguous block, but only has 8GB split into 500MB fragments
- Recent research (ASPLOS 2026) identifies this as primary bottleneck for large 3D scenes

**Consequences:**
- "Works on my laptop, fails in production" (different allocation patterns)
- First request works, subsequent requests fail (fragmentation accumulates)
- Appears to be memory leak but isn't
- Forces smaller batch sizes than hardware should support
- RTX 3090's 24GB becomes effectively 12-16GB usable

**Prevention:**
1. **Enable expandable segments (PyTorch 2.0+):**
   ```bash
   # In Dockerfile or docker-compose
   ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

2. **Clear cache between jobs:**
   ```python
   import torch
   import gc

   def run_reconstruction(inputs):
       try:
           result = model(inputs)
           return result
       finally:
           # Aggressive cleanup
           torch.cuda.empty_cache()
           gc.collect()
   ```

3. **Limit worker concurrency on shared GPU:**
   - Don't run multiple reconstruction jobs simultaneously on same GPU
   - Use Celery worker concurrency=1 for GPU workers
   - Queue jobs instead of parallel execution

4. **Restart workers periodically:**
   ```python
   # In Celery worker
   from celery import signals

   @signals.task_postrun.connect
   def task_postrun_handler(task_id, task, **kwargs):
       # Restart worker after N tasks to clear fragmentation
       if task.request.id % 10 == 0:
           worker.pool.restart()
   ```

5. **Monitor fragmentation:**
   ```python
   def check_gpu_fragmentation():
       stats = torch.cuda.memory_stats()
       allocated = stats['allocated_bytes.all.current']
       reserved = stats['reserved_bytes.all.current']
       fragmentation = (reserved - allocated) / reserved
       if fragmentation > 0.3:
           logger.warning(f"GPU fragmentation: {fragmentation:.1%}")
   ```

**Detection:**
- OOM errors with plenty of free memory shown by nvidia-smi
- "CUDA error: out of memory" but `torch.cuda.memory_allocated()` shows headroom
- Error message suggests "setting max_split_size_mb"
- Memory usage grows over time despite same workload

**Phase mapping:** Phase 3 (Job Queue & Worker Management)

**Sources:**
- [CLM: Removing GPU Memory Barrier for 3D Gaussian Splatting (ASPLOS 2026)](https://arxiv.org/html/2511.04951)
- [Mitigating CUDA GPU Memory Fragmentation](https://discuss.pytorch.org/t/mitigating-cuda-gpu-memory-fragmentation-and-oom-issues/108203)
- [PyTorch CUDA OOM Issues](https://github.com/pytorch/pytorch/issues/67680)

---

### Pitfall 5: Camera Pose Estimation Failures in Sparse/Wide-Baseline Views

**What goes wrong:**
ReconViaGen produces garbage meshes (distorted, incomplete, or "blob" artifacts) despite following all setup instructions correctly. Same code works on example data but fails on user-uploaded images.

**Why it happens:**
- **Root cause:** Multi-view reconstruction fundamentally depends on camera pose estimation
- ReconViaGen specifically includes "refined camera pose estimation strategy" because original methods fail
- Sparse views (< 10 images) create ambiguity - insufficient overlap for feature matching
- Wide baselines (camera moved significantly between shots) prevent finding common keypoints
- Occlusions and limited overlap cause 2D feature matching to fail
- Recent research (2024-2026) confirms "no method yet capable of solving all challenges"

**Consequences:**
- Works on curated datasets, fails on real user data
- Blamed on "bad user input" when it's actually algorithmic limitation
- Forces rigid input requirements (min 20 images, max 15° rotation between views)
- Wastes time debugging model when problem is input validation

**Prevention:**
1. **Validate inputs BEFORE queuing expensive reconstruction:**
   ```python
   def validate_camera_inputs(images):
       # Minimum viable set
       if len(images) < 12:
           raise ValidationError("Need at least 12 images for reliable reconstruction")

       # Feature matching test
       matches = compute_pairwise_matches(images)
       avg_matches = sum(matches.values()) / len(matches)
       if avg_matches < 100:
           raise ValidationError("Insufficient overlap between views")

       # Baseline check (if EXIF available)
       if has_exif_data(images):
           max_baseline = compute_max_baseline(images)
           if max_baseline > MAX_SAFE_BASELINE:
               raise ValidationError("Camera positions too far apart")
   ```

2. **Set clear input requirements in API docs:**
   ```yaml
   # OpenAPI spec
   /reconstruct:
     post:
       description: |
         REQUIREMENTS for successful reconstruction:
         - Minimum 12 images, recommended 20-30
         - Maximum 15° rotation between consecutive views
         - Consistent lighting across views
         - Sufficient overlap (object visible in 80%+ of each image)
   ```

3. **Return meaningful errors, not garbage:**
   ```python
   # In reconstruction worker
   def check_pose_estimation_quality(poses):
       if poses.uncertainty > THRESHOLD:
           raise ReconstructionError(
               "Camera pose estimation failed. Try: "
               "1) Add more images with better overlap "
               "2) Reduce distance between camera positions "
               "3) Ensure consistent lighting"
           )
   ```

4. **Provide example dataset for testing:**
   - Include "good" example with known-working camera poses
   - Include "bad" examples that should fail validation
   - Helps users understand requirements

**Detection:**
- Mesh has "blob" artifacts or background collapse
- Texture appears smeared or misaligned
- Mesh geometry doesn't match input images
- Works on examples, fails on user data
- Errors deep in reconstruction (not at start)

**Phase mapping:** Phase 3 (Input Validation) or Phase 4 (Quality Checks)

**Sources:**
- [Sparse-View 3D Reconstruction Challenges (2024)](https://arxiv.org/html/2507.16406v1)
- [Few-View Camera Pose Estimation](https://arxiv.org/html/2212.04492v3)
- [Multi-View Reconstruction Survey (2026)](https://www.sciencedirect.com/science/article/pii/S0262885625000253)

---

## Moderate Pitfalls

These cause delays (hours to days) or technical debt but are fixable.

### Pitfall 6: API Gateway Timeout for Long-Running Jobs

**What goes wrong:**
API returns 504 Gateway Timeout after 30 seconds even though reconstruction job is still running. Client thinks request failed, but job completes successfully minutes later (wasted compute).

**Why it happens:**
- Load balancers / API gateways have hard timeout limits (AWS API Gateway: 29s, nginx default: 60s)
- Reconstruction takes 2-10 minutes
- Client waiting for synchronous response hits timeout
- Job continues running but response is lost

**Prevention:**
1. **Use async job pattern (see Pitfall 3):**
   - POST returns 202 Accepted immediately with job_id
   - GET /jobs/{id} polls status (cheap, fast, no timeout)

2. **Document timeout expectations:**
   ```python
   # In API documentation
   """
   Reconstruction jobs take 2-10 minutes.
   Do NOT wait for synchronous response.
   Use job status endpoint to poll completion.
   """
   ```

3. **Set realistic timeouts in infrastructure:**
   ```nginx
   # nginx.conf
   proxy_read_timeout 300s;  # 5 min - only if using sync pattern
   proxy_connect_timeout 60s;
   proxy_send_timeout 60s;
   ```

**Detection:**
- 504 errors in production, works in development
- Jobs complete successfully but clients see errors
- Timeout errors at exactly 29s (API Gateway) or 60s (nginx)

**Phase mapping:** Phase 2 (API Architecture)

**Source:** [API Gateway Timeout Solutions](https://www.catchpoint.com/api-monitoring-tools/api-gateway-timeout)

---

### Pitfall 7: Missing nvdiffrast Build Dependencies in Docker

**What goes wrong:**
`pip install nvdiffrast` succeeds but import fails with "No module named 'nvdiffrast_plugin'" or compilation errors at runtime.

**Why it happens:**
- nvdiffrast compiles GPU kernels at first import (not during pip install)
- Requires full CUDA toolkit with nvcc compiler
- Runtime-only Docker images (nvidia/cuda:*-runtime) lack build tools
- First import triggers compilation, fails without development dependencies

**Prevention:**
1. **Use `-devel` base image:**
   ```dockerfile
   FROM nvidia/cuda:11.8.0-devel-ubuntu22.04  # NOT -runtime
   ```

2. **Install build dependencies:**
   ```dockerfile
   RUN apt-get update && apt-get install -y \
       build-essential \
       ninja-build \
       git
   ```

3. **Pre-compile during build:**
   ```dockerfile
   RUN pip install nvdiffrast
   RUN python -c "import nvdiffrast; print('nvdiffrast compiled successfully')"
   ```

**Detection:**
- Import errors: "nvdiffrast_plugin not found"
- Runtime compilation errors
- Works on host, fails in container

**Phase mapping:** Phase 1 (Docker Environment)

**Source:** [nvdiffrast Installation Issues](https://github.com/NVlabs/nvdiffrast/issues/170)

---

### Pitfall 8: spconv CUDA Path Hardcoding

**What goes wrong:**
spconv installation fails with "cannot find /usr/local/cuda" even though CUDA is installed in conda environment or different path.

**Why it happens:**
- spconv hardcodes `/usr/local/cuda` path in `cumm/common.py`
- Doesn't respect CUDA_HOME environment variable
- Conda installs CUDA to `$CONDA_PREFIX/pkgs/cuda-toolkit-*`

**Prevention:**
1. **Symlink CUDA to expected location:**
   ```dockerfile
   RUN ln -s /usr/local/cuda-11.8 /usr/local/cuda
   ```

2. **Use pre-built wheels matching CUDA version:**
   ```dockerfile
   RUN pip install spconv-cu118==2.3.6  # NOT generic 'spconv'
   ```

3. **Set CUDA environment variables:**
   ```dockerfile
   ENV CUDA_HOME=/usr/local/cuda-11.8
   ENV PATH=/usr/local/cuda-11.8/bin:$PATH
   ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
   ```

**Detection:**
- spconv import errors mentioning CUDA path
- Compilation errors during installation
- "ambiguous overload" errors during build

**Phase mapping:** Phase 1 (Docker Environment)

**Source:** [spconv CUDA Path Issues](https://github.com/traveller59/spconv/issues/658)

---

### Pitfall 9: Forgetting ReconViaGen Recursive Submodules

**What goes wrong:**
Git clone succeeds but running code fails with import errors for submodule dependencies.

**Why it happens:**
- ReconViaGen uses git submodules for some dependencies
- Regular `git clone` doesn't fetch submodules
- Missing dependencies only discovered at runtime

**Prevention:**
1. **Clone with submodules:**
   ```dockerfile
   RUN git clone --recursive https://github.com/estheryang11/ReconViaGen.git
   ```

2. **Or init after clone:**
   ```dockerfile
   RUN git clone https://github.com/estheryang11/ReconViaGen.git && \
       cd ReconViaGen && \
       git submodule update --init --recursive
   ```

**Detection:**
- Import errors for submodule packages
- Missing directories in repo
- Works for maintainers (who have submodules), fails for new users

**Phase mapping:** Phase 1 (Docker Environment)

**Source:** [ReconViaGen README](https://github.com/estheryang11/ReconViaGen)

---

### Pitfall 10: Not Validating Mesh Quality Before Returning Results

**What goes wrong:**
API successfully returns mesh files but they're unusable (missing geometry, texture artifacts, broken topology). Users discover issues only after download.

**Why it happens:**
- Reconstruction "completes" even when results are poor quality
- No automated quality checks before marking job as "success"
- Mesh may be technically valid but perceptually broken

**Prevention:**
1. **Automated quality checks:**
   ```python
   def validate_mesh_quality(mesh_path):
       mesh = trimesh.load(mesh_path)

       # Geometric checks
       if mesh.vertices.shape[0] < 1000:
           raise QualityError("Too few vertices - reconstruction failed")

       if not mesh.is_watertight:
           logger.warning("Mesh not watertight - may have holes")

       # Texture checks
       if has_texture(mesh):
           texture_resolution = get_texture_resolution(mesh)
           if texture_resolution < (512, 512):
               logger.warning("Low texture resolution")

       # Surface coverage
       surface_area = mesh.area
       bounding_box_area = mesh.bounding_box.area
       coverage = surface_area / bounding_box_area
       if coverage < 0.3:
           raise QualityError("Insufficient surface coverage")
   ```

2. **Render validation images:**
   ```python
   # Render mesh from multiple angles
   validation_renders = render_mesh_previews(mesh, n_views=8)

   # Return with result for user inspection
   return {
       "mesh_url": mesh_url,
       "preview_images": validation_renders,
       "quality_score": quality_score,
       "warnings": warnings
   }
   ```

3. **Quality-based status:**
   ```python
   # Don't just return success/failure
   return {
       "status": "completed",
       "quality": "high" | "medium" | "low",
       "warnings": [...],
       "mesh_url": url
   }
   ```

**Detection:**
- Users report "broken" or "unusable" results
- High download rate but low usage rate
- Support tickets about mesh quality

**Phase mapping:** Phase 4 (Output Validation)

**Sources:**
- [3D Mesh Quality Metrics (2024)](https://www.sloyd.ai/blog/top-7-metrics-for-evaluating-3d-model-quality)
- [Textured Mesh Quality Assessment](https://dl.acm.org/doi/10.1145/3592786)

---

## Minor Pitfalls

These cause annoyance but are easily fixable. Address as discovered.

### Pitfall 11: Not Pinning Python Package Versions

**What goes wrong:**
Docker build that worked yesterday fails today because dependency released breaking change.

**Prevention:**
```dockerfile
# Pin EVERYTHING
RUN pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    spconv-cu118==2.3.6 \
    xformers==0.0.27.post2 \
    # etc
```

**Phase mapping:** Phase 1 (Docker Environment)

---

### Pitfall 12: Development vs Production Base Images

**What goes wrong:**
Using heavyweight development base image (Ubuntu + full CUDA toolkit) in production wastes 5-10GB per container.

**Prevention:**
Multi-stage Docker build:
```dockerfile
# Build stage - full devel image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder
RUN pip install nvdiffrast  # compiles here

# Runtime stage - minimal runtime image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
```

**Phase mapping:** Phase 5 (Production Optimization) - not critical for MVP

---

### Pitfall 13: Not Logging GPU Metrics

**What goes wrong:**
Production issues hard to diagnose without visibility into GPU utilization, memory, temperature.

**Prevention:**
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def log_gpu_stats():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    logger.info(f"GPU: {util.gpu}% | Memory: {info.used/1e9:.1f}/{info.total/1e9:.1f}GB | Temp: {temp}°C")
```

**Phase mapping:** Phase 3 (Observability)

---

### Pitfall 14: File Upload Size Limits

**What goes wrong:**
Users upload 20 high-res images (200MB+) and request times out or fails silently.

**Prevention:**
```python
# In FastAPI
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_request_size=500 * 1024 * 1024  # 500MB
)

# In nginx
client_max_body_size 500M;
```

**Phase mapping:** Phase 2 (API Configuration)

---

### Pitfall 15: Not Handling Model Loading Cold Starts

**What goes wrong:**
First request takes 60+ seconds (model loading time) while subsequent requests are fast. Appears broken.

**Prevention:**
```python
# Load model at worker startup, not first request
from celery import signals

@signals.worker_process_init.connect
def init_worker(**kwargs):
    global model
    model = load_reconstruction_model()
    logger.info("Model pre-loaded and ready")
```

**Phase mapping:** Phase 3 (Worker Initialization)

---

## Phase-Specific Warnings

| Phase | Focus | Likely Pitfall | Mitigation |
|-------|-------|---------------|------------|
| Phase 1: Docker Environment | Getting models running in container | **Pitfall 1** (version mismatches), **Pitfall 2** (shared memory), **Pitfall 7** (nvdiffrast deps) | Strict version alignment, always use --shm-size=16G, use -devel images |
| Phase 2: API Architecture | Endpoint design and async patterns | **Pitfall 3** (blocking event loop), **Pitfall 6** (timeouts) | Always use async task queue, return 202 Accepted immediately |
| Phase 3: Job Queue & Workers | Celery/RQ integration, GPU management | **Pitfall 4** (GPU fragmentation), **Pitfall 15** (cold starts) | Enable expandable_segments, pre-load models, limit concurrency |
| Phase 4: Input/Output Validation | Handling user data | **Pitfall 5** (pose estimation), **Pitfall 10** (quality validation) | Validate before queuing, check mesh quality before returning |
| Phase 5: Production Optimization | Performance and reliability | **Pitfall 12** (image sizes), **Pitfall 13** (observability) | Multi-stage builds, comprehensive logging |

---

## Quick Reference: Must-Have Docker Run Flags

For 3D reconstruction APIs, these are **non-negotiable**:

```bash
docker run \
  --gpus all \                    # GPU access
  --shm-size=16G \                # Shared memory (Pitfall 2)
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \  # Fragmentation (Pitfall 4)
  -v /path/to/data:/data \        # Persistent storage
  -p 8000:8000 \                  # API port
  my-reconstruction-api
```

**For docker-compose:**
```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: '16gb'
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Confidence Assessment

| Category | Confidence | Source Quality | Notes |
|----------|------------|----------------|-------|
| Docker/CUDA issues | **HIGH** | Official docs + GitHub issues + 2026 research | Well-documented, recent sources |
| Async API patterns | **HIGH** | Multiple authoritative sources + 2026 articles | Industry standard patterns |
| GPU memory management | **HIGH** | PyTorch docs + ASPLOS 2026 research | Cutting-edge research confirms issues |
| Camera pose estimation | **MEDIUM-HIGH** | Recent academic papers (2024-2026) | Research-backed but still evolving |
| nvdiffrec specifics | **MEDIUM** | GitHub issues from 2022 | Older but directly relevant |
| ReconViaGen specifics | **MEDIUM-LOW** | README only, limited documentation | Unofficial implementation, less battle-tested |

---

## Gaps & Research Flags

**Needs phase-specific investigation later:**
- Phase 3: Optimal Celery vs RQ vs custom queue for GPU workloads
- Phase 4: Specific quality thresholds for mesh validation (domain-dependent)
- Phase 5: Multi-GPU strategies if scaling beyond 1 RTX 3090

**Could not verify:**
- ReconViaGen production usage examples (unofficial implementation)
- Specific memory requirements for different input sizes (needs empirical testing)

---

## Sources

### Critical Sources (HIGH confidence)
- [nvdiffrec Docker Issue #52](https://github.com/NVlabs/nvdiffrec/issues/52) - Shared memory pitfall
- [CLM-GS: GPU Memory Barrier Research (ASPLOS 2026)](https://arxiv.org/html/2511.04951) - Memory fragmentation
- [PyTorch CUDA OOM Discussion](https://discuss.pytorch.org/t/mitigating-cuda-gpu-memory-fragmentation-and-oom-issues/108203)
- [Async ML Inference Pitfalls](https://shiladityamajumder.medium.com/async-apis-with-fastapi-patterns-pitfalls-best-practices-2d72b2b66f25)
- [CUDA Version Compatibility Guide](https://www.oreateai.com/blog/systematic-solutions-for-compatibility-issues-between-pytorch-and-cuda-versions/5548a381b8d2622e73900c477c3b22cb)

### Supporting Sources (MEDIUM confidence)
- [Sparse-View 3D Reconstruction Survey](https://arxiv.org/html/2507.16406v1)
- [Multi-View Pose Estimation Survey (2026)](https://www.sciencedirect.com/science/article/pii/S0262885625000253)
- [spconv CUDA Path Issues](https://github.com/traveller59/spconv/issues/658)
- [nvdiffrast Installation Issues](https://github.com/NVlabs/nvdiffrast/issues/170)
- [3D Mesh Quality Metrics](https://www.sloyd.ai/blog/top-7-metrics-for-evaluating-3d-model-quality)
- [API Gateway Timeouts](https://www.catchpoint.com/api-monitoring-tools/api-gateway-timeout)

### Repository Documentation
- [ReconViaGen GitHub](https://github.com/estheryang11/ReconViaGen)
- [nvdiffrec GitHub](https://github.com/NVlabs/nvdiffrec)
- [xFormers Compatibility](https://www.felixsanz.dev/articles/compatibility-between-pytorch-cuda-and-xformers-versions)
