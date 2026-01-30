# Architecture Patterns

**Domain:** Docker-based Async ML Inference API for 3D Reconstruction
**Researched:** 2026-01-30
**Confidence:** MEDIUM

## Recommended Architecture

Modern ML inference APIs in 2026 follow a **microservices architecture with async job queue pattern**. For this 3D reconstruction system running ReconViaGen and nvdiffrec models, the recommended architecture is:

```
┌─────────────┐
│   Client    │
│ (Web/CLI)   │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────────────────────────────────────┐
│          FastAPI Server (REST API)          │
│  - Upload endpoints                         │
│  - Job status/retrieval endpoints           │
│  - Health checks                            │
└──────┬──────────────────────────────────────┘
       │
       │ Publishes jobs
       ▼
┌─────────────────────────────────────────────┐
│          Redis (Queue + Results)            │
│  - Task queue (job queue)                   │
│  - Job status tracking                      │
│  - Result metadata cache                    │
└──────┬──────────────────────────────────────┘
       │
       │ Workers consume
       ▼
┌─────────────────────────────────────────────┐
│      Celery Workers (GPU-enabled)           │
│  - ReconViaGen worker pool                  │
│  - nvdiffrec worker pool                    │
│  - Model loading & inference                │
│  - Result generation                        │
└──────┬──────────────────────────────────────┘
       │
       │ Writes results
       ▼
┌─────────────────────────────────────────────┐
│       Storage Layer (MinIO or Volume)       │
│  - Input images                             │
│  - Output meshes (.obj, .ply)               │
│  - Textures                                 │
│  - Quality reports                          │
│  - Preview renders                          │
└─────────────────────────────────────────────┘
       │
       │ Optional monitoring
       ▼
┌─────────────────────────────────────────────┐
│         Flower (Celery monitoring)          │
│  - Real-time worker status                  │
│  - Task progress tracking                   │
│  - Failure diagnostics                      │
└─────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With | Container |
|-----------|---------------|-------------------|-----------|
| **FastAPI Server** | REST API gateway, request validation, file uploads, job submission, result retrieval | Redis (publish jobs), Storage (read/write uploads), Client (HTTP) | `api` |
| **Redis** | Message broker, task queue, job state tracking, result metadata caching | FastAPI (receives jobs), Celery Workers (delivers jobs) | `redis` |
| **Celery Workers** | ML model inference execution, GPU workload processing, result generation | Redis (consume jobs, update status), Storage (write results) | `worker-reconviagen`, `worker-nvdiffrec` |
| **Storage (MinIO)** | Persistent object storage for uploads and results, S3-compatible API | FastAPI (uploads), Workers (results) | `minio` (or Docker volume) |
| **Flower** | Monitoring dashboard, worker health, task inspection | Redis (read worker state), Celery Workers (monitor) | `flower` (optional) |

### Data Flow

**Upload → Queue → Inference → Results:**

1. **Client Upload (HTTP POST)**
   - Client POSTs images to `/api/v1/reconstruct` endpoint
   - FastAPI validates request, saves images to storage
   - Returns job ID immediately (async pattern)

2. **Job Queuing**
   - FastAPI publishes job to Redis queue with job metadata
   - Job includes: model type (ReconViaGen/nvdiffrec), input paths, parameters
   - Client can poll `/api/v1/jobs/{job_id}` for status

3. **Worker Processing**
   - Celery worker pulls job from Redis queue when capacity available
   - Worker loads ML model (cached in GPU memory if possible)
   - Worker reads input images from storage
   - Worker executes inference (GPU-accelerated)
   - Worker generates outputs: mesh, texture, preview, quality report

4. **Result Storage**
   - Worker writes results to storage with predictable paths
   - Worker updates job status in Redis: `pending → processing → complete|failed`
   - Worker stores result metadata (file paths, quality metrics) in Redis

5. **Result Retrieval**
   - Client polls job status, receives `complete` status
   - Client GET `/api/v1/jobs/{job_id}/results` to retrieve download URLs
   - FastAPI generates pre-signed URLs (if using MinIO) or streams files directly

## Docker Compose Structure

### Service Definitions

```yaml
version: '3.8'

services:
  # REST API Gateway
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - STORAGE_URL=http://minio:9000
    volumes:
      - ./uploads:/app/uploads  # Temporary uploads (or use MinIO)
    depends_on:
      - redis
      - minio

  # Message Broker & Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  # Celery Worker: ReconViaGen
  worker-reconviagen:
    build:
      context: ./workers
      dockerfile: Dockerfile.gpu
    environment:
      - REDIS_URL=redis://redis:6379/0
      - MODEL_TYPE=reconviagen
      - NVIDIA_VISIBLE_DEVICES=0  # First GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    depends_on:
      - redis
      - minio
    volumes:
      - model-cache:/app/models  # Persistent model weights

  # Celery Worker: nvdiffrec
  worker-nvdiffrec:
    build:
      context: ./workers
      dockerfile: Dockerfile.gpu
    environment:
      - REDIS_URL=redis://redis:6379/0
      - MODEL_TYPE=nvdiffrec
      - NVIDIA_VISIBLE_DEVICES=0  # Share GPU or use different ID
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    depends_on:
      - redis
      - minio
    volumes:
      - model-cache:/app/models

  # Object Storage (S3-compatible)
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio-data:/data
    command: server /data --console-address ":9001"

  # Monitoring Dashboard (optional)
  flower:
    image: mher/flower:2.0
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      - redis

volumes:
  redis-data:
  minio-data:
  model-cache:
```

### GPU Passthrough Configuration

**Prerequisites:**
- NVIDIA Container Toolkit installed on host
- Docker configured with NVIDIA runtime

**Key Configuration:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # RTX 3090
          capabilities: [gpu]
```

**Alternative: All GPUs**
```yaml
device_ids: ['all']  # Use all available GPUs
```

## Patterns to Follow

### Pattern 1: Async API with Immediate Job ID Return
**What:** Don't block client waiting for inference completion. Return job ID immediately.

**When:** Any ML inference task that takes >1 second.

**Why:** 3D reconstruction takes 30 seconds to several minutes. Blocking HTTP requests causes timeouts and poor UX.

**Implementation:**
```python
from fastapi import FastAPI, UploadFile
from celery import Celery

app = FastAPI()
celery_app = Celery(broker='redis://redis:6379/0')

@celery_app.task
def run_reconstruction(model_type: str, image_paths: list):
    # Heavy ML inference here
    pass

@app.post("/api/v1/reconstruct")
async def create_reconstruction_job(
    model: str,
    images: list[UploadFile]
):
    # Save images to storage
    image_paths = save_images(images)

    # Queue job
    task = run_reconstruction.delay(model, image_paths)

    # Return immediately
    return {"job_id": task.id, "status": "pending"}

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    result = celery_app.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": result.state,  # PENDING, STARTED, SUCCESS, FAILURE
        "progress": result.info if result.state == "STARTED" else None
    }
```

### Pattern 2: Separate Worker Pools per Model
**What:** Run different models in separate worker processes/containers.

**When:** Models have different resource requirements or you want independent scaling.

**Why:**
- ReconViaGen and nvdiffrec may have different VRAM requirements
- Isolates failures (one model crash doesn't affect the other)
- Enables independent scaling and deployment

**Implementation:**
```bash
# ReconViaGen worker
celery -A tasks worker --queue reconviagen --concurrency 1

# nvdiffrec worker
celery -A tasks worker --queue nvdiffrec --concurrency 1
```

### Pattern 3: Pre-signed URLs for Large File Downloads
**What:** Generate temporary signed URLs instead of streaming large files through API.

**When:** Result files (meshes, textures) are >10MB.

**Why:** Reduces API server load, enables resumable downloads, better for CDN integration.

**Implementation:**
```python
from minio import Minio

minio_client = Minio("minio:9000")

@app.get("/api/v1/jobs/{job_id}/results")
async def get_results(job_id: str):
    # Generate pre-signed URLs valid for 1 hour
    mesh_url = minio_client.presigned_get_object(
        "results", f"{job_id}/mesh.obj", expires=3600
    )
    texture_url = minio_client.presigned_get_object(
        "results", f"{job_id}/texture.png", expires=3600
    )

    return {
        "mesh_url": mesh_url,
        "texture_url": texture_url,
        "preview_url": f"/api/v1/jobs/{job_id}/preview"  # Small, stream directly
    }
```

### Pattern 4: Model Weight Caching with Docker Volumes
**What:** Persist model weights across container restarts using named volumes.

**When:** Models are large (>1GB) and slow to download.

**Why:**
- Avoids re-downloading 10GB+ model weights on every container start
- Reduces startup time from minutes to seconds
- Saves bandwidth

**Implementation:**
```python
import os
from pathlib import Path

MODEL_CACHE_DIR = Path("/app/models")

def load_model(model_name: str):
    model_path = MODEL_CACHE_DIR / model_name

    if not model_path.exists():
        print(f"Downloading {model_name}...")
        download_model(model_name, model_path)

    print(f"Loading {model_name} from cache...")
    return torch.load(model_path)
```

### Pattern 5: Health Checks for GPU Availability
**What:** Implement health check endpoints that verify GPU access.

**When:** Always, for production systems.

**Why:** Detects GPU passthrough failures, driver issues, out-of-memory conditions.

**Implementation:**
```python
import torch

@app.get("/health")
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    if not gpu_available:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "GPU not available"}
        )

    return {
        "status": "healthy",
        "gpu_count": gpu_count,
        "gpu_name": torch.cuda.get_device_name(0),
        "memory_allocated": torch.cuda.memory_allocated(0) / 1e9,  # GB
        "memory_reserved": torch.cuda.memory_reserved(0) / 1e9
    }
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Running Inference in API Server Process
**What:** Executing ML inference directly in FastAPI request handler.

**Why bad:**
- Blocks event loop, causing all requests to wait
- No isolation if model crashes
- Cannot scale API and workers independently
- GPU memory stays allocated even when idle

**Instead:** Use Celery workers as shown in Pattern 1.

**Consequences:**
- API timeouts after 30 seconds
- One inference request blocks all other requests
- Cannot handle concurrent uploads
- System becomes unusable under load

### Anti-Pattern 2: Using FastAPI BackgroundTasks for Long-Running Jobs
**What:** Using `background_tasks.add_task()` for 3D reconstruction.

**Why bad:**
- Tasks run in same process as API server
- No persistence if server crashes
- No status tracking or retry mechanism
- Limited to single server (doesn't scale horizontally)

**Instead:** Use Celery with Redis for job queue persistence and distributed processing.

**When FastAPI BackgroundTasks are OK:**
- Sending notification emails (< 1 second)
- Logging events
- Cleanup tasks
- Anything that completes in < 5 seconds

### Anti-Pattern 3: Storing Results in Redis
**What:** Saving entire mesh files or textures in Redis as job results.

**Why bad:**
- Redis is in-memory, limited by RAM
- Mesh files can be 50-500MB each
- Redis is designed for small values (<1MB)
- Results evicted when memory full

**Instead:** Store results in MinIO or volume mounts, store only metadata in Redis.

**Correct approach:**
```python
# Bad: Store file in Redis
result = {
    "mesh_data": open("mesh.obj").read()  # 200MB string
}

# Good: Store file in MinIO, path in Redis
minio_client.fput_object("results", f"{job_id}/mesh.obj", "mesh.obj")
result = {
    "mesh_path": f"results/{job_id}/mesh.obj",
    "mesh_size_mb": 200
}
```

### Anti-Pattern 4: Sharing GPU Between All Workers Without Limits
**What:** Running 4+ Celery workers all trying to use the same GPU simultaneously.

**Why bad:**
- GPU has 24GB VRAM (RTX 3090)
- Each model might use 8-12GB VRAM
- 3+ simultaneous jobs cause OOM errors
- All jobs fail, system thrashes

**Instead:** Limit worker concurrency based on VRAM requirements.

**Correct configuration:**
```yaml
# Option 1: One worker per model (safest)
worker-reconviagen:
  environment:
    - CELERY_CONCURRENCY=1  # Process one job at a time

# Option 2: Calculate based on VRAM
# If model uses 8GB, 24GB GPU supports 2 concurrent jobs
worker-reconviagen:
  environment:
    - CELERY_CONCURRENCY=2
```

### Anti-Pattern 5: No Job Timeout Configuration
**What:** Allowing reconstruction jobs to run indefinitely.

**Why bad:**
- Stuck jobs (infinite loops, deadlocks) block workers forever
- GPU memory never released
- Worker capacity exhausted
- No automatic recovery

**Instead:** Configure timeouts at multiple levels.

**Implementation:**
```python
@celery_app.task(
    time_limit=600,  # Hard limit: kill task after 10 minutes
    soft_time_limit=570  # Soft limit: raise exception at 9.5 minutes
)
def run_reconstruction(model_type, image_paths):
    try:
        # Normal processing
        result = model.reconstruct(image_paths)
        return result
    except SoftTimeLimitExceeded:
        # Cleanup before hard kill
        cleanup_temp_files()
        raise
```

### Anti-Pattern 6: Treating Inference Runtime as Implementation Detail
**What:** Not considering inference engine choice upfront, using generic Python code.

**Why bad:**
- Creates technical debt expensive to unwind later
- Misses optimization opportunities (TensorRT, ONNX Runtime)
- Poor performance compared to optimized serving
- Difficult to swap models or frameworks

**Instead:** Treat inference runtime as strategic architectural decision from day 1.

**Consider:**
- Will you need TensorRT optimization later?
- Could ONNX Runtime give better performance?
- Is model format standardized (ONNX, SavedModel)?
- Plan for future runtime migration

## Scalability Considerations

### Single Server (MVP)

**Capacity:** RTX 3090 with 24GB VRAM
- ReconViaGen: ~1 concurrent job (8-12GB VRAM)
- nvdiffrec: ~1 concurrent job (8-12GB VRAM)
- Total throughput: ~6-12 jobs/hour depending on model

**Configuration:**
- 1 API server container
- 1 Redis container
- 1 worker container per model (2 total)
- 1 MinIO container or shared volume

**Scaling bottleneck:** GPU compute capacity

**When to scale:** Queue length >10 jobs and growing

### Multi-Server (<100 concurrent users)

**Add:** More GPU servers

**Architecture changes:**
- MinIO in distributed mode (4+ nodes) or S3
- Redis Cluster or managed Redis (AWS ElastiCache, Redis Cloud)
- Multiple worker servers, each with GPU
- Load balancer in front of API servers

**Configuration:**
```yaml
# API: Scale horizontally (stateless)
api:
  deploy:
    replicas: 3

# Workers: One per GPU server
# Server 1: RTX 3090
worker-server1-reconviagen:
  deploy:
    placement:
      constraints: [node.hostname == gpu-server-1]

# Server 2: RTX 3090
worker-server2-reconviagen:
  deploy:
    placement:
      constraints: [node.hostname == gpu-server-2]
```

**Capacity:** 2 GPU servers = 12-24 jobs/hour

### High Scale (>1000 concurrent users)

**Architecture changes:**
- Kubernetes with GPU node pools
- Managed object storage (S3, GCS)
- Managed Redis (Redis Enterprise, AWS ElastiCache)
- Auto-scaling worker pools
- CDN for result delivery
- Metrics and observability (Prometheus, Grafana)

**GPU Optimization:**
- Multi-instance GPU (MIG) for smaller models
- Model optimization (TensorRT, quantization)
- Batch inference where possible

**Cost optimization:**
- Spot instances for workers (batch jobs tolerate interruption)
- Autoscale workers to zero when queue empty
- Reserved instances for API servers

## Build Order Dependencies

### Phase 1: Local Development Stack (No GPU)
**Goal:** Validate architecture with mock inference

**Components:**
1. FastAPI server with upload endpoints
2. Redis container
3. Celery worker with mock model (returns fake results)
4. Docker volume for storage (defer MinIO)

**Validation:** Can submit job, get job ID, poll status, retrieve mock results

**Benefit:** Full workflow without GPU dependency

### Phase 2: GPU Integration
**Goal:** Add real ML inference

**Prerequisites:** Phase 1 complete

**Add:**
1. NVIDIA Container Toolkit on host
2. GPU-enabled worker Dockerfile
3. Model loading code
4. Real inference execution

**Validation:** Submit job, worker uses GPU, returns real mesh

**Dependency:** Requires Phase 1 (queue, API, storage all working)

### Phase 3: Production Storage
**Goal:** Replace volumes with S3-compatible storage

**Prerequisites:** Phase 2 complete

**Add:**
1. MinIO container (or external S3)
2. Pre-signed URL generation
3. Migration from volume paths to object keys

**Validation:** Results stored in MinIO, accessible via signed URLs

**Why after Phase 2:** Storage is swappable; GPU integration is harder

### Phase 4: Monitoring & Resilience
**Goal:** Production-ready reliability

**Prerequisites:** Phase 3 complete

**Add:**
1. Flower monitoring dashboard
2. Health check endpoints
3. Job timeout configuration
4. Error handling and retry logic
5. Logging and metrics

**Validation:** Can observe worker state, jobs auto-retry on failure

### Phase 5: Multi-Model Support
**Goal:** Support both ReconViaGen and nvdiffrec

**Prerequisites:** Phase 4 complete (one model working reliably)

**Add:**
1. Second worker pool
2. Model selection in API
3. Queue routing by model type

**Validation:** Can submit jobs to either model, tracked separately

**Why last:** Easier to debug one model, then clone pattern

## Component Build Priority

```
API Server ──> Redis ──> Celery Worker (mock) ──> Storage (volume)
    │                          │                       │
    └──> [Test full flow] <────┴───────────────────────┘
                                │
                                ▼
                    Replace mock with GPU inference
                                │
                                ▼
                    Replace volume with MinIO
                                │
                                ▼
                    Add monitoring & resilience
                                │
                                ▼
                    Add second model worker
```

**Critical path:** API → Redis → Worker → Storage
**Parallelizable:** Monitoring, second model (after first works)
**Defer until production:** Kubernetes, autoscaling, CDN

## Sources

### Architecture Patterns
- [FerrariDG/async-ml-inference](https://github.com/FerrariDG/async-ml-inference) - Reference async ML inference architecture
- [Mercari ML System Design Pattern: Asynchronous Pattern](https://mercari.github.io/ml-system-design-pattern/Serving-patterns/Asynchronous-pattern/design_en.html) - Async serving patterns
- [ML Inference Runtimes in 2026](https://medium.com/@digvijay17july/ml-inference-runtimes-in-2026-an-architects-guide-to-choosing-the-right-engine-d3989a87d052) - Strategic runtime decisions
- [Scalable Cloud-Native Pipeline for 3D Reconstruction](https://arxiv.org/html/2409.19322v1) - Microservices architecture for 3D reconstruction

### GPU & Docker
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) - Official installation guide
- [Docker Compose GPU Support](https://docs.docker.com/compose/how-tos/gpu-support/) - Official GPU configuration
- [Docker Model Runner Vulkan GPU Support](https://www.docker.com/blog/docker-model-runner-vulkan-gpu-support/) - 2026 GPU developments

### Storage
- [MinIO for AI](https://www.min.io/solutions/object-storage-for-ai) - Object storage for ML workloads
- [MinIO AIStor](https://www.min.io/product/aistor) - Exascale storage for AI
- [Google Cloud: Storage for AI/ML Workloads](https://docs.cloud.google.com/architecture/ai-ml/storage-for-ai-ml) - Storage architecture patterns

### Queue & Task Processing
- [FastAPI Background Tasks vs Celery](https://medium.com/@ajaygohil2563/fastapi-background-tasks-internal-architecture-and-comparison-with-celery-5c5897f65725) - When to use each
- [Celery vs RQ Comparison 2025](https://generalistprogrammer.com/comparisons/celery-vs-rq) - Queue system comparison
- [Amazon SageMaker Asynchronous Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html) - Async inference patterns

### Monitoring
- [Flower Documentation](https://flower.readthedocs.io/) - Celery monitoring
- [Celery Monitoring Guide](https://docs.celeryq.dev/en/stable/userguide/monitoring.html) - Official monitoring docs

### Anti-Patterns
- [Using AntiPatterns to avoid MLOps Mistakes](https://arxiv.org/abs/2107.00079) - MLOps anti-patterns
- [Machine Learning in Production: Anti-patterns](https://ahsanijaz.github.io/2019-02-10-patterns/) - Production ML mistakes
