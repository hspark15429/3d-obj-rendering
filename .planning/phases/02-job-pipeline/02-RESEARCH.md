# Phase 2: Job Pipeline - Research

**Researched:** 2026-01-31
**Domain:** Async job queue with FastAPI, Celery, Redis
**Confidence:** HIGH

## Summary

This phase implements an async job queue for long-running 3D reconstruction inference tasks. The locked decisions specify Celery + Redis for the queue, single multipart POST for file uploads (6 views + depth renders as PNG), polling-only status checks, and two-step cancellation.

The research confirms that FastAPI + Celery + Redis is a well-documented, production-proven stack. Key patterns include: using `shared_task` decorator to avoid circular imports, `bind=True` with `update_state()` for progress tracking, and structured file validation with magic bytes. For cancellation, the cooperative AbortableTask pattern (checking `is_aborted()` periodically) is more reliable than `revoke(terminate=True)` for graceful shutdown.

**Primary recommendation:** Use Celery's `shared_task` decorator with `bind=True` for progress-reporting tasks, store uploaded files in a job-specific directory, and implement cooperative cancellation via a Redis flag that tasks check periodically.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| celery | 5.6.x | Distributed task queue | De facto Python standard for async jobs |
| redis | 5.x (server) | Message broker + result backend | Simpler than RabbitMQ, fast, sufficient for this scale |
| redis (py) | 5.x | Python Redis client | Required by Celery for Redis transport |
| python-multipart | 0.0.9+ | Multipart form parsing | Required by FastAPI for file uploads |
| aiofiles | 24.x | Async file I/O | Non-blocking file writes for uploads |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| nanoid | 2.0.0 | Short unique ID generation | Job IDs (8 chars, URL-safe, human-friendly) |
| filetype | 1.2.x | Magic byte file validation | Verify PNG files by content, not extension |
| flower | 2.x | Celery monitoring dashboard | Development/debugging (optional in prod) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Redis broker | RabbitMQ | RabbitMQ has better reliability guarantees but more complex setup; Redis sufficient for single-node |
| nanoid | shortuuid | Both work; nanoid slightly higher entropy per character |
| filetype | python-magic | python-magic requires libmagic; filetype is pure Python |

**Installation:**
```bash
pip install celery[redis] redis aiofiles python-multipart nanoid filetype
```

## Architecture Patterns

### Recommended Project Structure
```
app/
├── main.py              # FastAPI app, routes
├── celery_app.py        # Celery app instance creation
├── config.py            # Settings (broker URL, etc.)
├── tasks/
│   ├── __init__.py
│   └── reconstruction.py # Celery tasks with shared_task
├── api/
│   ├── __init__.py
│   ├── jobs.py          # Job submission, status, cancel endpoints
│   └── schemas.py       # Pydantic models for request/response
├── services/
│   ├── __init__.py
│   ├── file_handler.py  # File validation, storage
│   └── job_manager.py   # Job state management
└── storage/             # Uploaded files directory (volume-mounted)
    └── jobs/
        └── {job_id}/    # Per-job directory
```

### Pattern 1: Celery App Factory with shared_task
**What:** Separate Celery instance creation to avoid circular imports
**When to use:** Always, when using FastAPI + Celery together
**Example:**
```python
# app/celery_app.py
from celery import Celery

def create_celery_app():
    celery = Celery(
        "worker",
        broker="redis://redis:6379/0",
        backend="redis://redis:6379/0",
    )
    celery.conf.update(
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        result_expires=3600,  # 1 hour
        worker_prefetch_multiplier=1,  # For priority/cancellation
    )
    return celery

celery_app = create_celery_app()

# app/tasks/reconstruction.py
from celery import shared_task

@shared_task(bind=True)
def process_reconstruction(self, job_id: str):
    # bind=True gives access to self for update_state
    ...
```
**Source:** [TestDriven.io - Application Factory](https://testdriven.io/courses/fastapi-celery/app-factory/)

### Pattern 2: Progress Tracking with update_state
**What:** Report task progress percentage to result backend
**When to use:** Long-running tasks where client needs progress updates
**Example:**
```python
# Source: Celery official docs
@shared_task(bind=True)
def process_reconstruction(self, job_id: str):
    total_steps = 5
    for step in range(total_steps):
        # Check cancellation before each step
        if is_job_cancelled(job_id):
            cleanup_job_files(job_id)
            return {"status": "cancelled"}

        # Do work...

        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": int((step + 1) / total_steps * 100),
                "current_step": step + 1,
                "total_steps": total_steps
            }
        )
    return {"status": "completed", "result_path": f"/jobs/{job_id}/output.obj"}
```
**Source:** [Celery Tasks Documentation](https://docs.celeryq.dev/en/stable/userguide/tasks.html)

### Pattern 3: Cooperative Cancellation via Redis Flag
**What:** Use Redis key to signal cancellation, task checks periodically
**When to use:** When you need reliable, graceful task cancellation
**Example:**
```python
# app/services/job_manager.py
import redis

r = redis.Redis(host="redis", port=6379, db=1)  # Separate DB for app state

def request_cancellation(job_id: str):
    """Mark job for cancellation (first step of two-step cancel)."""
    r.set(f"cancel_request:{job_id}", "pending", ex=3600)

def confirm_cancellation(job_id: str) -> bool:
    """Confirm cancellation (second step)."""
    if r.get(f"cancel_request:{job_id}"):
        r.set(f"cancel:{job_id}", "1", ex=3600)
        r.delete(f"cancel_request:{job_id}")
        return True
    return False

def is_job_cancelled(job_id: str) -> bool:
    """Check if job should abort (called by worker)."""
    return r.get(f"cancel:{job_id}") is not None

# In task:
@shared_task(bind=True)
def process_reconstruction(self, job_id: str):
    for step in process_steps:
        if is_job_cancelled(job_id):
            cleanup_job_files(job_id)
            self.update_state(state="REVOKED")
            return None
        # ... process step
```
**Source:** Community pattern, more reliable than AbortableTask for Redis backend

### Pattern 4: Multiple File Upload with Validation
**What:** Accept multiple files in single request, validate each
**When to use:** Multi-view image uploads
**Example:**
```python
# Source: FastAPI official docs + filetype library
from fastapi import UploadFile, HTTPException
import filetype

EXPECTED_FILES = 12  # 6 views + 6 depth renders
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'

async def validate_upload(files: list[UploadFile]) -> None:
    if len(files) != EXPECTED_FILES:
        raise HTTPException(400, f"Expected {EXPECTED_FILES} files, got {len(files)}")

    for f in files:
        # Check content type
        if f.content_type != "image/png":
            raise HTTPException(400, f"File {f.filename} must be PNG")

        # Read header for magic byte validation
        header = await f.read(8)
        await f.seek(0)
        if header[:8] != PNG_MAGIC:
            raise HTTPException(400, f"File {f.filename} is not a valid PNG")

        # Check size
        await f.seek(0, 2)  # Seek to end
        size = f.file.tell()
        await f.seek(0)
        if size > MAX_FILE_SIZE:
            raise HTTPException(400, f"File {f.filename} exceeds {MAX_FILE_SIZE} bytes")
```
**Source:** [FastAPI Request Files](https://fastapi.tiangolo.com/tutorial/request-files/)

### Anti-Patterns to Avoid
- **Synchronous file I/O in async endpoints:** Use aiofiles for non-blocking writes
- **Storing full file in memory:** Use streaming/chunked writes for large files
- **Using revoke(terminate=True) programmatically:** Kills worker process, not task
- **Checking content-type header only:** Easily spoofed; validate magic bytes
- **Single Redis instance for everything:** Separate broker, backend, and app state

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Short unique IDs | Custom random string generator | nanoid | Cryptographically secure, collision calculator available |
| File type detection | Extension checking | filetype library | Magic byte detection prevents spoofing |
| Task queue | Threading/multiprocessing | Celery | Retries, monitoring, persistence, distributed |
| Async file writes | sync open/write | aiofiles | Prevents blocking event loop |
| Progress tracking | Custom DB polling | Celery update_state | Built-in, integrates with result backend |

**Key insight:** The async job queue domain has well-tested solutions for every common problem. Hand-rolling any of these introduces subtle bugs around concurrency, file handling, and state management.

## Common Pitfalls

### Pitfall 1: Circular Imports Between FastAPI and Celery
**What goes wrong:** Importing Celery app in routes that also define tasks causes import loops
**Why it happens:** Both modules try to import from each other at load time
**How to avoid:** Use `shared_task` decorator (doesn't require Celery instance at import time), separate celery_app.py from tasks
**Warning signs:** `ImportError` on startup, tasks not registering

### Pitfall 2: Visibility Timeout Too Short for Long Tasks
**What goes wrong:** Task redelivered while still running, causing duplicate execution
**Why it happens:** Default Redis visibility timeout is 1 hour; if task exceeds this, Redis thinks worker died
**How to avoid:** Set `visibility_timeout` in broker_transport_options to exceed max task duration (e.g., 4 hours for inference)
**Warning signs:** Duplicate job results, tasks running twice

### Pitfall 3: File Pointer Position After Reading
**What goes wrong:** File appears empty when saving after validation
**Why it happens:** Reading file for validation moves pointer to end
**How to avoid:** Always `await file.seek(0)` after reading for validation
**Warning signs:** Zero-byte files saved to disk

### Pitfall 4: Blocking Event Loop with Synchronous I/O
**What goes wrong:** API becomes unresponsive during file uploads/saves
**Why it happens:** Using sync file operations in async endpoints blocks the entire event loop
**How to avoid:** Use `aiofiles` for all file I/O in FastAPI endpoints
**Warning signs:** High latency during file operations, timeouts

### Pitfall 5: Task Cancellation Race Conditions
**What goes wrong:** Task continues after "cancelled" status shown to user
**Why it happens:** `revoke()` only prevents queued tasks; running tasks need cooperative check
**How to avoid:** Implement cooperative cancellation with Redis flag, check at safe points
**Warning signs:** Resources consumed after cancellation, partial outputs

### Pitfall 6: Lost Revokes on Worker Restart
**What goes wrong:** Revoked tasks execute after worker restarts
**Why it happens:** Revoke list is in-memory by default
**How to avoid:** Use `--statedb` worker option OR cooperative Redis-based cancellation
**Warning signs:** "Cancelled" jobs completing after worker restart

## Code Examples

Verified patterns from official sources:

### Docker Compose for FastAPI + Celery + Redis
```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./storage:/app/storage
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  worker:
    build: .
    command: celery -A app.celery_app worker --loglevel=info --concurrency=1
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./storage:/app/storage
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
```
**Source:** [TestDriven.io - Dockerizing Celery](https://testdriven.io/courses/fastapi-celery/docker/)

### Job Submission Endpoint
```python
# app/api/jobs.py
from fastapi import APIRouter, UploadFile, HTTPException
from nanoid import generate as nanoid

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.post("/")
async def submit_job(
    views: list[UploadFile],
    depth_renders: list[UploadFile],
    model_type: str = "default"
):
    # Validate file counts
    if len(views) != 6:
        raise HTTPException(400, f"Expected 6 view images, got {len(views)}")
    if len(depth_renders) != 6:
        raise HTTPException(400, f"Expected 6 depth renders, got {len(depth_renders)}")

    # Generate job ID (8 chars, alphanumeric + _-)
    job_id = nanoid(size=8)

    # Validate and save files
    await validate_and_save_files(job_id, views + depth_renders)

    # Queue task
    from app.tasks.reconstruction import process_reconstruction
    process_reconstruction.delay(job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
```

### Job Status Endpoint
```python
@router.get("/{job_id}")
async def get_job_status(job_id: str):
    from app.celery_app import celery_app

    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        return {"job_id": job_id, "status": "queued"}
    elif result.state == "PROGRESS":
        return {
            "job_id": job_id,
            "status": "processing",
            "progress": result.info.get("progress", 0)
        }
    elif result.state == "SUCCESS":
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result.result
        }
    elif result.state in ("FAILURE", "REVOKED"):
        status = "cancelled" if result.state == "REVOKED" else "failed"
        return {"job_id": job_id, "status": status}

    return {"job_id": job_id, "status": result.state.lower()}
```

### Short ID Generation with Nanoid
```python
from nanoid import generate

# 8-character alphanumeric ID (human-friendly)
# ~48 bits of entropy, safe for millions of IDs
job_id = generate(
    alphabet="0123456789abcdefghijklmnopqrstuvwxyz",
    size=8
)
# Example: "k3m8xq2p"
```
**Source:** [nanoid PyPI](https://pypi.org/project/nanoid/)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Celery 4.x with pickle | Celery 5.x with JSON serializer | Celery 5.0 (2020) | Security improvement, JSON-only recommended |
| RabbitMQ default | Redis equally viable | Celery 5.x | Redis simpler for single-node, sufficient for most |
| task decorator | shared_task decorator | Best practice 2022+ | Avoids circular imports |
| revoke() for cancel | Cooperative cancellation | Current recommendation | More reliable, graceful shutdown |
| Extension-based validation | Magic byte validation | Security best practice | Prevents file type spoofing |

**Deprecated/outdated:**
- `@celery.task` decorator when FastAPI integration needed (use `@shared_task`)
- pickle serializer (security risk, use JSON)
- AbortableTask with Redis backend (requires database backend; use Redis flag instead)

## Open Questions

Things that couldn't be fully resolved:

1. **Exact progress update frequency**
   - What we know: Can update via `update_state()` at any point
   - What's unclear: Optimal frequency for 3D reconstruction (every step? every 10%?)
   - Recommendation: Update at major milestones (5-10 updates per job), not continuously

2. **Cancel window scope**
   - What we know: Queued jobs easy to cancel, processing requires cooperative check
   - What's unclear: How frequently the actual inference model can be interrupted
   - Recommendation: Allow cancellation of both queued and processing, with "cancellation requested" state if in-flight

3. **File storage cleanup for failed jobs**
   - What we know: Cancelled jobs delete files immediately per user decision
   - What's unclear: Retention policy for failed jobs (keep for debugging?)
   - Recommendation: Keep failed job files for 24h for debugging, then clean up

## Sources

### Primary (HIGH confidence)
- [FastAPI Request Files](https://fastapi.tiangolo.com/tutorial/request-files/) - File upload patterns
- [Celery Tasks Documentation](https://docs.celeryq.dev/en/stable/userguide/tasks.html) - update_state, bind=True
- [Celery Redis Backend](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html) - Configuration
- [Celery AbortableTask](https://docs.celeryq.dev/en/stable/reference/celery.contrib.abortable.html) - Cooperative cancellation
- [nanoid PyPI](https://pypi.org/project/nanoid/) - ID generation API

### Secondary (MEDIUM confidence)
- [TestDriven.io Celery + FastAPI Course](https://testdriven.io/courses/fastapi-celery/) - Architecture patterns, Docker setup
- [Better Stack FastAPI File Uploads](https://betterstack.com/community/guides/scaling-python/uploading-files-using-fastapi/) - Validation patterns
- [FastAPI + Celery Guide 2026](https://blog.greeden.me/en/2026/01/27/the-complete-guide-to-background-processing-with-fastapi-x-celery-redishow-to-separate-heavy-work-from-your-api-to-keep-services-stable/) - Production patterns

### Tertiary (LOW confidence)
- Community patterns for Redis-based cancellation (custom implementation, not official)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official documentation, well-established ecosystem
- Architecture: HIGH - Multiple authoritative sources agree
- File validation: HIGH - FastAPI + filetype documented patterns
- Cancellation: MEDIUM - Cooperative pattern is community best practice, not official
- Progress tracking: HIGH - Official Celery documentation

**Research date:** 2026-01-31
**Valid until:** 2026-03-01 (30 days - stable ecosystem)
