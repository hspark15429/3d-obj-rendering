# API Reference

Complete endpoint reference for the 3D reconstruction API. All endpoints return structured JSON responses with consistent error handling.

## Overview

**Base URL:** `http://localhost:8000`

**API Version:** 0.1.0

**Content Types:**
- Requests: `multipart/form-data` (file uploads), `application/json` (other endpoints)
- Responses: `application/json` (except `/download` which returns `application/zip`)

## Authentication

This is a demo API with **no authentication**. All endpoints are publicly accessible.

For production deployment, implement authentication via:
- JWT tokens in `Authorization: Bearer <token>` header
- API keys via query parameter or header
- OAuth 2.0 for third-party integrations

## Endpoints

### GET /health

Health check endpoint with GPU information.

**Description:** Returns service health status, GPU name, driver version, and memory usage. Use this to verify the API is running and GPU is available before submitting jobs.

**Request:** None

**Response:**

```json
{
  "status": "healthy",
  "gpu": {
    "name": "NVIDIA GeForce RTX 4090",
    "driver_version": "535.129.03",
    "memory_total_gb": 24.0,
    "memory_free_gb": 18.5,
    "memory_used_gb": 5.5
  }
}
```

**Status Codes:**
- `200 OK` - Service healthy
- `503 Service Unavailable` - GPU not initialized or unavailable

**Error Response (503):**

```json
{
  "status": "unhealthy",
  "error": "GPU not initialized"
}
```

**Curl Example:**

```bash
curl -X GET http://localhost:8000/health
```

---

### POST /jobs

Submit a new 3D reconstruction job.

**Description:** Upload 6 multi-view images and 6 depth renders to create a reconstruction job. The API validates files, generates a job ID, and queues the task for processing. Returns immediately with job ID for status polling.

**Request (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| views[] | File | Yes | 6 multi-view PNG images (max 20MB each) |
| depth_renders[] | File | Yes | 6 depth render PNG images (max 20MB each) |
| model_type | String | No | Model to use: `reconviagen`, `nvdiffrec`, or `both` (default: `reconviagen`) |

**Response:**

```json
{
  "job_id": "k3m8xq2p",
  "status": "queued",
  "model_type": "reconviagen",
  "created_at": "2026-01-31T12:30:00Z"
}
```

**Status Codes:**
- `200 OK` - Job submitted successfully
- `422 Unprocessable Entity` - Validation failed (see error codes below)

**Error Response (422):**

```json
{
  "error": {
    "code": "INVALID_FILE_COUNT",
    "message": "Expected 6 view images, got 5",
    "details": {
      "field": "views"
    },
    "suggestion": "Upload exactly 6 view images and 6 depth render images."
  }
}
```

**Curl Example:**

```bash
curl -X POST http://localhost:8000/jobs \
  -F "views=@view1.png" \
  -F "views=@view2.png" \
  -F "views=@view3.png" \
  -F "views=@view4.png" \
  -F "views=@view5.png" \
  -F "views=@view6.png" \
  -F "depth_renders=@depth1.png" \
  -F "depth_renders=@depth2.png" \
  -F "depth_renders=@depth3.png" \
  -F "depth_renders=@depth4.png" \
  -F "depth_renders=@depth5.png" \
  -F "depth_renders=@depth6.png" \
  -F "model_type=reconviagen"
```

---

### GET /jobs/{job_id}

Get the current status of a reconstruction job.

**Description:** Poll this endpoint to track job progress. Returns current status, progress percentage, and which model is currently running. Updates in real-time as the worker processes the task.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | String | Job identifier (returned from POST /jobs) |

**Response:**

```json
{
  "job_id": "k3m8xq2p",
  "status": "processing",
  "progress": 45,
  "current_model": "reconviagen",
  "created_at": "2026-01-31T12:30:00Z",
  "updated_at": "2026-01-31T12:32:15Z",
  "error": null
}
```

**Status Values:**
- `queued` - Job waiting in queue
- `processing` - Worker is running the task
- `completed` - Job finished successfully
- `failed` - Job failed (see `error` field)
- `cancelled` - Job was cancelled by user

**Status Codes:**
- `200 OK` - Status retrieved successfully

**Curl Example:**

```bash
curl -X GET http://localhost:8000/jobs/k3m8xq2p
```

**Polling Recommended Interval:**
- Queue/early processing: 5-10 seconds
- Mid-processing: 15-30 seconds
- Near completion: 5 seconds

---

### POST /jobs/{job_id}/cancel

Cancel a running or queued job (two-step confirmation).

**Description:** Two-step cancellation prevents accidental abort of long-running jobs. Step 1 requests cancellation (returns `cancel_requested`). Step 2 confirms cancellation with `confirm=true` (actually cancels the job).

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | String | Job identifier |

**Request Body (optional):**

```json
{
  "confirm": false
}
```

**Step 1 - Request Cancellation:**

Request:
```bash
curl -X POST http://localhost:8000/jobs/k3m8xq2p/cancel
```

Response (200):
```json
{
  "job_id": "k3m8xq2p",
  "status": "cancel_requested",
  "message": "Cancellation requested. Send confirm=true to confirm cancellation."
}
```

**Step 2 - Confirm Cancellation:**

Request:
```bash
curl -X POST http://localhost:8000/jobs/k3m8xq2p/cancel \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'
```

Response (200):
```json
{
  "job_id": "k3m8xq2p",
  "status": "cancelled",
  "message": "Job cancellation confirmed. Task will abort at next checkpoint."
}
```

**Status Codes:**
- `200 OK` - Cancellation requested or confirmed
- `400 Bad Request` - Cannot cancel job in current state (already completed/failed)

**Error Response (400):**

```json
{
  "error": {
    "code": "JOB_NOT_READY",
    "message": "Cannot cancel job in state 'SUCCESS'",
    "details": {
      "job_id": "k3m8xq2p",
      "current_state": "SUCCESS"
    },
    "suggestion": "Wait for job to complete before downloading. Poll GET /jobs/{job_id} for status."
  }
}
```

**Notes:**
- Cancellation request expires after 1 hour if not confirmed
- Worker checks for cancellation before each major step (load weights, inference, quality metrics)
- Cancelled jobs have their files cleaned up automatically

---

### GET /jobs/{job_id}/download

Download all job results as a single ZIP file.

**Description:** Download completed reconstruction outputs including mesh files (OBJ, PLY, GLB), textures, preview images, and quality metrics report. ZIP is created in-memory with compression level 6.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | String | Job identifier |

**Response Headers:**
- `Content-Type: application/zip`
- `Content-Disposition: attachment; filename={job_id}.zip`

**ZIP Contents:**

```
{job_id}.zip
├── {model_name}/
│   ├── mesh.glb          # GLB mesh with baked textures
│   ├── mesh.obj          # OBJ mesh
│   ├── mesh.ply          # PLY mesh
│   ├── texture.png       # Texture map (if separate)
│   ├── preview_textured_0.png
│   ├── preview_textured_1.png
│   ├── preview_wireframe_0.png
│   ├── preview_wireframe_1.png
│   └── quality.json      # PSNR/SSIM metrics
└── README.txt            # (optional)
```

**Status Codes:**
- `200 OK` - ZIP download started
- `404 Not Found` - Job never existed
- `409 Conflict` - Job still processing (not ready)
- `410 Gone` - Job results expired/cleaned up
- `500 Internal Server Error` - Job failed or outputs incomplete

**Error Response (404):**

```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "Job 'k3m8xq2p' not found",
    "details": {
      "job_id": "k3m8xq2p"
    },
    "suggestion": "Check the job ID and try again."
  }
}
```

**Error Response (409):**

```json
{
  "error": {
    "code": "JOB_NOT_READY",
    "message": "Job 'k3m8xq2p' is still processing (45% complete)",
    "details": {
      "job_id": "k3m8xq2p",
      "progress": 45,
      "state": "PROGRESS"
    },
    "suggestion": "Wait for job to complete before downloading. Poll GET /jobs/{job_id} for status."
  }
}
```

**Error Response (410):**

```json
{
  "error": {
    "code": "JOB_EXPIRED",
    "message": "Job 'k3m8xq2p' results have expired or been cleaned up",
    "details": {
      "job_id": "k3m8xq2p"
    },
    "suggestion": "Job results are no longer available. Submit a new job if needed."
  }
}
```

**Error Response (500):**

```json
{
  "error": {
    "code": "INCOMPLETE_RESULTS",
    "message": "Job 'k3m8xq2p' has incomplete results",
    "details": {
      "job_id": "k3m8xq2p",
      "missing": ["mesh.obj", "quality.json"]
    },
    "suggestion": "Job outputs are incomplete. Contact support."
  }
}
```

**Curl Example:**

```bash
curl -X GET http://localhost:8000/jobs/k3m8xq2p/download -o results.zip
```

---

## Error Codes

All errors follow a consistent JSON structure with actionable suggestions:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": {},
    "suggestion": "Actionable fix"
  }
}
```

### Complete Error Code Table

| Code | HTTP Status | Meaning | Suggestion |
|------|-------------|---------|------------|
| **VALIDATION_FAILED** | 422 | Request validation failed (generic) | Fix the validation errors and try again. |
| **FILE_TOO_LARGE** | 422 | Uploaded file exceeds size limit | Reduce image resolution or compress PNGs before uploading (max 20MB per file). |
| **INVALID_FILE_FORMAT** | 422 | File is not a valid PNG image | Ensure all uploaded files are valid PNG images. |
| **INVALID_FILE_COUNT** | 422 | Wrong number of files uploaded | Upload exactly 6 view images and 6 depth render images. |
| **JOB_NOT_FOUND** | 404 | Job ID does not exist | Check the job ID and try again. |
| **JOB_NOT_READY** | 409 | Job not in correct state for operation | Wait for job to complete before downloading. Poll GET /jobs/{job_id} for status. |
| **JOB_EXPIRED** | 410 | Job results no longer available | Job results are no longer available. Submit a new job if needed. |
| **MODEL_FAILED** | 500 | Model processing failed (generic) | Model processing failed. Try again with different inputs or contact support. |
| **MODEL_OOM** | 500 | System ran out of memory | System ran out of memory. Try again when resources are available. |
| **MODEL_VRAM_OOM** | 500 | GPU ran out of VRAM | GPU ran out of VRAM (16GB limit). Try again when GPU resources are available. |
| **MODEL_CONVERGENCE_FAILED** | 500 | Model failed to converge | Model failed to converge. Try uploading higher-quality or better-lit input images. |
| **QUALITY_THRESHOLD_FAILED** | 500 | Reconstruction quality below threshold | Reconstruction quality below threshold. Try uploading higher-quality input images. |
| **INCOMPLETE_RESULTS** | 500 | Job outputs are missing files | Job outputs are incomplete. Contact support. |
| **GPU_UNAVAILABLE** | 503 | No GPU available for processing | No GPU available. Job will retry automatically. |
| **DISK_FULL** | 503 | Server disk is full | Server disk is full. Contact support. |
| **MEMORY_EXHAUSTED** | 503 | Server out of memory | Server out of memory. Try again later. |
| **UNKNOWN_ERROR** | 500 | Unexpected error occurred | An unexpected error occurred. Contact support if this persists. |

### Error Code Details

#### Validation Errors (422)

**VALIDATION_FAILED**
- General validation failure for malformed requests
- Check `details.fields` for specific field errors
- Example: Missing required parameters, invalid enum values

**FILE_TOO_LARGE**
- Single file exceeds 20MB limit
- Total upload exceeds 200MB limit
- Compress images or reduce resolution

**INVALID_FILE_FORMAT**
- File extension is not `.png`
- File magic bytes don't match PNG signature
- Corrupted or truncated PNG file

**INVALID_FILE_COUNT**
- Expected 6 views, got different count
- Expected 6 depth renders, got different count
- Check `details.field` to see which array is incorrect

#### Not Found Errors (404)

**JOB_NOT_FOUND**
- Job ID never existed (typo in ID)
- Job expired from Celery (> 24 hours old)
- Job was never submitted successfully

#### State Conflict Errors (409)

**JOB_NOT_READY**
- Attempting to download while job is still processing
- Attempting to cancel already completed job
- Check current status via `GET /jobs/{job_id}`

#### Gone Errors (410)

**JOB_EXPIRED**
- Job completed but results were cleaned up
- Output directory deleted or missing
- Re-submit job to regenerate outputs

#### Model/Processing Errors (500)

**MODEL_FAILED**
- Generic model execution failure
- Check `details.model` for which model failed
- Check `details.pipeline_stage` for which stage failed

**MODEL_OOM**
- System RAM exhausted during processing
- Typically occurs during mesh processing or quality metrics
- Wait for other jobs to complete

**MODEL_VRAM_OOM**
- GPU VRAM exhausted during inference
- ReconViaGen needs ~14-16GB VRAM
- Wait for other jobs to complete or use smaller model

**MODEL_CONVERGENCE_FAILED**
- Model optimization didn't converge
- Input images may be low quality or inconsistent lighting
- Try with higher quality inputs

**QUALITY_THRESHOLD_FAILED**
- PSNR < 20dB or SSIM < 0.75
- Reconstruction doesn't match input views well enough
- Check `details.psnr` and `details.ssim` for actual values
- Upload higher quality or better-lit images

**INCOMPLETE_RESULTS**
- Expected output files missing
- Indicates partial model failure or disk issue
- Check `details.missing` for which files are missing

#### Resource Errors (503)

**GPU_UNAVAILABLE**
- GPU driver error or GPU in use
- Service will retry automatically
- Transient error, usually resolves

**DISK_FULL**
- Job storage volume full
- Cannot save input files or outputs
- Contact administrator to free disk space

**MEMORY_EXHAUSTED**
- System out of memory for queue/API operations
- Wait for other jobs to complete
- May indicate memory leak if persistent

**UNKNOWN_ERROR**
- Unhandled exception in API or worker
- Internal server error
- Check server logs for details

## Response Format

### Success Response Structure

All successful responses return JSON with relevant fields:

```json
{
  "field1": "value1",
  "field2": "value2"
}
```

### Error Response Structure

All error responses include structured error object:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {
      "additional": "context"
    },
    "suggestion": "Actionable fix"
  }
}
```

**Fields:**
- `code`: Error code from table above (use for client-side handling)
- `message`: Human-readable error description
- `details`: Additional context (job_id, missing fields, metrics, etc.)
- `suggestion`: Actionable guidance for fixing the error

## Rate Limiting

This demo API has **no rate limiting**. For production deployment, implement:

- Per-IP rate limits (e.g., 10 requests/minute for job submission)
- Concurrent job limits per client
- File size/bandwidth limits
- Queue depth limits to prevent resource exhaustion

Implement rate limiting at reverse proxy (nginx, traefik) or application level (slowapi, flask-limiter).

## Versioning

Current version: **0.1.0**

API versioning not implemented yet. For future versions:
- Include version in path: `/v1/jobs`, `/v2/jobs`
- Or use `Accept: application/vnd.api+json; version=1` header
- Maintain backward compatibility for at least one major version

## Client Examples

### Python Client

```python
import requests
import time
from pathlib import Path

def submit_and_download(views_dir, depth_dir, model_type="reconviagen"):
    # Prepare files
    files = []
    for i in range(1, 7):
        files.append(("views", open(views_dir / f"view{i}.png", "rb")))
        files.append(("depth_renders", open(depth_dir / f"depth{i}.png", "rb")))

    data = {"model_type": model_type}

    # Submit job
    response = requests.post("http://localhost:8000/jobs", files=files, data=data)
    response.raise_for_status()
    job_id = response.json()["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll status
    while True:
        status_response = requests.get(f"http://localhost:8000/jobs/{job_id}")
        status_response.raise_for_status()
        status_data = status_response.json()

        if status_data["status"] == "completed":
            print(f"Job completed!")
            break
        elif status_data["status"] == "failed":
            print(f"Job failed: {status_data.get('error')}")
            return None

        progress = status_data.get("progress", 0)
        print(f"Progress: {progress}%")
        time.sleep(10)

    # Download results
    download_response = requests.get(f"http://localhost:8000/jobs/{job_id}/download")
    download_response.raise_for_status()

    with open(f"{job_id}.zip", "wb") as f:
        f.write(download_response.content)

    print(f"Results saved to {job_id}.zip")
    return job_id

# Usage
submit_and_download(Path("input/views"), Path("input/depth"))
```

### JavaScript/TypeScript Client

```typescript
async function submitAndDownload(viewFiles: File[], depthFiles: File[], modelType = 'reconviagen') {
  const formData = new FormData();

  viewFiles.forEach(file => formData.append('views', file));
  depthFiles.forEach(file => formData.append('depth_renders', file));
  formData.append('model_type', modelType);

  // Submit job
  const submitResponse = await fetch('http://localhost:8000/jobs', {
    method: 'POST',
    body: formData
  });

  if (!submitResponse.ok) {
    throw new Error(`Submit failed: ${submitResponse.statusText}`);
  }

  const { job_id } = await submitResponse.json();
  console.log(`Job submitted: ${job_id}`);

  // Poll status
  while (true) {
    const statusResponse = await fetch(`http://localhost:8000/jobs/${job_id}`);
    const statusData = await statusResponse.json();

    if (statusData.status === 'completed') {
      console.log('Job completed!');
      break;
    } else if (statusData.status === 'failed') {
      throw new Error(`Job failed: ${statusData.error}`);
    }

    console.log(`Progress: ${statusData.progress}%`);
    await new Promise(resolve => setTimeout(resolve, 10000));
  }

  // Download results
  const downloadResponse = await fetch(`http://localhost:8000/jobs/${job_id}/download`);
  const blob = await downloadResponse.blob();

  // Trigger browser download
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${job_id}.zip`;
  a.click();

  return job_id;
}
```

### cURL Workflow

```bash
#!/bin/bash
# Complete workflow using curl

# 1. Check health
curl -X GET http://localhost:8000/health | jq .

# 2. Submit job
JOB_ID=$(curl -X POST http://localhost:8000/jobs \
  -F "views=@view1.png" \
  -F "views=@view2.png" \
  -F "views=@view3.png" \
  -F "views=@view4.png" \
  -F "views=@view5.png" \
  -F "views=@view6.png" \
  -F "depth_renders=@depth1.png" \
  -F "depth_renders=@depth2.png" \
  -F "depth_renders=@depth3.png" \
  -F "depth_renders=@depth4.png" \
  -F "depth_renders=@depth5.png" \
  -F "depth_renders=@depth6.png" \
  -F "model_type=reconviagen" | jq -r .job_id)

echo "Job ID: $JOB_ID"

# 3. Poll status
while true; do
  STATUS=$(curl -s http://localhost:8000/jobs/$JOB_ID | jq -r .status)
  PROGRESS=$(curl -s http://localhost:8000/jobs/$JOB_ID | jq -r .progress)

  echo "Status: $STATUS, Progress: $PROGRESS%"

  if [ "$STATUS" == "completed" ]; then
    echo "Job completed!"
    break
  elif [ "$STATUS" == "failed" ]; then
    echo "Job failed!"
    exit 1
  fi

  sleep 10
done

# 4. Download results
curl -X GET http://localhost:8000/jobs/$JOB_ID/download -o results.zip
echo "Results downloaded to results.zip"
```

## Support

For issues or questions:
- Check error code and suggestion in response
- Review logs: `docker compose logs -f api worker`
- Verify GPU is healthy: `GET /health`
- Ensure input files meet requirements (6 views, 6 depths, PNG format, < 20MB each)

Common issues:
- **GPU unavailable:** Check `nvidia-smi` and ensure Docker has GPU access
- **Job stuck in queued:** Worker may be down, check `docker compose ps`
- **Download 410 Gone:** Results expired, resubmit job
- **Quality threshold failed:** Input images low quality or inconsistent lighting
