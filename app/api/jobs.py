"""
Job API router for 3D reconstruction service.

Endpoints:
- POST /jobs - Submit new reconstruction job
- GET /jobs/{job_id} - Get job status
- GET /jobs/{job_id}/download - Download job results as ZIP
- POST /jobs/{job_id}/cancel - Cancel job (two-step)
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import StreamingResponse
from nanoid import generate

from app.api.schemas import (
    JobSubmitResponse,
    JobStatusResponse,
    CancelRequest,
    CancelResponse,
    JobStatus,
    ModelType,
)
from app.services.file_handler import (
    validate_upload_files,
    save_job_files,
    get_job_path,
    FileValidationError,
)
from app.services.result_packager import (
    create_result_zip,
    validate_job_outputs,
    IncompleteResultsError,
)
from app.api.error_codes import ErrorCode, make_error_detail
from app.services.job_manager import (
    request_cancellation,
    confirm_cancellation,
    is_job_cancelled,
    cancel_pending,
)
from app.celery_app import celery_app
from app.tasks.reconstruction import process_reconstruction


router = APIRouter(prefix="/jobs", tags=["jobs"])


def generate_job_id() -> str:
    """
    Generate human-friendly 8-character job ID.

    Uses lowercase alphanumeric characters for easy typing.
    Example: "k3m8xq2p"

    Returns:
        str: 8-character job identifier
    """
    return generate(alphabet="0123456789abcdefghijklmnopqrstuvwxyz", size=8)


@router.post("/", response_model=JobSubmitResponse)
async def submit_job(
    views: list[UploadFile] = File(..., description="6 multi-view PNG images"),
    depth_renders: list[UploadFile] = File(..., description="6 depth render PNG images"),
    model_type: ModelType = Form(ModelType.RECONVIAGEN, description="Model type: reconviagen, nvdiffrec, or both"),
):
    """
    Submit a new 3D reconstruction job.

    Accepts 12 PNG files (6 multi-view images + 6 depth renders) and queues
    a reconstruction task. Returns immediately with job ID for status polling.

    Args:
        views: List of 6 multi-view PNG images
        depth_renders: List of 6 depth render PNG images
        model_type: Reconstruction model to use (reconviagen, nvdiffrec, or both)

    Returns:
        JobSubmitResponse: Job ID, status (queued), model_type, and creation timestamp

    Raises:
        HTTPException 400: Invalid model type or file validation failed
    """
    # Validate uploaded files
    try:
        await validate_upload_files(views, depth_renders)
    except FileValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": e.message, "field": e.field}
        )

    # Generate job ID
    job_id = generate_job_id()

    # Save files to storage
    await save_job_files(job_id, views, depth_renders)

    # Queue reconstruction task with model_type
    # Use model_type.value to pass string to Celery (avoid enum serialization issues)
    process_reconstruction.apply_async(
        args=[job_id, model_type.value],
        task_id=job_id
    )

    # Return response with model_type
    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        model_type=model_type,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of a reconstruction job.

    Queries Celery for task state and maps to job status.
    Shows progress percentage and current model when task is processing.

    Args:
        job_id: Job identifier

    Returns:
        JobStatusResponse: Current job status with progress and current_model

    State mapping:
        - PENDING -> queued
        - STARTED -> processing (progress=0)
        - PROGRESS -> processing (progress from task, current_model from meta)
        - SUCCESS -> completed
        - FAILURE -> failed
        - REVOKED -> cancelled
    """
    # Get task result from Celery
    result = celery_app.AsyncResult(job_id)

    # Map Celery state to JobStatus
    state = result.state
    progress = None
    current_model = None
    error = None
    created_at = datetime.now(timezone.utc)  # Placeholder - should be from DB in production
    updated_at = None

    if state == "PENDING":
        # Check if cancellation is pending
        if cancel_pending(job_id):
            status = JobStatus.QUEUED
        else:
            status = JobStatus.QUEUED

    elif state == "STARTED":
        status = JobStatus.PROCESSING
        progress = 0
        updated_at = datetime.now(timezone.utc)

    elif state == "PROGRESS":
        status = JobStatus.PROCESSING
        if result.info and isinstance(result.info, dict):
            progress = result.info.get("progress", 0)
            current_model = result.info.get("model")
        updated_at = datetime.now(timezone.utc)

    elif state == "SUCCESS":
        status = JobStatus.COMPLETED
        progress = 100
        updated_at = datetime.now(timezone.utc)

    elif state == "FAILURE":
        status = JobStatus.FAILED
        if result.info:
            error = str(result.info)
        updated_at = datetime.now(timezone.utc)

    elif state == "REVOKED":
        status = JobStatus.CANCELLED
        updated_at = datetime.now(timezone.utc)

    else:
        # Unknown state - treat as queued
        status = JobStatus.QUEUED

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        progress=progress,
        current_model=current_model,
        created_at=created_at,
        updated_at=updated_at,
        error=error,
    )


@router.post("/{job_id}/cancel", response_model=CancelResponse)
async def cancel_job(
    job_id: str,
    body: Optional[CancelRequest] = Body(None),
):
    """
    Cancel a reconstruction job (two-step confirmation).

    Step 1: POST without confirm or confirm=false
        - Sets cancel_request flag
        - Returns "cancel_requested" status
        - User must confirm to actually cancel

    Step 2: POST with confirm=true
        - Confirms cancellation
        - Worker will abort at next checkpoint
        - Returns "cancelled" status

    Args:
        job_id: Job identifier
        body: Cancel request with optional confirm flag

    Returns:
        CancelResponse: Cancellation status and message
    """
    # Check job exists and get current state
    result = celery_app.AsyncResult(job_id)
    state = result.state

    # Cannot cancel completed/failed jobs
    if state in ["SUCCESS", "FAILURE"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in state '{state}'"
        )

    # Step 1: Request cancellation
    if body is None or not body.confirm:
        request_cancellation(job_id)
        return CancelResponse(
            job_id=job_id,
            status="cancel_requested",
            message="Cancellation requested. Send confirm=true to confirm cancellation.",
        )

    # Step 2: Confirm cancellation
    if body.confirm:
        success = confirm_cancellation(job_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail="No pending cancellation request. Request cancellation first.",
            )

        # Revoke the Celery task
        result.revoke(terminate=True)

        return CancelResponse(
            job_id=job_id,
            status="cancelled",
            message="Job cancellation confirmed. Task will abort at next checkpoint.",
        )


@router.get("/{job_id}/download")
async def download_results(job_id: str):
    """
    Download all job results as a single ZIP file.

    ZIP contains: mesh files (OBJ, PLY, GLB), textures, preview images, quality.json

    Returns:
        StreamingResponse: ZIP file

    Raises:
        HTTPException 404: Job not found (never existed)
        HTTPException 409: Job not completed yet (still processing)
        HTTPException 410: Job results expired/cleaned up
        HTTPException 500: Job failed or outputs incomplete
    """
    # Get Celery task result
    result = celery_app.AsyncResult(job_id)
    state = result.state

    # Check job state and return appropriate errors
    if state == "PENDING":
        # Job never existed or expired from Celery
        raise HTTPException(
            status_code=404,
            detail=make_error_detail(
                ErrorCode.JOB_NOT_FOUND,
                f"Job '{job_id}' not found",
                details={"job_id": job_id}
            )
        )

    if state in ["STARTED", "PROGRESS"]:
        # Job still processing
        progress = 0
        if state == "PROGRESS" and result.info and isinstance(result.info, dict):
            progress = result.info.get("progress", 0)

        raise HTTPException(
            status_code=409,
            detail=make_error_detail(
                ErrorCode.JOB_NOT_READY,
                f"Job '{job_id}' is still processing ({progress}% complete)",
                details={"job_id": job_id, "progress": progress, "state": state}
            )
        )

    if state == "FAILURE":
        # Job failed
        error_msg = str(result.info) if result.info else "Unknown error"
        raise HTTPException(
            status_code=500,
            detail=make_error_detail(
                ErrorCode.MODEL_FAILED,
                f"Job '{job_id}' failed: {error_msg}",
                details={"job_id": job_id, "error": error_msg}
            )
        )

    if state == "REVOKED":
        # Job was cancelled
        raise HTTPException(
            status_code=500,
            detail=make_error_detail(
                ErrorCode.MODEL_FAILED,
                f"Job '{job_id}' was cancelled",
                details={"job_id": job_id, "cancelled": True}
            )
        )

    # SUCCESS - check output directory exists
    job_path = get_job_path(job_id)
    output_dir = job_path / "output"

    if not output_dir.exists():
        # Results were cleaned up or never created
        raise HTTPException(
            status_code=410,
            detail=make_error_detail(
                ErrorCode.JOB_EXPIRED,
                f"Job '{job_id}' results have expired or been cleaned up",
                details={"job_id": job_id}
            )
        )

    # Validate outputs exist
    is_valid, missing = validate_job_outputs(output_dir)
    if not is_valid:
        raise HTTPException(
            status_code=500,
            detail=make_error_detail(
                ErrorCode.INCOMPLETE_RESULTS,
                f"Job '{job_id}' has incomplete results",
                details={"job_id": job_id, "missing": missing}
            )
        )

    # Create ZIP
    try:
        zip_buffer = create_result_zip(job_id, output_dir)
    except IncompleteResultsError as e:
        raise HTTPException(
            status_code=500,
            detail=make_error_detail(
                ErrorCode.INCOMPLETE_RESULTS,
                e.message,
                details={"job_id": job_id, "missing": e.missing_items}
            )
        )

    # Return streaming response with ZIP
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_id}.zip"}
    )
