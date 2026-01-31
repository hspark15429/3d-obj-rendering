"""
Job API router for 3D reconstruction service.

Endpoints:
- POST /jobs - Submit new reconstruction job
- GET /jobs/{job_id} - Get job status
- POST /jobs/{job_id}/cancel - Cancel job (two-step)
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from nanoid import generate

from app.api.schemas import (
    JobSubmitResponse,
    JobStatusResponse,
    CancelRequest,
    CancelResponse,
    JobStatus,
)
from app.services.file_handler import (
    validate_upload_files,
    save_job_files,
    FileValidationError,
)
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
    model_type: str = Query("reconviagen", description="Model: reconviagen or nvdiffrec"),
):
    """
    Submit a new 3D reconstruction job.

    Accepts 12 PNG files (6 multi-view images + 6 depth renders) and queues
    a reconstruction task. Returns immediately with job ID for status polling.

    Args:
        views: List of 6 multi-view PNG images
        depth_renders: List of 6 depth render PNG images
        model_type: Reconstruction model to use

    Returns:
        JobSubmitResponse: Job ID, status (queued), and creation timestamp

    Raises:
        HTTPException 400: Invalid model type or file validation failed
    """
    # Validate model type
    if model_type not in ["reconviagen", "nvdiffrec"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type '{model_type}'. Must be 'reconviagen' or 'nvdiffrec'."
        )

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

    # Queue reconstruction task (use job_id as Celery task_id for status lookup)
    task = process_reconstruction.apply_async(args=(job_id, model_type), task_id=job_id)

    # Return response
    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of a reconstruction job.

    Queries Celery for task state and maps to job status.
    Shows progress percentage when task is processing.

    Args:
        job_id: Job identifier

    Returns:
        JobStatusResponse: Current job status with progress

    State mapping:
        - PENDING -> queued
        - STARTED -> processing (progress=0)
        - PROGRESS -> processing (progress from task)
        - SUCCESS -> completed
        - FAILURE -> failed
        - REVOKED -> cancelled
    """
    # Get task result from Celery
    result = celery_app.AsyncResult(job_id)

    # Map Celery state to JobStatus
    state = result.state
    progress = None
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
