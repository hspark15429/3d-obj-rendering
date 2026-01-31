"""
Celery task for 3D reconstruction processing.

This is a PLACEHOLDER implementation. Phase 3 will replace sleep() calls
with actual model inference (ReconViaGen/nvdiffrec).

Progress tracking and cancellation patterns are production-ready.
"""
import logging
import time
from celery import shared_task
from app.services.job_manager import is_job_cancelled, clear_cancellation
from app.services.file_handler import delete_job_files

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="reconstruction.process")
def process_reconstruction(self, job_id: str, model_type: str = "reconviagen") -> dict:
    """
    Process 3D reconstruction from uploaded images.

    Demonstrates:
    - Progress tracking via self.update_state()
    - Cancellation checking at each step
    - File cleanup on cancellation
    - Placeholder steps (Phase 3 adds real model calls)

    Args:
        self: Celery task instance (bind=True provides this)
        job_id: Job identifier
        model_type: Model to use ("reconviagen" or "nvdiffrec")

    Returns:
        dict: Result with status and output path
            - {"status": "completed", "job_id": str, "output_path": str}
            - {"status": "cancelled", "job_id": str}
    """
    logger.info(f"Starting reconstruction job {job_id} with model {model_type}")

    # Define processing steps with progress percentages
    # Phase 3 will replace sleep() with actual model calls
    steps = [
        ("Loading input files", 10),
        ("Preprocessing images", 20),
        ("Running reconstruction", 60),  # This will be the actual model inference
        ("Post-processing mesh", 80),
        ("Generating previews", 90),
        ("Finalizing output", 100),
    ]

    # Process each step
    for step_name, progress_percent in steps:
        # CRITICAL: Check for cancellation before each step
        if is_job_cancelled(job_id):
            logger.info(f"Job {job_id} cancelled at step: {step_name}")

            # Clean up job files
            delete_job_files(job_id)

            # Update task state to REVOKED
            self.update_state(
                state="REVOKED",
                meta={"status": "cancelled", "job_id": job_id}
            )

            # Return cancellation result
            return {"status": "cancelled", "job_id": job_id}

        # Log current step
        logger.info(f"Job {job_id}: {step_name} ({progress_percent}%)")

        # Report progress to Celery
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": progress_percent,
                "step": step_name
            }
        )

        # Placeholder work (Phase 3 replaces this with model calls)
        time.sleep(2)

    # Job completed successfully
    logger.info(f"Job {job_id} completed successfully")

    # Clean up cancellation flags
    clear_cancellation(job_id)

    # Return success result
    return {
        "status": "completed",
        "job_id": job_id,
        "output_path": f"/jobs/{job_id}/output"
    }
