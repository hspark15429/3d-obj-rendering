"""
Celery task for 3D reconstruction processing.

Executes ReconViaGen and/or nvdiffrec models to produce textured mesh outputs.
Supports single model execution or sequential execution of both models.
"""
import logging
import shutil
from pathlib import Path
from typing import Literal

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from app.models import get_model, AVAILABLE_MODELS
from app.services.job_manager import is_job_cancelled, clear_cancellation
from app.services.file_handler import delete_job_files, get_job_path, STORAGE_ROOT
from app.services.vram_manager import cleanup_gpu_memory
from app.services.preview_generator import PreviewGenerator

logger = logging.getLogger(__name__)

# Model type including 'both' option
ModelSelection = Literal['reconviagen', 'nvdiffrec', 'both']


@shared_task(
    bind=True,
    name="reconstruction.process",
    soft_time_limit=7200,   # 2 hours soft limit (for 'both' mode)
    time_limit=7500,        # 2 hours 5 min hard limit
    acks_late=True,
    reject_on_worker_lost=True
)
def process_reconstruction(
    self,
    job_id: str,
    model_type: ModelSelection = "reconviagen"
) -> dict:
    """
    Process 3D reconstruction from uploaded images.

    Executes selected model(s) to produce textured mesh outputs.
    For 'both' mode, runs ReconViaGen first, then nvdiffrec sequentially
    with VRAM cleanup between models.

    Args:
        self: Celery task instance (bind=True provides this)
        job_id: Job identifier
        model_type: 'reconviagen', 'nvdiffrec', or 'both'

    Returns:
        dict: Result with status and output paths
            - {"status": "completed", "job_id": str, "outputs": dict}
            - {"status": "cancelled", "job_id": str}
            - {"status": "failed", "job_id": str, "error": str}
    """
    logger.info(f"Starting reconstruction job {job_id} with model_type={model_type}")

    # Get input directory
    job_dir = get_job_path(job_id)
    input_dir = job_dir

    if not input_dir.exists():
        logger.error(f"Job directory not found: {input_dir}")
        return {
            "status": "failed",
            "job_id": job_id,
            "error": "Job input files not found"
        }

    # Determine which models to run
    if model_type == 'both':
        models_to_run = ['reconviagen', 'nvdiffrec']
    else:
        models_to_run = [model_type]

    # Track results per model
    outputs = {}
    total_models = len(models_to_run)

    try:
        for model_index, current_model in enumerate(models_to_run):
            # Check for cancellation before each model
            if is_job_cancelled(job_id):
                logger.info(f"Job {job_id} cancelled before running {current_model}")
                delete_job_files(job_id)
                self.update_state(
                    state="REVOKED",
                    meta={"status": "cancelled", "job_id": job_id}
                )
                return {"status": "cancelled", "job_id": job_id}

            # Calculate progress offset for multi-model execution
            # Each model gets a portion of 0-100%
            progress_offset = int(100 * model_index / total_models)
            progress_scale = 100 / total_models

            logger.info(f"Running model {model_index + 1}/{total_models}: {current_model}")

            # Update progress: starting model
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress_offset,
                    "step": f"Starting {current_model}",
                    "model": current_model,
                    "model_index": model_index + 1,
                    "total_models": total_models
                }
            )

            # Create output directory for this model
            output_dir = job_dir / "output" / current_model
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get and run model
            model = get_model(current_model, celery_task=self)

            try:
                # Load weights
                model.load_weights()

                # Check cancellation after weight loading
                if is_job_cancelled(job_id):
                    logger.info(f"Job {job_id} cancelled after loading {current_model}")
                    model.cleanup()
                    cleanup_gpu_memory()
                    delete_job_files(job_id)
                    return {"status": "cancelled", "job_id": job_id}

                # Run inference
                result = model.inference(input_dir, output_dir)

                # Cleanup model resources
                model.cleanup()

            except Exception as model_error:
                logger.error(f"Model {current_model} failed: {model_error}", exc_info=True)
                model.cleanup()
                cleanup_gpu_memory()

                # Per locked decision: stop on first failure
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "error": f"Model failed to process images",
                    "model": current_model
                }

            # Check model result
            if result['status'] != 'success':
                logger.error(f"Model {current_model} returned failure: {result.get('error')}")

                # Per locked decision: stop on first failure
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "error": result.get('error', 'Unknown model error'),
                    "model": current_model
                }

            # Store output paths
            outputs[current_model] = {
                "mesh_obj": result.get('mesh_path'),
                "mesh_ply": result.get('ply_path'),
                "texture": result.get('texture_path')
            }

            # Run quality pipeline for completed model
            # Per CONTEXT.md: quality metrics are required, if rendering fails job fails
            try:
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "progress": progress_offset + int(progress_scale * 0.8),
                        "step": f"Computing quality metrics for {current_model}",
                        "model": current_model,
                        "model_index": model_index + 1,
                        "total_models": total_models
                    }
                )

                preview_gen = PreviewGenerator()

                # Determine mesh path for quality pipeline (prefer GLB)
                mesh_path = result.get('mesh_path')
                if mesh_path:
                    mesh_path = Path(mesh_path)
                else:
                    # Fallback to OBJ if GLB not available
                    mesh_path = output_dir / "mesh.glb"
                    if not mesh_path.exists():
                        mesh_path = output_dir / "mesh.obj"

                quality_result = preview_gen.generate_all(
                    job_id=job_id,
                    model=current_model,
                    mesh_path=mesh_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    metadata={
                        "input_file": "uploaded.zip",
                        "duration": None  # TODO: track actual duration
                    }
                )

                # Add quality info to outputs
                outputs[current_model]["quality"] = quality_result["quality_report"]
                outputs[current_model]["previews"] = {
                    "textured": [str(p) for p in quality_result["previews"]["textured"]],
                    "wireframe": [str(p) for p in quality_result["previews"]["wireframe"]]
                }
                outputs[current_model]["quality_status"] = quality_result["status"]

                logger.info(f"Quality pipeline complete for {current_model}: status={quality_result['status']}")

            except Exception as quality_error:
                logger.error(f"Quality pipeline failed for {current_model}: {quality_error}", exc_info=True)
                # Per CONTEXT.md: quality metrics are required, if rendering fails job fails
                cleanup_gpu_memory()
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "error": f"Quality metrics computation failed: {str(quality_error)}",
                    "model": current_model
                }

            # CRITICAL: Cleanup VRAM before next model (for 'both' mode)
            if model_index < total_models - 1:
                logger.info(f"Cleaning up VRAM before next model")
                cleanup_gpu_memory()

                # Check cancellation between models
                if is_job_cancelled(job_id):
                    logger.info(f"Job {job_id} cancelled between models")
                    delete_job_files(job_id)
                    return {"status": "cancelled", "job_id": job_id}

        # All models completed successfully
        logger.info(f"Job {job_id} completed successfully with {total_models} model(s)")

        # Final progress update
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": 100,
                "step": "Complete",
                "models_completed": list(outputs.keys())
            }
        )

        # Clear cancellation flags
        clear_cancellation(job_id)

        # Determine overall quality status from last model
        models_completed = list(outputs.keys())
        overall_quality_status = "unknown"
        if models_completed:
            last_model = models_completed[-1]
            overall_quality_status = outputs[last_model].get("quality_status", "unknown")

        return {
            "status": "completed",
            "job_id": job_id,
            "outputs": outputs,
            "models_run": models_completed,
            "quality_status": overall_quality_status
        }

    except SoftTimeLimitExceeded:
        logger.warning(f"Job {job_id} exceeded time limit")
        cleanup_gpu_memory()
        delete_job_files(job_id)
        return {
            "status": "failed",
            "job_id": job_id,
            "error": "Job exceeded time limit"
        }

    except Exception as e:
        logger.error(f"Unexpected error in job {job_id}: {e}", exc_info=True)
        cleanup_gpu_memory()
        return {
            "status": "failed",
            "job_id": job_id,
            "error": "An unexpected error occurred"
        }
