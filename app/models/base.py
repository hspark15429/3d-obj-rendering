"""
Abstract base class for 3D reconstruction models.

Provides standardized interface for ReconViaGen, nvdiffrec, and other
reconstruction models. Each model wrapper must implement load_weights()
and inference() methods.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any


class BaseReconstructionModel(ABC):
    """Abstract base for reconstruction models (ReconViaGen, nvdiffrec)."""

    model_name: str = "base"  # Override in subclasses

    def __init__(self, celery_task: Optional[Any] = None):
        """
        Initialize model wrapper.

        Args:
            celery_task: Optional Celery task for progress reporting.
                        Must have update_state() method.
        """
        self.celery_task = celery_task
        self._model = None

    @abstractmethod
    def load_weights(self) -> None:
        """
        Load pre-downloaded model weights from /app/weights/.

        Weights are expected to be pre-downloaded during Docker build
        or volume mount. This method should load them into GPU memory.

        Raises:
            FileNotFoundError: If weights file doesn't exist
            RuntimeError: If CUDA not available or insufficient VRAM
        """
        pass

    @abstractmethod
    def inference(self, input_dir: Path, output_dir: Path) -> dict:
        """
        Run inference on input images, produce mesh output.

        Args:
            input_dir: Path containing views/ and depth/ subdirectories
                      with preprocessed images
            output_dir: Path to write mesh.obj, mesh.ply, texture.png

        Returns:
            dict with:
                'status': 'success' or 'failed'
                'error': Error message (if failed)
                'mesh_path': Path to OBJ file (if success)
                'ply_path': Path to PLY file (if success)
                'texture_path': Path to texture image (if success)
        """
        pass

    def report_progress(self, progress: int, step: str) -> None:
        """
        Report progress to Celery task.

        Args:
            progress: Progress percentage (0-100)
            step: Current step description
        """
        if self.celery_task:
            self.celery_task.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'step': f'{self.model_name}: {step}'
                }
            )

    def cleanup(self) -> None:
        """
        Release model resources. Call after inference completes.

        This method should be called to free GPU memory before
        running another model. Uses VRAM manager for full cleanup.
        """
        if self._model is not None:
            del self._model
            self._model = None

        # Import here to avoid circular dependency
        from app.services.vram_manager import cleanup_gpu_memory
        cleanup_gpu_memory()
