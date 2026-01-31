"""
3D reconstruction model wrappers.

Provides standardized interfaces to ReconViaGen and nvdiffrec models
for the reconstruction task.

Usage:
    from app.models import get_model

    model = get_model('reconviagen', celery_task=self)
    model.load_weights()
    result = model.inference(input_dir, output_dir)
"""
from typing import Optional, Literal

from app.models.base import BaseReconstructionModel
from app.models.reconviagen import ReconViaGenModel
from app.models.nvdiffrec import NvdiffrecModel

__all__ = [
    'BaseReconstructionModel',
    'ReconViaGenModel',
    'NvdiffrecModel',
    'get_model',
    'AVAILABLE_MODELS',
]

# Model type literal for type hints
ModelType = Literal['reconviagen', 'nvdiffrec']

# Available model types
AVAILABLE_MODELS = ['reconviagen', 'nvdiffrec']


def get_model(
    model_type: ModelType,
    celery_task=None
) -> BaseReconstructionModel:
    """
    Factory function to get model instance by type.

    Args:
        model_type: 'reconviagen' or 'nvdiffrec'
        celery_task: Optional Celery task for progress reporting

    Returns:
        Model instance implementing BaseReconstructionModel

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == 'reconviagen':
        return ReconViaGenModel(celery_task=celery_task)
    elif model_type == 'nvdiffrec':
        return NvdiffrecModel(celery_task=celery_task)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {', '.join(AVAILABLE_MODELS)}"
        )
