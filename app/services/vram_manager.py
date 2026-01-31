"""
GPU VRAM management utilities.

Provides functions for cleaning up GPU memory between model runs
and checking VRAM availability before loading models.
"""
import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def cleanup_gpu_memory() -> dict:
    """
    Complete GPU memory cleanup pattern.

    Order matters:
    1. Force Python garbage collection
    2. Clear PyTorch's caching allocator

    Returns:
        dict with 'allocated_gb' and 'reserved_gb' after cleanup,
        or 'error' key if CUDA not available
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU cleanup")
        return {'error': 'CUDA not available', 'allocated_gb': 0, 'reserved_gb': 0}

    gc.collect()
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    logger.info(f"VRAM after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return {
        'allocated_gb': round(allocated, 2),
        'reserved_gb': round(reserved, 2)
    }


def check_vram_available(required_gb: float = 10.0) -> dict:
    """
    Check if sufficient VRAM is available.

    Args:
        required_gb: Minimum required VRAM in GB (default 10GB for reconstruction models)

    Returns:
        dict with:
            'available': bool - True if enough VRAM free
            'free_gb': float - Currently free VRAM
            'total_gb': float - Total GPU memory
            'allocated_gb': float - Currently allocated
            'error': str - Error message if CUDA not available
    """
    if not torch.cuda.is_available():
        return {
            'available': False,
            'free_gb': 0,
            'total_gb': 0,
            'allocated_gb': 0,
            'error': 'CUDA not available'
        }

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    free = total - allocated

    return {
        'available': free >= required_gb,
        'free_gb': round(free, 2),
        'total_gb': round(total, 2),
        'allocated_gb': round(allocated, 2)
    }


def get_vram_usage() -> dict:
    """
    Get current VRAM usage stats.

    Returns:
        dict with 'allocated_gb', 'reserved_gb', 'total_gb',
        or 'error' if CUDA not available
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}

    return {
        'allocated_gb': round(torch.cuda.memory_allocated() / (1024**3), 2),
        'reserved_gb': round(torch.cuda.memory_reserved() / (1024**3), 2),
        'total_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    }


def get_gpu_info() -> dict:
    """
    Get GPU device information.

    Returns:
        dict with 'name', 'compute_capability', 'total_memory_gb',
        or 'error' if CUDA not available
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}

    props = torch.cuda.get_device_properties(0)
    return {
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'total_memory_gb': round(props.total_memory / (1024**3), 2),
        'device_count': torch.cuda.device_count()
    }
