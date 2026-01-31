# Services package
#
# Uses lazy imports to avoid loading heavy dependencies (torch, nvdiffrast)
# until they are actually needed. This allows running tests without GPU.

from typing import TYPE_CHECKING

# Only import type hints at type-check time
if TYPE_CHECKING:
    from app.services.quality_metrics import (
        compute_psnr,
        compute_ssim,
        compute_depth_error,
        compute_overall_metrics,
        classify_quality_status,
        load_image_float,
        PSNR_NORMAL_THRESHOLD,
        PSNR_WARNING_THRESHOLD,
        SSIM_NORMAL_THRESHOLD,
        SSIM_WARNING_THRESHOLD,
    )
    from app.services.mesh_renderer import (
        MeshRenderer,
        load_camera_poses,
        build_mvp_matrix,
        render_mesh,
        render_depth,
    )
    from app.services.preview_generator import (
        PreviewGenerator,
        generate_quality_report,
        generate_previews,
    )


# Lazy loading via __getattr__ for runtime imports
def __getattr__(name: str):
    """Lazy import of service modules to defer heavy dependencies."""
    # Quality metrics (no heavy dependencies)
    if name in {
        "compute_psnr",
        "compute_ssim",
        "compute_depth_error",
        "compute_overall_metrics",
        "classify_quality_status",
        "load_image_float",
        "PSNR_NORMAL_THRESHOLD",
        "PSNR_WARNING_THRESHOLD",
        "SSIM_NORMAL_THRESHOLD",
        "SSIM_WARNING_THRESHOLD",
    }:
        from app.services import quality_metrics
        return getattr(quality_metrics, name)

    # Mesh renderer (requires torch/nvdiffrast)
    if name in {
        "MeshRenderer",
        "load_camera_poses",
        "build_mvp_matrix",
        "render_mesh",
        "render_depth",
    }:
        from app.services import mesh_renderer
        return getattr(mesh_renderer, name)

    # Preview generator (requires mesh_renderer)
    if name in {
        "PreviewGenerator",
        "generate_quality_report",
        "generate_previews",
    }:
        from app.services import preview_generator
        return getattr(preview_generator, name)

    raise AttributeError(f"module 'app.services' has no attribute '{name}'")


__all__ = [
    # Quality metrics
    "compute_psnr",
    "compute_ssim",
    "compute_depth_error",
    "compute_overall_metrics",
    "classify_quality_status",
    "load_image_float",
    "PSNR_NORMAL_THRESHOLD",
    "PSNR_WARNING_THRESHOLD",
    "SSIM_NORMAL_THRESHOLD",
    "SSIM_WARNING_THRESHOLD",
    # Mesh renderer
    "MeshRenderer",
    "load_camera_poses",
    "build_mvp_matrix",
    "render_mesh",
    "render_depth",
    # Preview generator
    "PreviewGenerator",
    "generate_quality_report",
    "generate_previews",
]
