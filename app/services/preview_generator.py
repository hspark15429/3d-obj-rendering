"""
Preview generation service for 3D reconstruction quality assessment.

This module provides preview image generation (textured + wireframe from 6 angles)
and quality report generation for completed reconstruction jobs.

Usage:
    from app.services.preview_generator import PreviewGenerator, generate_quality_report

    # Generate previews and quality metrics
    generator = PreviewGenerator()
    result = generator.generate_all(
        job_id="abc123",
        model="reconviagen",
        mesh_path=Path("output/mesh.glb"),
        input_dir=Path("input/"),
        output_dir=Path("output/"),
        metadata={"input_file": "input.zip", "duration": 125.3}
    )

    # result contains: previews, quality_report, status
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Lazy import of mesh_renderer to avoid torch dependency at import time
# This allows generate_quality_report() to be used without torch
if TYPE_CHECKING:
    from app.services.mesh_renderer import MeshRenderer

from app.services.quality_metrics import (
    classify_quality_status,
    compute_depth_error,
    compute_overall_metrics,
    compute_psnr,
    compute_ssim,
    load_image_float,
    PSNR_NORMAL_THRESHOLD,
    PSNR_WARNING_THRESHOLD,
    SSIM_NORMAL_THRESHOLD,
    SSIM_WARNING_THRESHOLD,
)

logger = logging.getLogger(__name__)


class PreviewGenerator:
    """
    Preview image generator using nvdiffrast mesh renderer.

    Generates textured and wireframe preview images from canonical camera angles,
    computes quality metrics by comparing renders to input images.
    """

    def __init__(self, renderer: Optional["MeshRenderer"] = None):
        """
        Initialize preview generator.

        Args:
            renderer: Optional MeshRenderer instance. If None, creates one.
        """
        self._renderer = renderer

    @property
    def renderer(self) -> "MeshRenderer":
        """Lazy-load renderer on first access."""
        if self._renderer is None:
            from app.services.mesh_renderer import MeshRenderer
            self._renderer = MeshRenderer()
        return self._renderer

    def generate_previews(
        self,
        mesh_path: Path,
        output_dir: Path,
        camera_poses: Dict,
        resolution: Tuple[int, int]
    ) -> Dict[str, List[Path]]:
        """
        Generate preview images from mesh at specified camera poses.

        Renders both textured and wireframe images from each camera pose,
        saving as PNG files.

        Args:
            mesh_path: Path to GLB mesh file
            output_dir: Directory to save preview images
            camera_poses: Camera poses dict from load_camera_poses()
            resolution: Output image resolution (height, width)

        Returns:
            Dict with "textured" and "wireframe" keys, each containing
            list of Path objects for saved preview images.

        Raises:
            FileNotFoundError: If mesh file not found
            RuntimeError: If rendering fails
        """
        mesh_path = Path(mesh_path)
        output_dir = Path(output_dir)

        # Create previews subdirectory per result_packager.py expectations
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating preview images from {mesh_path}")
        logger.info(f"Resolution: {resolution}, poses: {len(camera_poses['frames'])}")

        # Load mesh once
        mesh = self.renderer.load_mesh(mesh_path)

        textured_paths: List[Path] = []
        wireframe_paths: List[Path] = []

        camera_angle_x = camera_poses["camera_angle_x"]
        frames = camera_poses["frames"]

        # Use up to 6 camera poses for previews
        num_previews = min(6, len(frames))

        for i in range(num_previews):
            frame = frames[i]
            transform_matrix = frame["transform_matrix"]

            logger.debug(f"Rendering preview {i + 1}/{num_previews}")

            # Build MVP matrix (lazy import)
            from app.services.mesh_renderer import build_mvp_matrix
            mvp = build_mvp_matrix(transform_matrix, camera_angle_x, resolution)

            # Render textured image
            textured = self.renderer.render_textured(mesh, mvp, resolution)
            textured_uint8 = (textured * 255).astype(np.uint8)
            textured_path = previews_dir / f"textured_{i:02d}.png"
            Image.fromarray(textured_uint8).save(textured_path)
            textured_paths.append(textured_path)

            # Render wireframe image
            wireframe = self.renderer.render_wireframe(mesh, mvp, resolution)
            wireframe_path = previews_dir / f"wireframe_{i:02d}.png"
            Image.fromarray(wireframe).save(wireframe_path)
            wireframe_paths.append(wireframe_path)

            logger.debug(f"Saved: {textured_path.name}, {wireframe_path.name}")

        logger.info(
            f"Generated {len(textured_paths)} textured and "
            f"{len(wireframe_paths)} wireframe previews"
        )

        return {
            "textured": textured_paths,
            "wireframe": wireframe_paths,
        }

    def compute_quality_metrics(
        self,
        mesh_path: Path,
        input_dir: Path,
        camera_poses: Dict,
        resolution: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Compute quality metrics by comparing rendered mesh to input images.

        For each camera pose:
        - Renders mesh to match input view
        - Computes PSNR/SSIM between rendered and input image
        - Computes depth MAE/RMSE if depth maps available

        Args:
            mesh_path: Path to GLB mesh file
            input_dir: Directory containing input images and depth maps
                - views/ subdirectory with view_XX.png images
                - depth/ subdirectory with depth_XX.png (optional)
            camera_poses: Camera poses dict from load_camera_poses()
            resolution: Rendering resolution (height, width)

        Returns:
            Dict with "psnr", "ssim", "depth_mae", "depth_rmse" keys.
            Depth metrics are 0.0 if no depth maps available.

        Raises:
            FileNotFoundError: If mesh or input images not found
            RuntimeError: If rendering fails
        """
        mesh_path = Path(mesh_path)
        input_dir = Path(input_dir)

        logger.info(f"Computing quality metrics for {mesh_path}")

        # Load mesh once
        mesh = self.renderer.load_mesh(mesh_path)

        camera_angle_x = camera_poses["camera_angle_x"]
        frames = camera_poses["frames"]

        # Prepare for metric computation
        image_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        depth_maes: List[float] = []
        depth_rmses: List[float] = []

        # Determine input paths
        views_dir = input_dir / "views"
        depth_dir = input_dir / "depth"

        for i, frame in enumerate(frames):
            transform_matrix = frame["transform_matrix"]
            file_path = frame.get("file_path", f"view_{i:02d}")

            # Determine input image path
            # Try views/ subdirectory first, then root input_dir
            input_image_path = None
            for candidate in [
                views_dir / f"view_{i:02d}.png",
                views_dir / f"{Path(file_path).stem}.png",
                input_dir / f"view_{i:02d}.png",
                input_dir / f"{Path(file_path).stem}.png",
            ]:
                if candidate.exists():
                    input_image_path = candidate
                    break

            if input_image_path is None:
                logger.warning(f"Input image not found for frame {i}, skipping")
                continue

            logger.debug(f"Processing frame {i}: {input_image_path.name}")

            # Build MVP and render (lazy import)
            from app.services.mesh_renderer import build_mvp_matrix
            mvp = build_mvp_matrix(transform_matrix, camera_angle_x, resolution)
            rendered = self.renderer.render_textured(mesh, mvp, resolution)

            # Load input image
            input_img = load_image_float(input_image_path)

            # Resize if needed (in case input resolution differs)
            if input_img.shape[:2] != rendered.shape[:2]:
                logger.debug(
                    f"Resizing input from {input_img.shape[:2]} to {rendered.shape[:2]}"
                )
                input_pil = Image.fromarray((input_img * 255).astype(np.uint8))
                input_pil = input_pil.resize(
                    (resolution[1], resolution[0]), Image.BILINEAR
                )
                input_img = np.array(input_pil, dtype=np.float32) / 255.0

            # Collect image pair for PSNR/SSIM
            image_pairs.append((input_img, rendered))

            # Compute depth metrics if depth maps available
            depth_path = None
            for candidate in [
                depth_dir / f"depth_{i:02d}.png",
                depth_dir / f"{Path(file_path).stem}_depth.png",
                input_dir / f"depth_{i:02d}.png",
            ]:
                if candidate.exists():
                    depth_path = candidate
                    break

            if depth_path is not None:
                try:
                    # Load input depth
                    input_depth = np.array(
                        Image.open(depth_path).convert("L"), dtype=np.float32
                    ) / 255.0

                    # Render depth
                    rendered_depth = self.renderer.render_depth(mesh, mvp, resolution)

                    # Normalize rendered depth to [0, 1] range
                    depth_min = rendered_depth[rendered_depth > 0].min() if (rendered_depth > 0).any() else 0
                    depth_max = rendered_depth.max()
                    if depth_max > depth_min:
                        rendered_depth_norm = (rendered_depth - depth_min) / (depth_max - depth_min)
                        rendered_depth_norm = np.where(rendered_depth > 0, rendered_depth_norm, 0)
                    else:
                        rendered_depth_norm = rendered_depth

                    # Resize input depth if needed
                    if input_depth.shape != rendered_depth_norm.shape:
                        input_depth_pil = Image.fromarray(
                            (input_depth * 255).astype(np.uint8)
                        )
                        input_depth_pil = input_depth_pil.resize(
                            (resolution[1], resolution[0]), Image.BILINEAR
                        )
                        input_depth = np.array(input_depth_pil, dtype=np.float32) / 255.0

                    # Compute depth error
                    depth_metrics = compute_depth_error(input_depth, rendered_depth_norm)
                    depth_maes.append(depth_metrics["mae"])
                    depth_rmses.append(depth_metrics["rmse"])
                except Exception as e:
                    logger.warning(f"Depth comparison failed for frame {i}: {e}")

        # Compute overall metrics
        if not image_pairs:
            raise RuntimeError("No valid image pairs for quality computation")

        overall = compute_overall_metrics(image_pairs)

        # Average depth metrics
        depth_mae = float(np.mean(depth_maes)) if depth_maes else 0.0
        depth_rmse = float(np.mean(depth_rmses)) if depth_rmses else 0.0

        result = {
            "psnr": overall["psnr"],
            "ssim": overall["ssim"],
            "depth_mae": depth_mae,
            "depth_rmse": depth_rmse,
        }

        logger.info(
            f"Quality metrics: PSNR={result['psnr']:.2f} dB, "
            f"SSIM={result['ssim']:.4f}, "
            f"depth_mae={result['depth_mae']:.4f}"
        )

        return result

    def generate_all(
        self,
        job_id: str,
        model: str,
        mesh_path: Path,
        input_dir: Path,
        output_dir: Path,
        metadata: Dict
    ) -> Dict:
        """
        Main entry point for quality pipeline.

        Loads camera poses, generates preview images, computes quality metrics,
        and generates quality report.

        Args:
            job_id: Job identifier
            model: Model name used for reconstruction
            mesh_path: Path to GLB mesh file
            input_dir: Directory containing input images and transforms_train.json
            output_dir: Directory to save outputs (previews and quality.json)
            metadata: Additional metadata dict with optional keys:
                - input_file: Original input filename
                - duration: Processing duration in seconds

        Returns:
            Dict with:
                - previews: Dict with textured and wireframe path lists
                - quality_report: Dict containing quality.json contents
                - status: Quality status string ("normal", "warning", "failure")

        Raises:
            FileNotFoundError: If required files not found
            RuntimeError: If rendering or quality computation fails
        """
        mesh_path = Path(mesh_path)
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        logger.info(f"Starting quality pipeline for job {job_id}")
        logger.info(f"Mesh: {mesh_path}, Input: {input_dir}, Output: {output_dir}")

        # Load camera poses (lazy import)
        from app.services.mesh_renderer import load_camera_poses
        camera_poses = load_camera_poses(input_dir)
        logger.info(f"Loaded {len(camera_poses['frames'])} camera poses")

        # Extract resolution from first input image
        resolution = self._get_input_resolution(input_dir, camera_poses)
        logger.info(f"Using resolution: {resolution}")

        # Generate preview images
        previews = self.generate_previews(
            mesh_path=mesh_path,
            output_dir=output_dir,
            camera_poses=camera_poses,
            resolution=resolution,
        )

        # Compute quality metrics
        metrics = self.compute_quality_metrics(
            mesh_path=mesh_path,
            input_dir=input_dir,
            camera_poses=camera_poses,
            resolution=resolution,
        )

        # Update metadata with resolution
        full_metadata = {
            **metadata,
            "resolution": list(resolution),
        }

        # Generate quality report
        quality_path = output_dir / "quality.json"
        quality_report = generate_quality_report(
            job_id=job_id,
            model=model,
            metrics=metrics,
            metadata=full_metadata,
            output_path=quality_path,
        )

        status = quality_report["status"]

        logger.info(f"Quality pipeline complete: status={status}")

        return {
            "previews": previews,
            "quality_report": quality_report,
            "status": status,
        }

    def _get_input_resolution(
        self,
        input_dir: Path,
        camera_poses: Dict
    ) -> Tuple[int, int]:
        """
        Extract resolution from first input image.

        Args:
            input_dir: Input directory
            camera_poses: Camera poses dict

        Returns:
            Tuple of (height, width)

        Raises:
            FileNotFoundError: If no input images found
        """
        views_dir = input_dir / "views"
        frames = camera_poses.get("frames", [])

        # Try multiple candidate paths
        candidates = [
            views_dir / "view_00.png",
            input_dir / "view_00.png",
        ]

        if frames:
            file_path = frames[0].get("file_path", "")
            candidates.extend([
                views_dir / f"{Path(file_path).stem}.png",
                input_dir / f"{Path(file_path).stem}.png",
            ])

        for candidate in candidates:
            if candidate.exists():
                img = Image.open(candidate)
                return (img.height, img.width)

        # Fallback to default resolution
        logger.warning("Could not find input images, using default 512x512 resolution")
        return (512, 512)


def generate_quality_report(
    job_id: str,
    model: str,
    metrics: Dict[str, float],
    metadata: Dict,
    output_path: Path
) -> Dict:
    """
    Generate and save quality report JSON.

    Creates quality.json with status classification, metrics, thresholds,
    and metadata per RESEARCH.md specification.

    Args:
        job_id: Job identifier
        model: Model name used for reconstruction
        metrics: Dict with psnr, ssim, depth_mae, depth_rmse values
        metadata: Dict with optional input_file, duration, resolution
        output_path: Path to save quality.json

    Returns:
        Dict containing the full quality report structure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Classify quality status
    psnr = metrics.get("psnr", 0.0)
    ssim = metrics.get("ssim", 0.0)
    status = classify_quality_status(psnr, ssim)

    # Build human-readable summary
    summary = f"Reconstruction quality: {status.title()} (PSNR {psnr:.1f} dB, SSIM {ssim:.2f})"

    # Build quality report
    report = {
        "job_id": job_id,
        "model": model,
        "status": status,
        "summary": summary,
        "metrics": {
            "psnr": metrics.get("psnr", 0.0),
            "ssim": metrics.get("ssim", 0.0),
            "depth_mae": metrics.get("depth_mae", 0.0),
            "depth_rmse": metrics.get("depth_rmse", 0.0),
        },
        "thresholds": {
            "psnr_normal": PSNR_NORMAL_THRESHOLD,
            "psnr_warning": PSNR_WARNING_THRESHOLD,
            "ssim_normal": SSIM_NORMAL_THRESHOLD,
            "ssim_warning": SSIM_WARNING_THRESHOLD,
        },
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input_file": metadata.get("input_file"),
            "processing_duration_sec": metadata.get("duration"),
            "image_resolution": metadata.get("resolution"),
            "num_views": 6,
        },
    }

    # Write to file
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Quality report saved to {output_path}")
    logger.info(f"Status: {status}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    return report


def generate_previews(
    mesh_path: Path,
    output_dir: Path,
    input_dir: Path,
    resolution: Optional[Tuple[int, int]] = None
) -> Dict[str, List[Path]]:
    """
    Convenience function to generate preview images.

    Args:
        mesh_path: Path to GLB mesh file
        output_dir: Directory to save preview images
        input_dir: Directory containing transforms_train.json
        resolution: Optional resolution override (height, width)

    Returns:
        Dict with "textured" and "wireframe" path lists.
    """
    from app.services.mesh_renderer import load_camera_poses
    generator = PreviewGenerator()
    camera_poses = load_camera_poses(input_dir)

    if resolution is None:
        resolution = generator._get_input_resolution(input_dir, camera_poses)

    return generator.generate_previews(
        mesh_path=mesh_path,
        output_dir=output_dir,
        camera_poses=camera_poses,
        resolution=resolution,
    )
