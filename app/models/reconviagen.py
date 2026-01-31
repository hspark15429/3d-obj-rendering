"""
ReconViaGen model wrapper for 3D reconstruction using TRELLIS-VGGT.

Uses TrellisPipelineWrapper to run TRELLIS-VGGT inference and exports
textured meshes via postprocessing_utils.to_glb().

Based on: https://github.com/estheryang11/ReconViaGen
Reference: https://huggingface.co/spaces/Stable-X/ReconViaGen/blob/main/app.py
"""
import os
import logging
from pathlib import Path
from typing import Optional, List

# CRITICAL: Set SPCONV_ALGO before any trellis imports
# This must be done before spconv is imported or it will hang/crash
os.environ['SPCONV_ALGO'] = 'native'

import torch
from PIL import Image

from app.models.base import BaseReconstructionModel
from app.models.trellis import TrellisPipelineWrapper
from app.services.vram_manager import cleanup_gpu_memory, check_vram_available

logger = logging.getLogger(__name__)

# Model configuration
WEIGHTS_PATH = Path("/app/weights/reconviagen")
REQUIRED_VRAM_GB = 16.0  # TRELLIS needs ~14-16GB VRAM
INFERENCE_TIMEOUT_SEC = 1800  # 30 minutes


class ReconViaGenModel(BaseReconstructionModel):
    """
    ReconViaGen reconstruction model using TRELLIS-VGGT pipeline.

    Converts multi-view RGB images to textured 3D mesh via GLB export.
    Uses TrellisPipelineWrapper for model loading and inference.
    """

    model_name = "reconviagen"

    def __init__(self, celery_task=None):
        super().__init__(celery_task)
        self._pipeline: Optional[TrellisPipelineWrapper] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_weights(self) -> None:
        """
        Load TRELLIS-VGGT model weights via TrellisPipelineWrapper.

        Raises:
            RuntimeError: If insufficient VRAM available
        """
        self.report_progress(5, "Checking VRAM availability")

        # Check VRAM before loading
        vram_status = check_vram_available(REQUIRED_VRAM_GB)
        if not vram_status['available']:
            raise RuntimeError(
                f"Insufficient VRAM: {vram_status['free_gb']:.1f}GB available, "
                f"{REQUIRED_VRAM_GB}GB required"
            )

        self.report_progress(10, "Loading TRELLIS model weights")
        logger.info("Initializing TrellisPipelineWrapper")

        # Initialize and load pipeline
        self._pipeline = TrellisPipelineWrapper(device="cuda")
        self._pipeline.load()

        logger.info("TRELLIS-VGGT model loaded successfully")
        self.report_progress(15, "Model loaded")

    def _load_images(self, views_dir: Path) -> List[Image.Image]:
        """
        Load view images as PIL Images.

        Args:
            views_dir: Directory containing view_00.png...view_N.png

        Returns:
            List of PIL Images resized to 512px height
        """
        view_files = sorted(views_dir.glob("view_*.png"))
        images = []

        for f in view_files:
            img = Image.open(f).convert("RGB")
            # Resize to 512px height maintaining aspect ratio (TRELLIS expects this)
            if img.height != 512:
                ratio = 512 / img.height
                new_size = (int(img.width * ratio), 512)
                img = img.resize(new_size, Image.LANCZOS)
            images.append(img)

        logger.info(f"Loaded {len(images)} images from {views_dir}")
        return images

    def inference(self, input_dir: Path, output_dir: Path) -> dict:
        """
        Run TRELLIS-VGGT inference on input images.

        Args:
            input_dir: Directory containing views/ subdirectory
                       with view_00.png...view_N.png
            output_dir: Directory to write mesh.glb, mesh.obj, mesh.ply

        Returns:
            dict with:
                - status: 'success' or 'failed'
                - error: Error message if failed
                - mesh_path: Path to GLB file if success
                - obj_path: Path to OBJ file if success
                - ply_path: Path to PLY file if success
        """
        # Import postprocessing_utils here to avoid import at module level
        from trellis.utils import postprocessing_utils

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        try:
            # Step 1: Load input images
            self.report_progress(20, "Loading input images")
            views_dir = input_dir / "views"

            if not views_dir.exists():
                return {
                    'status': 'failed',
                    'error': "Input directory missing views/ subdirectory"
                }

            images = self._load_images(views_dir)
            if len(images) == 0:
                return {
                    'status': 'failed',
                    'error': "No view images found in views/ directory"
                }

            logger.info(f"Processing {len(images)} input images")

            # Step 2: Run TRELLIS reconstruction
            self.report_progress(30, "Running 3D reconstruction")

            if self._pipeline is None or not self._pipeline.is_loaded():
                return {
                    'status': 'failed',
                    'error': "Model not loaded. Call load_weights() first."
                }

            outputs, _, _ = self._pipeline.run(
                images=images,
                seed=42,
                sparse_steps=30,
                sparse_cfg=7.5,
                slat_steps=12,
                slat_cfg=3.0,
            )

            self.report_progress(60, "Processing reconstruction output")

            # Get Gaussian and mesh outputs
            gs = outputs['gaussian'][0]
            mesh = outputs['mesh'][0]

            # Step 3: Export mesh using official to_glb pattern
            self.report_progress(70, "Exporting mesh to GLB")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Export to GLB with texture (official Stable-X pattern)
            glb = postprocessing_utils.to_glb(
                gs,
                mesh,
                simplify=0.95,       # Mesh simplification ratio
                texture_size=1024    # Texture resolution
            )

            glb_path = output_dir / "mesh.glb"
            glb.export(str(glb_path))
            logger.info(f"Exported GLB to {glb_path}")

            # Step 4: Validate GLB file size (>1KB means real geometry)
            glb_size = glb_path.stat().st_size
            if glb_size < 1024:
                return {
                    'status': 'failed',
                    'error': f"GLB file too small ({glb_size} bytes), may be invalid"
                }
            logger.info(f"GLB file size: {glb_size / 1024:.1f} KB")

            # Step 5: Convert GLB to OBJ/PLY for compatibility
            self.report_progress(80, "Converting to OBJ/PLY formats")

            obj_path = output_dir / "mesh.obj"
            ply_path = output_dir / "mesh.ply"

            try:
                import trimesh
                # Load GLB and export to other formats
                scene = trimesh.load(str(glb_path))
                # Handle both single mesh and scene
                if isinstance(scene, trimesh.Scene):
                    # Combine all meshes in scene
                    combined = trimesh.util.concatenate(
                        [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
                    )
                else:
                    combined = scene

                # Export OBJ
                combined.export(str(obj_path))
                logger.info(f"Exported OBJ to {obj_path}")

                # Export PLY
                combined.export(str(ply_path))
                logger.info(f"Exported PLY to {ply_path}")

            except Exception as e:
                logger.warning(f"GLB conversion failed, GLB is still available: {e}")
                # GLB is the primary output, conversion failure is non-fatal
                obj_path = None
                ply_path = None

            # Step 6: Validate mesh vertex count
            self.report_progress(90, "Validating mesh output")
            try:
                import trimesh
                mesh_check = trimesh.load(str(glb_path))
                if isinstance(mesh_check, trimesh.Scene):
                    total_verts = sum(
                        g.vertices.shape[0] for g in mesh_check.geometry.values()
                        if hasattr(g, 'vertices')
                    )
                else:
                    total_verts = mesh_check.vertices.shape[0]

                if total_verts < 1000:
                    logger.warning(f"Mesh has only {total_verts} vertices, may be low quality")
                else:
                    logger.info(f"Mesh has {total_verts} vertices")

            except Exception as e:
                logger.warning(f"Could not verify vertex count: {e}")

            self.report_progress(95, "Cleanup")

            return {
                'status': 'success',
                'mesh_path': str(glb_path),
                'obj_path': str(obj_path) if obj_path else None,
                'ply_path': str(ply_path) if ply_path else None,
                'glb_size_kb': glb_size / 1024
            }

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during ReconViaGen inference: {e}")
            cleanup_gpu_memory()
            return {
                'status': 'failed',
                'error': "Out of GPU memory. TRELLIS requires ~16GB VRAM."
            }
        except Exception as e:
            logger.error(f"ReconViaGen inference failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': f"Model failed to process images: {str(e)}"
            }

    def cleanup(self) -> None:
        """Release GPU memory by unloading the TRELLIS pipeline."""
        if self._pipeline is not None:
            self._pipeline.cleanup()
            self._pipeline = None

        # Call parent cleanup for additional memory release
        super().cleanup()
        logger.info("ReconViaGen model resources released")
