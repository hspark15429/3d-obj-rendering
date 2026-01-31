"""
nvdiffrec model wrapper for 3D reconstruction.

nvdiffrec uses differentiable rendering and optimization to reconstruct
textured meshes from multi-view images. Unlike feedforward models,
it iteratively refines the mesh (default: 1000 iterations).

Based on: https://github.com/NVlabs/nvdiffrec
"""
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from app.models.base import BaseReconstructionModel
from app.services.mesh_export import save_mesh_both_formats, validate_mesh_output
from app.services.vram_manager import cleanup_gpu_memory, check_vram_available

logger = logging.getLogger(__name__)

# Model configuration
WEIGHTS_PATH = Path("/app/weights/nvdiffrec")
REQUIRED_VRAM_GB = 14.0  # nvdiffrec needs more VRAM for optimization
DEFAULT_ITERATIONS = 1000  # Balance between quality and speed
INFERENCE_TIMEOUT_SEC = 3600  # 60 minutes (optimization takes longer)


class NvdiffrecModel(BaseReconstructionModel):
    """
    nvdiffrec reconstruction model wrapper.

    Uses differentiable marching tetrahedra (DMTet) and nvdiffrast
    for optimization-based mesh reconstruction with textures.
    """

    model_name = "nvdiffrec"

    def __init__(self, celery_task=None, iterations: int = DEFAULT_ITERATIONS):
        super().__init__(celery_task)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._iterations = iterations

    def load_weights(self) -> None:
        """
        Initialize nvdiffrec components.

        Unlike feedforward models, nvdiffrec doesn't have traditional weights.
        It initializes a DMTet grid and optimizes from scratch per input.
        However, we can use pre-computed priors for faster convergence.

        Raises:
            RuntimeError: If insufficient VRAM available
        """
        self.report_progress(5, "Checking VRAM availability")

        # Check VRAM before initialization
        vram_status = check_vram_available(REQUIRED_VRAM_GB)
        if not vram_status['available']:
            raise RuntimeError(
                f"Insufficient VRAM: {vram_status['free_gb']:.1f}GB available, "
                f"{REQUIRED_VRAM_GB}GB required"
            )

        self.report_progress(10, "Initializing nvdiffrec")
        logger.info(f"Initializing nvdiffrec with {self._iterations} iterations")

        # STUB: In real implementation, initialize nvdiffrec components
        # import nvdiffrast.torch as dr
        # self._glctx = dr.RasterizeGLContext()
        # self._dmtet = DMTetGeometry(...)

        # Verify weights directory
        if not WEIGHTS_PATH.exists():
            logger.warning(f"Weights directory not found: {WEIGHTS_PATH}")
            WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)

        logger.info("nvdiffrec initialized (STUB)")
        self.report_progress(15, "Initialization complete")

    def inference(self, input_dir: Path, output_dir: Path) -> dict:
        """
        Run nvdiffrec optimization on input images.

        This is NOT traditional inference - nvdiffrec optimizes a mesh
        from scratch for each input using differentiable rendering.

        Args:
            input_dir: Directory containing views/ and depth/ subdirectories
            output_dir: Directory to write mesh.obj, mesh.ply, texture.png

        Returns:
            dict with status, paths, or error
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        try:
            # Step 1: Load input images
            self.report_progress(20, "Loading input images")
            views_dir = input_dir / "views"
            depth_dir = input_dir / "depth"

            if not views_dir.exists() or not depth_dir.exists():
                return {
                    'status': 'failed',
                    'error': "Input directory missing views/ or depth/ subdirectory"
                }

            view_files = sorted(views_dir.glob("view_*.png"))
            depth_files = sorted(depth_dir.glob("depth_*.png"))

            if len(view_files) != 6 or len(depth_files) != 6:
                return {
                    'status': 'failed',
                    'error': f"Expected 6 view and 6 depth files, got {len(view_files)} views and {len(depth_files)} depth"
                }

            logger.info(f"Found {len(view_files)} views and {len(depth_files)} depth images")

            # Step 2: Initialize optimization
            self.report_progress(25, "Initializing mesh optimization")

            # STUB: In real implementation:
            # target_images = self._load_images(view_files)
            # cameras = self._setup_cameras(depth_files)
            # mesh = self._init_mesh()
            # optimizer = torch.optim.Adam(mesh.parameters(), lr=0.01)

            # Step 3: Run optimization loop
            self.report_progress(30, f"Optimizing (0/{self._iterations} iterations)")

            # STUB: Simulate optimization with progress updates
            for i in range(0, self._iterations, 100):
                # Progress from 30% to 80% during optimization
                progress = 30 + int(50 * (i / self._iterations))
                self.report_progress(progress, f"Optimizing ({i}/{self._iterations} iterations)")

                # STUB: In real implementation:
                # optimizer.zero_grad()
                # rendered = self._render(mesh, cameras)
                # loss = self._compute_loss(rendered, target_images)
                # loss.backward()
                # optimizer.step()

                # Simulate some work (remove in real implementation)
                import time
                time.sleep(0.02)  # 20ms per 100 iterations = 2s total for 1000 iterations

            self.report_progress(80, "Optimization complete")

            # Step 4: Extract and export mesh
            self.report_progress(85, "Extracting final mesh")

            output_dir.mkdir(parents=True, exist_ok=True)

            # STUB: Create placeholder mesh
            verts, faces, texture, uvs = self._create_placeholder_mesh()

            self.report_progress(90, "Exporting mesh files")

            # Export using mesh_export service
            export_result = save_mesh_both_formats(
                verts=verts,
                faces=faces,
                texture_map=texture,
                verts_uvs=uvs,
                output_dir=output_dir,
                mesh_name="mesh"
            )

            # Step 5: Validate output
            self.report_progress(95, "Validating output")
            validation = validate_mesh_output(output_dir, "mesh")

            if not validation['valid']:
                return {
                    'status': 'failed',
                    'error': f"Output validation failed: {validation.get('error', 'Unknown error')}"
                }

            return {
                'status': 'success',
                'mesh_path': export_result['obj_path'],
                'ply_path': export_result['ply_path'],
                'texture_path': export_result.get('texture_path'),
                'iterations': self._iterations
            }

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during nvdiffrec optimization: {e}")
            cleanup_gpu_memory()
            return {
                'status': 'failed',
                'error': "Out of GPU memory. Try reducing resolution or iteration count."
            }
        except Exception as e:
            logger.error(f"nvdiffrec optimization failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': f"Model failed to process images: {str(e)}"
            }

    def _create_placeholder_mesh(self):
        """Create a simple sphere mesh for testing (STUB)."""
        # Create a UV sphere as placeholder (more interesting than cube)
        n_lat, n_lon = 16, 32

        verts = []
        for i in range(n_lat + 1):
            lat = np.pi * i / n_lat
            for j in range(n_lon):
                lon = 2 * np.pi * j / n_lon
                x = np.sin(lat) * np.cos(lon)
                y = np.cos(lat)
                z = np.sin(lat) * np.sin(lon)
                verts.append([x * 0.5, y * 0.5, z * 0.5])

        verts = torch.tensor(verts, dtype=torch.float32)

        # Generate faces
        faces = []
        for i in range(n_lat):
            for j in range(n_lon):
                p1 = i * n_lon + j
                p2 = i * n_lon + (j + 1) % n_lon
                p3 = (i + 1) * n_lon + j
                p4 = (i + 1) * n_lon + (j + 1) % n_lon
                faces.append([p1, p3, p2])
                faces.append([p2, p3, p4])

        faces = torch.tensor(faces, dtype=torch.long)

        # Texture (different color than ReconViaGen for distinction)
        texture = torch.zeros((256, 256, 3), dtype=torch.float32)
        texture[:, :, 0] = 0.3  # Red
        texture[:, :, 1] = 0.5  # Green
        texture[:, :, 2] = 0.7  # Blue

        # UV coordinates (spherical mapping)
        uvs = []
        for i in range(n_lat + 1):
            for j in range(n_lon):
                u = j / n_lon
                v = i / n_lat
                uvs.append([u, v])

        uvs = torch.tensor(uvs, dtype=torch.float32)

        return verts, faces, texture, uvs
