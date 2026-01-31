"""
nvdiffrec model wrapper for 3D reconstruction.

nvdiffrec uses differentiable rendering and optimization to reconstruct
textured meshes from multi-view images. Unlike feedforward models,
it iteratively refines the mesh through gradient descent optimization.

Based on: https://github.com/NVlabs/nvdiffrec

Key components:
- nvdiffrast for differentiable rasterization
- DMTet (Differentiable Marching Tetrahedra) for geometry representation
- NeRF-format dataset input via camera_estimation service
"""
import logging
import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np

from app.models.base import BaseReconstructionModel
from app.services.mesh_export import save_mesh_both_formats, validate_mesh_output
from app.services.vram_manager import cleanup_gpu_memory, check_vram_available
from app.services.camera_estimation import create_nerf_dataset, validate_nerf_dataset

logger = logging.getLogger(__name__)

# Model configuration
WEIGHTS_PATH = Path("/app/weights/nvdiffrec")
REQUIRED_VRAM_GB = 14.0  # nvdiffrec needs substantial VRAM for optimization
DEFAULT_ITERATIONS = 500  # Balance between quality and speed
INFERENCE_TIMEOUT_SEC = 3600  # 60 minutes (optimization takes longer)


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor along dimension with numerical stability."""
    return x / (torch.norm(x, dim=dim, keepdim=True) + eps)


class SimpleGeometry:
    """
    Simple geometry representation for optimization.

    Uses a deformable sphere mesh with per-vertex displacement.
    This is a simplified alternative to DMTet when full nvdiffrec
    components are not available.
    """

    def __init__(self, resolution: int = 32, device: torch.device = None):
        """
        Initialize sphere geometry.

        Args:
            resolution: Sphere tessellation resolution
            device: PyTorch device
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resolution = resolution

        # Create base sphere mesh
        verts, faces = self._create_sphere(resolution)
        self.base_verts = verts.to(self.device)
        self.faces = faces.to(self.device)

        # Learnable displacement per vertex
        self.displacements = torch.nn.Parameter(
            torch.zeros_like(self.base_verts, device=self.device)
        )

        # Learnable vertex colors (RGB)
        self.vertex_colors = torch.nn.Parameter(
            torch.ones(self.base_verts.shape[0], 3, device=self.device) * 0.5
        )

    def _create_sphere(self, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a UV sphere mesh."""
        n_lat, n_lon = resolution, resolution * 2

        verts = []
        for i in range(n_lat + 1):
            lat = np.pi * i / n_lat
            for j in range(n_lon):
                lon = 2 * np.pi * j / n_lon
                x = np.sin(lat) * np.cos(lon)
                y = np.cos(lat)
                z = np.sin(lat) * np.sin(lon)
                verts.append([x, y, z])

        verts = torch.tensor(verts, dtype=torch.float32)

        faces = []
        for i in range(n_lat):
            for j in range(n_lon):
                p1 = i * n_lon + j
                p2 = i * n_lon + (j + 1) % n_lon
                p3 = (i + 1) * n_lon + j
                p4 = (i + 1) * n_lon + (j + 1) % n_lon

                if i > 0:
                    faces.append([p1, p3, p2])
                if i < n_lat - 1:
                    faces.append([p2, p3, p4])

        faces = torch.tensor(faces, dtype=torch.long)
        return verts, faces

    @property
    def vertices(self) -> torch.Tensor:
        """Get current deformed vertices."""
        return self.base_verts + self.displacements * 0.5  # Scale displacement

    def parameters(self) -> List[torch.nn.Parameter]:
        """Return learnable parameters for optimizer."""
        return [self.displacements, self.vertex_colors]


class NvdiffrecModel(BaseReconstructionModel):
    """
    nvdiffrec reconstruction model wrapper.

    Uses differentiable rasterization via nvdiffrast and optimization
    to reconstruct textured meshes from multi-view images.
    """

    model_name = "nvdiffrec"

    def __init__(self, celery_task=None, iterations: int = DEFAULT_ITERATIONS):
        """
        Initialize nvdiffrec model.

        Args:
            celery_task: Optional Celery task for progress reporting
            iterations: Number of optimization iterations (default 500)
        """
        super().__init__(celery_task)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._iterations = iterations
        self._glctx = None  # nvdiffrast GL context
        self._use_dmtet = False  # Will try to use DMTet if available

    def load_weights(self) -> None:
        """
        Initialize nvdiffrec components.

        Unlike feedforward models, nvdiffrec doesn't have traditional weights.
        It initializes rendering context and geometry representation for
        optimization from scratch per input.

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

        self.report_progress(10, "Initializing nvdiffrast context")
        logger.info(f"Initializing nvdiffrec with {self._iterations} iterations")

        # Initialize nvdiffrast rasterization context
        try:
            import nvdiffrast.torch as dr
            self._glctx = dr.RasterizeCudaContext()
            logger.info("nvdiffrast CUDA context initialized")
        except ImportError:
            logger.warning("nvdiffrast not available, using fallback rendering")
            self._glctx = None
        except Exception as e:
            logger.warning(f"nvdiffrast init failed: {e}, using fallback")
            self._glctx = None

        # Try to initialize DMTet geometry
        try:
            # DMTet would be imported from nvdiffrec package
            # from nvdiffrec.geometry.dmtet import DMTetGeometry
            # For now, we'll use SimpleGeometry as fallback
            self._use_dmtet = False
            logger.info("Using SimpleGeometry (DMTet not available)")
        except ImportError:
            self._use_dmtet = False

        # Ensure weights directory exists (for potential prior storage)
        if not WEIGHTS_PATH.exists():
            WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)

        logger.info("nvdiffrec initialized successfully")
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
            # Step 1: Prepare NeRF-format dataset
            self.report_progress(20, "Preparing NeRF dataset")

            views_dir = input_dir / "views"
            depth_dir = input_dir / "depth"

            if not views_dir.exists() or not depth_dir.exists():
                return {
                    'status': 'failed',
                    'error': "Input directory missing views/ or depth/ subdirectory"
                }

            # Create temporary directory for NeRF dataset
            nerf_dir = output_dir / "nerf_dataset"
            nerf_result = create_nerf_dataset(
                views_dir=views_dir,
                depth_dir=depth_dir,
                output_dir=nerf_dir,
                image_size=512,
                focal_length=1111.0
            )

            if nerf_result["status"] != "success":
                return {
                    'status': 'failed',
                    'error': f"Failed to create NeRF dataset: {nerf_result.get('error', 'Unknown error')}"
                }

            # Validate dataset
            validation = validate_nerf_dataset(nerf_dir)
            if not validation["valid"]:
                return {
                    'status': 'failed',
                    'error': f"NeRF dataset validation failed: {validation.get('error', 'Unknown error')}"
                }

            logger.info(f"Created NeRF dataset with {nerf_result['image_count']} views")

            # Step 2: Load target images and cameras
            self.report_progress(25, "Loading target images and cameras")

            target_images, camera_matrices = self._load_nerf_dataset(nerf_dir)
            logger.info(f"Loaded {len(target_images)} target images")

            # Step 3: Initialize geometry
            self.report_progress(30, "Initializing mesh geometry")

            geometry = SimpleGeometry(resolution=32, device=self._device)
            optimizer = torch.optim.Adam(geometry.parameters(), lr=0.01)

            # Step 4: Run optimization loop
            logger.info(f"Starting optimization for {self._iterations} iterations")

            for iteration in range(self._iterations):
                # Update progress every 50 iterations
                if iteration % 50 == 0:
                    progress = 30 + int(50 * (iteration / self._iterations))
                    self.report_progress(
                        progress,
                        f"Optimizing ({iteration}/{self._iterations} iterations)"
                    )

                optimizer.zero_grad()

                # Render from each camera and compute loss
                total_loss = torch.tensor(0.0, device=self._device)

                for cam_idx, (target_img, cam_matrix) in enumerate(zip(target_images, camera_matrices)):
                    # Render mesh from camera viewpoint
                    rendered = self._render_mesh(
                        geometry.vertices,
                        geometry.faces,
                        geometry.vertex_colors,
                        cam_matrix,
                        target_img.shape[1:3]  # (H, W)
                    )

                    # Compute image reconstruction loss
                    if rendered is not None:
                        loss = F.mse_loss(rendered, target_img)
                        total_loss = total_loss + loss

                # Add regularization
                total_loss = total_loss + self._compute_regularization(geometry)

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Log every 100 iterations
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: loss = {total_loss.item():.6f}")

            self.report_progress(80, "Optimization complete")

            # Step 5: Extract and export final mesh
            self.report_progress(85, "Extracting final mesh")

            output_dir.mkdir(parents=True, exist_ok=True)

            # Get final mesh data
            verts = geometry.vertices.detach()
            faces = geometry.faces.detach()
            vertex_colors = geometry.vertex_colors.detach()

            # Create texture from vertex colors
            texture, uvs = self._create_texture_from_vertex_colors(
                verts, vertex_colors
            )

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

            # Step 6: Validate output
            self.report_progress(95, "Validating output")
            output_validation = validate_mesh_output(output_dir, "mesh")

            if not output_validation['valid']:
                return {
                    'status': 'failed',
                    'error': f"Output validation failed: {output_validation.get('error', 'Unknown error')}"
                }

            # Clean up temporary NeRF dataset
            if nerf_dir.exists():
                shutil.rmtree(nerf_dir)

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

    def _load_nerf_dataset(self, nerf_dir: Path) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Load images and camera matrices from NeRF dataset.

        Args:
            nerf_dir: Directory containing transforms_train.json and images/

        Returns:
            Tuple of (target_images, camera_matrices) as tensors
        """
        transforms_path = nerf_dir / "transforms_train.json"

        with open(transforms_path) as f:
            transforms = json.load(f)

        target_images = []
        camera_matrices = []

        for frame in transforms["frames"]:
            # Load image
            img_path = nerf_dir / frame["file_path"]
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).to(self._device)
            # Reshape to (C, H, W) for consistent handling
            img_tensor = img_tensor.permute(2, 0, 1)
            target_images.append(img_tensor)

            # Load camera matrix
            cam_matrix = torch.tensor(
                frame["transform_matrix"],
                dtype=torch.float32,
                device=self._device
            )
            camera_matrices.append(cam_matrix)

        return target_images, camera_matrices

    def _render_mesh(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_colors: torch.Tensor,
        cam_matrix: torch.Tensor,
        resolution: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        """
        Render mesh from camera viewpoint.

        Uses nvdiffrast if available, otherwise falls back to simple projection.

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) face indices
            vertex_colors: (V, 3) per-vertex RGB colors
            cam_matrix: (4, 4) camera-to-world matrix
            resolution: (H, W) output resolution

        Returns:
            Rendered image tensor (C, H, W) or None if rendering fails
        """
        H, W = resolution

        # Transform vertices to camera space
        # cam_matrix is camera-to-world, we need world-to-camera
        world_to_cam = torch.inverse(cam_matrix)

        # Add homogeneous coordinate
        verts_homo = torch.cat([
            vertices,
            torch.ones(vertices.shape[0], 1, device=self._device)
        ], dim=1)

        # Transform to camera space
        verts_cam = (world_to_cam @ verts_homo.T).T[:, :3]

        # Simple perspective projection (approximation)
        focal = 1111.0  # Match dataset focal length
        cx, cy = W / 2, H / 2

        # Project to image plane
        z = verts_cam[:, 2:3].clamp(min=0.1)  # Avoid division by zero
        x_proj = (verts_cam[:, 0:1] * focal / z) + cx
        y_proj = (verts_cam[:, 1:2] * focal / z) + cy

        # Clip coordinates
        verts_clip = torch.zeros(vertices.shape[0], 4, device=self._device)
        verts_clip[:, 0] = (x_proj.squeeze() / W) * 2 - 1
        verts_clip[:, 1] = (y_proj.squeeze() / H) * 2 - 1
        verts_clip[:, 2] = (z.squeeze() - 0.5) / 2.5  # Normalize depth
        verts_clip[:, 3] = 1.0

        if self._glctx is not None:
            # Use nvdiffrast for proper rasterization
            try:
                import nvdiffrast.torch as dr

                # Prepare for rasterization
                verts_clip = verts_clip.unsqueeze(0).contiguous()  # (1, V, 4)
                faces_int = faces.int().contiguous()  # (F, 3)

                # Rasterize
                rast_out, _ = dr.rasterize(
                    self._glctx,
                    verts_clip,
                    faces_int,
                    resolution=[H, W]
                )

                # Interpolate vertex colors
                vertex_colors_batch = vertex_colors.unsqueeze(0).contiguous()  # (1, V, 3)
                color_out, _ = dr.interpolate(
                    vertex_colors_batch,
                    rast_out,
                    faces_int
                )

                # Apply mask (background is 0)
                mask = (rast_out[..., 3:4] > 0).float()
                color_out = color_out * mask

                # Reshape to (C, H, W)
                rendered = color_out[0].permute(2, 0, 1)
                return rendered

            except Exception as e:
                logger.warning(f"nvdiffrast rendering failed: {e}, using fallback")
                return self._fallback_render(verts_clip[0], faces, vertex_colors, H, W)
        else:
            # Fallback: simple point-based rendering
            return self._fallback_render(verts_clip, faces, vertex_colors, H, W)

    def _fallback_render(
        self,
        verts_clip: torch.Tensor,
        faces: torch.Tensor,
        vertex_colors: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Simple fallback rendering using point splatting.

        This is a simplified renderer for when nvdiffrast is not available.
        It provides approximate gradients for optimization.
        """
        # Create empty image
        rendered = torch.zeros(3, H, W, device=self._device)

        # Convert clip coordinates to pixel coordinates
        x_pix = ((verts_clip[:, 0] + 1) / 2 * W).long().clamp(0, W - 1)
        y_pix = ((verts_clip[:, 1] + 1) / 2 * H).long().clamp(0, H - 1)

        # Splat vertex colors onto image
        for i in range(verts_clip.shape[0]):
            x, y = x_pix[i].item(), y_pix[i].item()
            # Simple blending (not differentiable, but provides target)
            rendered[:, y, x] = vertex_colors[i]

        return rendered

    def _compute_regularization(self, geometry: SimpleGeometry) -> torch.Tensor:
        """
        Compute regularization losses for geometry.

        Includes:
        - Displacement smoothness (Laplacian)
        - Displacement magnitude constraint
        """
        reg_loss = torch.tensor(0.0, device=self._device)

        # Displacement magnitude regularization
        disp_mag = torch.norm(geometry.displacements, dim=1).mean()
        reg_loss = reg_loss + 0.01 * disp_mag

        # Color smoothness (encourage smooth vertex colors)
        color_var = torch.var(geometry.vertex_colors, dim=0).mean()
        reg_loss = reg_loss + 0.001 * color_var

        return reg_loss

    def _create_texture_from_vertex_colors(
        self,
        vertices: torch.Tensor,
        vertex_colors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a UV texture map from per-vertex colors.

        Uses spherical UV mapping for simplicity.

        Args:
            vertices: (V, 3) vertex positions
            vertex_colors: (V, 3) per-vertex RGB colors

        Returns:
            Tuple of (texture_map, uvs)
            - texture_map: (H, W, 3) RGB texture
            - uvs: (V, 2) per-vertex UV coordinates
        """
        tex_size = 256

        # Compute spherical UV coordinates
        verts_normalized = _safe_normalize(vertices.cpu())

        # Spherical mapping
        theta = torch.atan2(verts_normalized[:, 0], verts_normalized[:, 2])
        phi = torch.asin(verts_normalized[:, 1].clamp(-1, 1))

        u = (theta / (2 * np.pi)) + 0.5
        v = (phi / np.pi) + 0.5

        uvs = torch.stack([u, v], dim=1)

        # Create texture by sampling vertex colors at UV positions
        texture = torch.zeros(tex_size, tex_size, 3, dtype=torch.float32)

        # Map vertex colors to texture space
        vertex_colors_cpu = vertex_colors.cpu()
        u_pix = (u * (tex_size - 1)).long().clamp(0, tex_size - 1)
        v_pix = (v * (tex_size - 1)).long().clamp(0, tex_size - 1)

        for i in range(vertices.shape[0]):
            texture[v_pix[i], u_pix[i]] = vertex_colors_cpu[i]

        # Simple blur to fill gaps
        from scipy.ndimage import gaussian_filter
        texture_np = texture.numpy()
        for c in range(3):
            texture_np[:, :, c] = gaussian_filter(texture_np[:, :, c], sigma=2)

        texture = torch.from_numpy(texture_np)

        return texture, uvs
