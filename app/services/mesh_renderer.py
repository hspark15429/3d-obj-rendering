"""
Mesh rendering service using nvdiffrast.

Provides GPU-accelerated rendering of meshes from specific camera poses
for quality metric computation and preview image generation.

Uses nvdiffrast for differentiable rasterization with proper handling
of OpenGL coordinate conventions (vertical flip required for standard
top-down image ordering).
"""
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


def load_camera_poses(dataset_dir: Path) -> Dict:
    """
    Load camera poses from transforms_train.json.

    Args:
        dataset_dir: Directory containing transforms_train.json

    Returns:
        Dict with:
            - camera_angle_x: float - horizontal FOV in radians
            - frames: list of dicts with transform_matrix and file_path

    Raises:
        FileNotFoundError: If transforms_train.json not found
        json.JSONDecodeError: If JSON parsing fails
    """
    dataset_dir = Path(dataset_dir)
    transforms_path = dataset_dir / "transforms_train.json"

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms_train.json not found in {dataset_dir}")

    with open(transforms_path) as f:
        data = json.load(f)

    frames = []
    for frame in data["frames"]:
        frames.append({
            "transform_matrix": np.array(frame["transform_matrix"], dtype=np.float32),
            "file_path": frame["file_path"]
        })

    logger.debug(f"Loaded {len(frames)} camera poses from {transforms_path}")

    return {
        "camera_angle_x": float(data["camera_angle_x"]),
        "frames": frames
    }


def build_mvp_matrix(
    transform_matrix: np.ndarray,
    camera_angle_x: float,
    resolution: Tuple[int, int],
    near: float = 0.1,
    far: float = 10.0
) -> torch.Tensor:
    """
    Build Model-View-Projection matrix from NeRF camera transform.

    Args:
        transform_matrix: 4x4 camera-to-world transform matrix (NeRF format)
        camera_angle_x: Horizontal field of view in radians
        resolution: Output resolution (height, width)
        near: Near clipping plane
        far: Far clipping plane

    Returns:
        4x4 torch.Tensor MVP matrix
    """
    H, W = resolution

    # Camera-to-world -> World-to-camera (view matrix)
    c2w = np.array(transform_matrix, dtype=np.float32)
    if c2w.shape == (4, 4):
        # Invert camera-to-world to get world-to-camera
        view_matrix = np.linalg.inv(c2w)
    else:
        raise ValueError(f"Expected 4x4 transform matrix, got shape {c2w.shape}")

    # Compute focal length from FOV
    focal = W / (2.0 * math.tan(camera_angle_x / 2.0))

    # Aspect ratio
    aspect = W / H

    # Build perspective projection matrix (OpenGL convention)
    # Using symmetric frustum
    fov_y = 2.0 * math.atan(H / (2.0 * focal))

    t = near * math.tan(fov_y / 2.0)
    r = t * aspect

    proj_matrix = np.zeros((4, 4), dtype=np.float32)
    proj_matrix[0, 0] = near / r
    proj_matrix[1, 1] = near / t
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2.0 * far * near / (far - near)
    proj_matrix[3, 2] = -1.0

    # MVP = Projection @ View @ Model (Model is identity for world space vertices)
    mvp = proj_matrix @ view_matrix

    return torch.from_numpy(mvp.astype(np.float32))


class MeshRenderer:
    """
    GPU-accelerated mesh renderer using nvdiffrast.

    Provides methods for rendering textured meshes, depth maps, and wireframes
    from arbitrary camera poses. All rendered outputs are vertically flipped
    to match standard top-down image ordering (OpenGL uses bottom-up).
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize mesh renderer.

        Args:
            device: PyTorch device ("cuda" or "cpu")

        Raises:
            RuntimeError: If nvdiffrast fails to initialize
        """
        self.device = torch.device(device)
        self._glctx = None

        # Initialize nvdiffrast context
        try:
            import nvdiffrast.torch as dr
            if device == "cuda" and torch.cuda.is_available():
                self._glctx = dr.RasterizeCudaContext()
                logger.info("MeshRenderer: nvdiffrast CUDA context initialized")
            else:
                # Try GL context for CPU fallback
                try:
                    self._glctx = dr.RasterizeGLContext()
                    logger.info("MeshRenderer: nvdiffrast GL context initialized")
                except Exception:
                    logger.warning("MeshRenderer: GL context unavailable, using CUDA context")
                    self._glctx = dr.RasterizeCudaContext()
        except ImportError as e:
            raise RuntimeError(f"nvdiffrast not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize nvdiffrast: {e}")

    def load_mesh(self, glb_path: Path) -> Dict:
        """
        Load mesh from GLB file.

        Args:
            glb_path: Path to GLB file

        Returns:
            Dict with:
                - vertices: torch.Tensor (V, 3) float32
                - faces: torch.Tensor (F, 3) int32
                - uvs: torch.Tensor (V, 2) float32 or None
                - texture: torch.Tensor (H, W, 3) float32 or None
                - has_texture: bool

        Raises:
            FileNotFoundError: If GLB file not found
            ValueError: If mesh cannot be loaded
        """
        glb_path = Path(glb_path)

        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")

        logger.info(f"Loading mesh from {glb_path}")

        # Load with trimesh
        scene = trimesh.load(str(glb_path), force="scene")

        # Get mesh geometry
        if isinstance(scene, trimesh.Scene):
            # Merge all meshes in scene
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError(f"No mesh geometry found in {glb_path}")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            raise ValueError(f"Unexpected trimesh type: {type(scene)}")

        # Extract geometry
        vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(self.device)
        faces = torch.from_numpy(mesh.faces.astype(np.int32)).to(self.device)

        # Try to extract texture and UVs
        uvs = None
        texture = None
        has_texture = False

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uvs = torch.from_numpy(mesh.visual.uv.astype(np.float32)).to(self.device)

            # Try to get texture image
            if hasattr(mesh.visual, "material"):
                material = mesh.visual.material
                if hasattr(material, "image") and material.image is not None:
                    tex_img = np.array(material.image.convert("RGB"), dtype=np.float32) / 255.0
                    texture = torch.from_numpy(tex_img).to(self.device)
                    has_texture = True
                elif hasattr(material, "baseColorTexture") and material.baseColorTexture is not None:
                    tex_img = np.array(material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
                    texture = torch.from_numpy(tex_img).to(self.device)
                    has_texture = True

        # Handle vertex colors as fallback
        vertex_colors = None
        if not has_texture and hasattr(mesh.visual, "vertex_colors"):
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) > 0:
                vertex_colors = torch.from_numpy(
                    vc[:, :3].astype(np.float32) / 255.0
                ).to(self.device)

        logger.info(
            f"Loaded mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces, "
            f"has_texture={has_texture}, has_uvs={uvs is not None}"
        )

        return {
            "vertices": vertices,
            "faces": faces,
            "uvs": uvs,
            "texture": texture,
            "vertex_colors": vertex_colors,
            "has_texture": has_texture
        }

    def render_textured(
        self,
        mesh: Dict,
        mvp: torch.Tensor,
        resolution: Tuple[int, int],
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        Render textured mesh.

        Args:
            mesh: Mesh dict from load_mesh()
            mvp: 4x4 MVP matrix from build_mvp_matrix()
            resolution: Output resolution (height, width)
            bg_color: Background color RGB in [0, 1]

        Returns:
            (H, W, 3) numpy array in [0, 1] float32, vertically flipped
        """
        import nvdiffrast.torch as dr

        H, W = resolution
        mvp = mvp.to(self.device)

        vertices = mesh["vertices"]
        faces = mesh["faces"]

        # Transform vertices to clip space
        # vertices: (V, 3) -> (V, 4) homogeneous
        verts_homo = torch.cat([
            vertices,
            torch.ones(vertices.shape[0], 1, device=self.device)
        ], dim=1)

        # Apply MVP: (V, 4) @ (4, 4).T = (V, 4)
        pos_clip = verts_homo @ mvp.T

        # Add batch dimension for nvdiffrast
        pos_clip = pos_clip.unsqueeze(0).contiguous()  # (1, V, 4)
        faces_int = faces.int().contiguous()  # (F, 3)

        # Rasterize
        rast_out, rast_out_db = dr.rasterize(
            self._glctx,
            pos_clip,
            faces_int,
            resolution=[H, W]
        )

        # Determine color source
        if mesh["has_texture"] and mesh["uvs"] is not None:
            # Interpolate UVs
            uvs = mesh["uvs"].unsqueeze(0).contiguous()  # (1, V, 2)
            texc, texd = dr.interpolate(
                uvs,
                rast_out,
                faces_int,
                rast_db=rast_out_db,
                diff_attrs="all"
            )

            # Sample texture
            texture = mesh["texture"].unsqueeze(0).contiguous()  # (1, H_tex, W_tex, 3)
            color = dr.texture(
                texture,
                texc,
                texd,
                filter_mode="linear-mipmap-linear"
            )
        elif mesh.get("vertex_colors") is not None:
            # Use vertex colors
            vc = mesh["vertex_colors"].unsqueeze(0).contiguous()  # (1, V, 3)
            color, _ = dr.interpolate(vc, rast_out, faces_int)
        else:
            # Solid gray color
            solid_color = torch.full(
                (1, vertices.shape[0], 3), 0.7,
                device=self.device, dtype=torch.float32
            )
            color, _ = dr.interpolate(solid_color, rast_out, faces_int)

        # Apply background color using mask
        mask = (rast_out[..., 3:4] > 0).float()
        bg = torch.tensor(bg_color, device=self.device).view(1, 1, 1, 3)
        color = color * mask + bg * (1.0 - mask)

        # Remove batch dimension and convert to numpy
        rendered = color[0].cpu().numpy()

        # CRITICAL: Flip vertically (OpenGL uses bottom-up ordering)
        rendered = rendered[::-1, :, :]

        logger.debug(f"Rendered textured image: {rendered.shape}")
        return rendered.astype(np.float32)

    def render_depth(
        self,
        mesh: Dict,
        mvp: torch.Tensor,
        resolution: Tuple[int, int]
    ) -> np.ndarray:
        """
        Render depth map from mesh.

        Args:
            mesh: Mesh dict from load_mesh()
            mvp: 4x4 MVP matrix from build_mvp_matrix()
            resolution: Output resolution (height, width)

        Returns:
            (H, W) numpy array with depth values, vertically flipped
            Background pixels have depth 0
        """
        import nvdiffrast.torch as dr

        H, W = resolution
        mvp = mvp.to(self.device)

        vertices = mesh["vertices"]
        faces = mesh["faces"]

        # Transform vertices to clip space
        verts_homo = torch.cat([
            vertices,
            torch.ones(vertices.shape[0], 1, device=self.device)
        ], dim=1)

        pos_clip = verts_homo @ mvp.T
        pos_clip = pos_clip.unsqueeze(0).contiguous()
        faces_int = faces.int().contiguous()

        # Rasterize
        rast_out, _ = dr.rasterize(
            self._glctx,
            pos_clip,
            faces_int,
            resolution=[H, W]
        )

        # Extract depth from rasterization (z-buffer is in clip space)
        # rast_out[..., 2] contains interpolated depth values
        depth = rast_out[0, :, :, 2].cpu().numpy()

        # Set background to 0 (where triangle ID is 0)
        mask = rast_out[0, :, :, 3].cpu().numpy() > 0
        depth = depth * mask

        # CRITICAL: Flip vertically
        depth = depth[::-1, :]

        logger.debug(f"Rendered depth map: {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")
        return depth.astype(np.float32)

    def render_wireframe(
        self,
        mesh: Dict,
        mvp: torch.Tensor,
        resolution: Tuple[int, int],
        line_color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Render wireframe overlay on solid mesh.

        Uses filled mesh with edge highlighting for wireframe effect.

        Args:
            mesh: Mesh dict from load_mesh()
            mvp: 4x4 MVP matrix from build_mvp_matrix()
            resolution: Output resolution (height, width)
            line_color: Edge color RGB in [0, 255]
            bg_color: Background color RGB in [0, 255]

        Returns:
            (H, W, 3) numpy array in [0, 255] uint8, vertically flipped
        """
        import nvdiffrast.torch as dr

        H, W = resolution
        mvp = mvp.to(self.device)

        vertices = mesh["vertices"]
        faces = mesh["faces"]

        # Transform vertices to clip space
        verts_homo = torch.cat([
            vertices,
            torch.ones(vertices.shape[0], 1, device=self.device)
        ], dim=1)

        pos_clip = verts_homo @ mvp.T
        pos_clip_batch = pos_clip.unsqueeze(0).contiguous()
        faces_int = faces.int().contiguous()

        # Rasterize solid mesh with light gray color
        rast_out, _ = dr.rasterize(
            self._glctx,
            pos_clip_batch,
            faces_int,
            resolution=[H, W]
        )

        # Create solid fill color
        fill_color = torch.full(
            (1, vertices.shape[0], 3), 0.9,
            device=self.device, dtype=torch.float32
        )
        color, _ = dr.interpolate(fill_color, rast_out, faces_int)

        # Get mask
        mask = (rast_out[..., 3:4] > 0).float()

        # Apply background
        bg = torch.tensor(
            [c / 255.0 for c in bg_color],
            device=self.device
        ).view(1, 1, 1, 3)
        color = color * mask + bg * (1.0 - mask)

        # Convert to numpy
        rendered = color[0].cpu().numpy()

        # CRITICAL: Flip vertically
        rendered = rendered[::-1, :, :]

        # Draw edges using projected vertices
        rendered_uint8 = (rendered * 255).astype(np.uint8).copy()

        # Project vertices to screen space
        pos_ndc = pos_clip[:, :3] / pos_clip[:, 3:4]
        x_screen = ((pos_ndc[:, 0] + 1) / 2 * W).cpu().numpy()
        y_screen = ((1 - (pos_ndc[:, 1] + 1) / 2) * H).cpu().numpy()  # Flip Y for screen coords

        # Extract unique edges
        faces_np = faces.cpu().numpy()
        edges = set()
        for f in faces_np:
            for i in range(3):
                e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                edges.add(e)

        # Draw edges using PIL
        from PIL import ImageDraw
        img = Image.fromarray(rendered_uint8)
        draw = ImageDraw.Draw(img)

        for e0, e1 in edges:
            x0, y0 = x_screen[e0], y_screen[e0]
            x1, y1 = x_screen[e1], y_screen[e1]

            # Only draw if both vertices are in front of camera
            if pos_clip[e0, 3] > 0 and pos_clip[e1, 3] > 0:
                # Clip to image bounds
                if (0 <= x0 < W and 0 <= y0 < H) or (0 <= x1 < W and 0 <= y1 < H):
                    draw.line([(x0, y0), (x1, y1)], fill=line_color, width=1)

        rendered_uint8 = np.array(img)

        logger.debug(f"Rendered wireframe: {rendered_uint8.shape}")
        return rendered_uint8


def render_mesh(
    glb_path: Union[str, Path],
    transform_matrix: np.ndarray,
    camera_angle_x: float,
    resolution: Tuple[int, int],
    device: str = "cuda"
) -> np.ndarray:
    """
    Convenience function to render a mesh from a single camera pose.

    Args:
        glb_path: Path to GLB mesh file
        transform_matrix: 4x4 camera-to-world matrix
        camera_angle_x: Horizontal FOV in radians
        resolution: Output resolution (height, width)
        device: PyTorch device

    Returns:
        (H, W, 3) numpy array in [0, 1] float32
    """
    renderer = MeshRenderer(device=device)
    mesh = renderer.load_mesh(Path(glb_path))
    mvp = build_mvp_matrix(transform_matrix, camera_angle_x, resolution)
    return renderer.render_textured(mesh, mvp, resolution)


def render_depth(
    glb_path: Union[str, Path],
    transform_matrix: np.ndarray,
    camera_angle_x: float,
    resolution: Tuple[int, int],
    device: str = "cuda"
) -> np.ndarray:
    """
    Convenience function to render a depth map from a single camera pose.

    Args:
        glb_path: Path to GLB mesh file
        transform_matrix: 4x4 camera-to-world matrix
        camera_angle_x: Horizontal FOV in radians
        resolution: Output resolution (height, width)
        device: PyTorch device

    Returns:
        (H, W) numpy array with depth values
    """
    renderer = MeshRenderer(device=device)
    mesh = renderer.load_mesh(Path(glb_path))
    mvp = build_mvp_matrix(transform_matrix, camera_angle_x, resolution)
    return renderer.render_depth(mesh, mvp, resolution)
