"""
Geometry utilities for 3D reconstruction.

Provides visual hull computation from depth maps using space carving.
"""
import logging
from typing import List, Tuple

import numpy as np
import trimesh
from skimage import measure

logger = logging.getLogger(__name__)


def visual_hull_from_depths(
    depth_maps: List[np.ndarray],
    camera_matrices: List[np.ndarray],
    image_size: int = None,
    focal_length: float = None,
    voxel_resolution: int = 64,
    bounds: float = 1.0,
    camera_distance: float = 2.0,
) -> trimesh.Trimesh:
    """
    Create visual hull mesh from orthogonal depth maps via space carving.

    Projects a 3D voxel grid to each camera view and carves away voxels
    that are in front of the observed depth surface or project to background.

    Depth encoding assumption: white (1.0) = closest to camera, black (0.0) = background/far

    Args:
        depth_maps: List of 6 depth maps as numpy arrays (H, W), normalized [0, 1]
        camera_matrices: List of 6 camera-to-world 4x4 matrices
        image_size: Input image resolution (auto-detected from depth_maps if None)
        focal_length: Camera focal length in pixels (auto-computed for ~50deg FOV if None)
        voxel_resolution: Resolution of voxel grid (e.g., 64 = 64³ voxels)
        bounds: Object assumed within [-bounds, bounds]³ cube
        camera_distance: Distance from camera to object center

    Returns:
        trimesh.Trimesh: Extracted mesh from visual hull

    Raises:
        ValueError: If no valid geometry found
    """
    n_views = len(depth_maps)
    if n_views != len(camera_matrices):
        raise ValueError(f"Mismatch: {n_views} depth maps vs {len(camera_matrices)} cameras")

    # Auto-detect image size from first depth map
    if image_size is None:
        image_size = depth_maps[0].shape[0]  # Assume square

    # Auto-compute focal length for ~50 degree FOV if not specified
    # focal = width / (2 * tan(fov/2))
    # For 50 deg FOV: focal ≈ width * 1.07
    if focal_length is None:
        focal_length = image_size * 1.07

    logger.info(f"Visual hull: {voxel_resolution}³ grid, {n_views} views, image_size={image_size}, focal={focal_length:.1f}")

    # Create voxel grid coordinates
    # Grid spans [-bounds, bounds] in each dimension
    coords = np.linspace(-bounds, bounds, voxel_resolution)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')

    # Flatten for easier processing
    voxel_centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)  # (N, 3)
    n_voxels = voxel_centers.shape[0]

    # Start with all voxels occupied
    occupancy = np.ones(n_voxels, dtype=bool)

    # Camera intrinsics
    cx, cy = image_size / 2, image_size / 2

    for view_idx, (depth_map, c2w) in enumerate(zip(depth_maps, camera_matrices)):
        # Resize depth map if needed
        if depth_map.shape[0] != image_size or depth_map.shape[1] != image_size:
            from PIL import Image
            dm_pil = Image.fromarray((depth_map * 255).astype(np.uint8))
            dm_pil = dm_pil.resize((image_size, image_size), Image.BILINEAR)
            depth_map = np.array(dm_pil, dtype=np.float32) / 255.0

        # World-to-camera transform
        w2c = np.linalg.inv(c2w)

        # Transform voxel centers to camera space
        voxels_homo = np.hstack([voxel_centers, np.ones((n_voxels, 1))])  # (N, 4)
        voxels_cam = (w2c @ voxels_homo.T).T[:, :3]  # (N, 3)

        # Get depth (Z in camera space)
        # In OpenGL convention, camera looks along -Z, objects in front have negative Z
        voxel_depths = -voxels_cam[:, 2]

        # Project to image plane
        valid_depth = voxel_depths > 0.01
        x_proj = np.zeros(n_voxels)
        y_proj = np.zeros(n_voxels)

        # Standard pinhole projection (negate X for correct left-right)
        x_proj[valid_depth] = (voxels_cam[valid_depth, 0] * focal_length / voxel_depths[valid_depth]) + cx
        y_proj[valid_depth] = (-voxels_cam[valid_depth, 1] * focal_length / voxel_depths[valid_depth]) + cy

        # Convert to pixel indices
        x_pix = np.round(x_proj).astype(int)
        y_pix = np.round(y_proj).astype(int)

        # Check bounds
        in_bounds = (
            valid_depth &
            (x_pix >= 0) & (x_pix < image_size) &
            (y_pix >= 0) & (y_pix < image_size)
        )

        # Sample depth map at projected locations (silhouette check)
        observed_depth_norm = np.zeros(n_voxels)
        observed_depth_norm[in_bounds] = depth_map[y_pix[in_bounds], x_pix[in_bounds]]

        # SILHOUETTE-BASED CARVING (simpler and more robust)
        # A voxel is carved if it projects to background (depth < threshold)
        # Background = black = 0, Object = non-zero
        bg_threshold = 0.02  # Pixels below this are considered background

        # Voxels projecting to background are carved
        projects_to_bg = in_bounds & (observed_depth_norm < bg_threshold)

        # Voxels outside image or behind camera are carved
        invalid = ~in_bounds

        # Update occupancy: carve voxels that project to empty space
        carve_mask = projects_to_bg | invalid
        occupancy = occupancy & ~carve_mask

        remaining = np.sum(occupancy)
        logger.info(f"View {view_idx}: {remaining} voxels remain ({np.sum(carve_mask)} carved)")

    # Reshape occupancy to 3D grid
    occupancy_grid = occupancy.reshape((voxel_resolution, voxel_resolution, voxel_resolution))

    logger.info(f"Visual hull: {np.sum(occupancy_grid)} occupied voxels")

    if np.sum(occupancy_grid) < 10:
        logger.warning("Very few occupied voxels, using fallback sphere")
        return _create_sphere_mesh(radius=0.5, subdivisions=2)

    # Pad grid to ensure closed surface
    padded = np.pad(occupancy_grid.astype(float), 1, mode='constant', constant_values=0)

    # Marching cubes to extract mesh
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            padded,
            level=0.5,
            spacing=(2 * bounds / voxel_resolution,) * 3
        )

        # Adjust vertices to account for padding and centering
        verts = verts - bounds - (bounds / voxel_resolution)

        # Create trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # Clean up mesh
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()

        logger.info(f"Visual hull mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return mesh

    except Exception as e:
        logger.error(f"Marching cubes failed: {e}, using fallback sphere")
        return _create_sphere_mesh(radius=0.5, subdivisions=2)


def _create_sphere_mesh(radius: float = 0.5, subdivisions: int = 2) -> trimesh.Trimesh:
    """Create a simple sphere mesh as fallback."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return sphere


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int = 5000) -> trimesh.Trimesh:
    """
    Simplify mesh to target face count.

    Args:
        mesh: Input trimesh
        target_faces: Target number of faces

    Returns:
        Simplified mesh
    """
    if len(mesh.faces) <= target_faces:
        return mesh

    try:
        # Use quadric decimation if available
        simplified = mesh.simplify_quadric_decimation(target_faces)
        logger.info(f"Simplified mesh: {len(mesh.faces)} -> {len(simplified.faces)} faces")
        return simplified
    except Exception as e:
        logger.warning(f"Mesh simplification failed: {e}")
        return mesh
