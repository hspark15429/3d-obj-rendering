"""
Mesh export service using trimesh.

Exports meshes in both OBJ (with MTL and texture) and PLY formats.
Provides validation utilities for mesh output files.
"""
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def save_mesh_both_formats(
    verts: torch.Tensor,
    faces: torch.Tensor,
    texture_map: Optional[torch.Tensor],
    verts_uvs: Optional[torch.Tensor],
    output_dir: Path,
    mesh_name: str = "mesh"
) -> dict:
    """
    Save mesh in both OBJ and PLY formats with texture.

    Args:
        verts: FloatTensor (V, 3) - vertex positions
        faces: LongTensor (F, 3) - face indices (0-indexed)
        texture_map: Optional FloatTensor (H, W, 3) in [0, 1] - RGB texture
        verts_uvs: Optional FloatTensor (V, 2) - UV coordinates per vertex
        output_dir: Path to output directory
        mesh_name: Base name for output files (default: "mesh")

    Returns:
        dict with 'obj_path', 'ply_path', 'texture_path' (if texture provided)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}

    # Ensure tensors are on CPU and convert to numpy
    verts_np = verts.cpu().numpy() if verts.is_cuda else verts.numpy()
    faces_np = faces.cpu().numpy() if faces.is_cuda else faces.numpy()

    obj_path = output_dir / f"{mesh_name}.obj"
    ply_path = output_dir / f"{mesh_name}.ply"

    # Create base trimesh
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)

    # Save OBJ with texture if provided
    if texture_map is not None and verts_uvs is not None:
        texture_map_np = texture_map.cpu().numpy() if texture_map.is_cuda else texture_map.numpy()
        verts_uvs_np = verts_uvs.cpu().numpy() if verts_uvs.is_cuda else verts_uvs.numpy()

        # Save texture image
        texture_path = output_dir / f"{mesh_name}.png"
        texture_img = (texture_map_np * 255).astype(np.uint8)
        Image.fromarray(texture_img).save(str(texture_path))
        result['texture_path'] = str(texture_path)

        # Create material with texture
        material = trimesh.visual.material.SimpleMaterial(
            image=Image.fromarray(texture_img)
        )

        # Create TextureVisuals with UVs
        visuals = trimesh.visual.TextureVisuals(
            uv=verts_uvs_np,
            material=material
        )
        mesh.visual = visuals

        # Export OBJ with MTL
        mesh.export(str(obj_path), file_type='obj')
        logger.info(f"Saved OBJ with texture: {obj_path}")
    else:
        # Save OBJ without texture
        mesh.export(str(obj_path), file_type='obj')
        logger.info(f"Saved OBJ without texture: {obj_path}")

    result['obj_path'] = str(obj_path)

    # Save PLY (binary format for smaller files)
    mesh.export(str(ply_path), file_type='ply')
    result['ply_path'] = str(ply_path)

    logger.info(f"Saved PLY: {ply_path}")

    return result


def save_texture_image(
    texture: torch.Tensor,
    output_path: Path
) -> str:
    """
    Save texture tensor as PNG image.

    Args:
        texture: FloatTensor (H, W, 3) in [0, 1] range
        output_path: Path for output PNG

    Returns:
        str: Path to saved texture
    """
    output_path = Path(output_path)

    # Convert to numpy and scale to 0-255
    texture = texture.cpu() if texture.is_cuda else texture
    texture_np = (texture.numpy() * 255).astype(np.uint8)

    # Save as PNG
    Image.fromarray(texture_np).save(str(output_path))
    logger.info(f"Saved texture: {output_path}")

    return str(output_path)


def validate_mesh_output(output_dir: Path, mesh_name: str = "mesh") -> dict:
    """
    Validate mesh output files exist and are non-empty.

    Args:
        output_dir: Directory containing mesh files
        mesh_name: Base name of mesh files

    Returns:
        dict with:
            'valid': bool - True if all required files present and non-empty
            'files': dict of filename -> {'path': str, 'size_bytes': int}
            'error': str - Error message if invalid
    """
    output_dir = Path(output_dir)

    required_files = [
        (f"{mesh_name}.obj", "OBJ mesh"),
        (f"{mesh_name}.ply", "PLY mesh"),
    ]

    optional_files = [
        (f"{mesh_name}.mtl", "Material file"),
        (f"{mesh_name}.png", "Texture image"),
    ]

    files = {}
    errors = []

    # Check required files
    for filename, description in required_files:
        file_path = output_dir / filename
        if not file_path.exists():
            errors.append(f"Missing {description}: {filename}")
        elif file_path.stat().st_size == 0:
            errors.append(f"Empty {description}: {filename}")
        else:
            files[filename] = {
                'path': str(file_path),
                'size_bytes': file_path.stat().st_size
            }

    # Check optional files (just note if present)
    for filename, description in optional_files:
        file_path = output_dir / filename
        if file_path.exists() and file_path.stat().st_size > 0:
            files[filename] = {
                'path': str(file_path),
                'size_bytes': file_path.stat().st_size
            }

    if errors:
        return {
            'valid': False,
            'files': files,
            'error': '; '.join(errors)
        }

    return {
        'valid': True,
        'files': files
    }


def get_mesh_stats(obj_path: Path) -> dict:
    """
    Get basic mesh statistics from OBJ file.

    Args:
        obj_path: Path to OBJ file

    Returns:
        dict with 'vertex_count', 'face_count', 'has_uvs', 'has_normals',
        or 'error' if file cannot be parsed
    """
    obj_path = Path(obj_path)

    if not obj_path.exists():
        return {'error': f'File not found: {obj_path}'}

    vertex_count = 0
    face_count = 0
    has_uvs = False
    has_normals = False

    try:
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    vertex_count += 1
                elif line.startswith('f '):
                    face_count += 1
                elif line.startswith('vt '):
                    has_uvs = True
                elif line.startswith('vn '):
                    has_normals = True

        return {
            'vertex_count': vertex_count,
            'face_count': face_count,
            'has_uvs': has_uvs,
            'has_normals': has_normals
        }
    except Exception as e:
        return {'error': f'Failed to parse OBJ: {str(e)}'}
