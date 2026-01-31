"""
Camera pose estimation and NeRF dataset preparation for nvdiffrec.

nvdiffrec expects input in NeRF synthetic dataset format:
- transforms_train.json with camera intrinsics and extrinsics
- Images with alpha channel (RGBA) for masking

Since our input has known camera poses (orthogonal views), we generate
canonical poses rather than estimating them with COLMAP/MASt3R.
"""
import json
import math
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import shutil

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Standard orthogonal view directions (camera looks at origin)
# Order: front (+Z), back (-Z), right (+X), left (-X), top (+Y), bottom (-Y)
CANONICAL_VIEWS = [
    {"name": "front",  "position": [0, 0, 2.5],  "up": [0, 1, 0]},
    {"name": "back",   "position": [0, 0, -2.5], "up": [0, 1, 0]},
    {"name": "right",  "position": [2.5, 0, 0],  "up": [0, 1, 0]},
    {"name": "left",   "position": [-2.5, 0, 0], "up": [0, 1, 0]},
    {"name": "top",    "position": [0, 2.5, 0],  "up": [0, 0, -1]},
    {"name": "bottom", "position": [0, -2.5, 0], "up": [0, 0, 1]},
]


def look_at_matrix(eye: List[float], target: List[float], up: List[float]) -> np.ndarray:
    """
    Create a look-at camera matrix (camera-to-world transform).

    Args:
        eye: Camera position [x, y, z]
        target: Look-at target [x, y, z]
        up: Up vector [x, y, z]

    Returns:
        4x4 camera-to-world transformation matrix (OpenGL convention)
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    # Forward vector (camera looks along -Z in OpenGL)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Right vector
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute up to ensure orthogonality
    up = np.cross(right, forward)

    # Build rotation matrix (columns are right, up, -forward)
    rotation = np.eye(4, dtype=np.float64)
    rotation[:3, 0] = right
    rotation[:3, 1] = up
    rotation[:3, 2] = -forward  # OpenGL convention

    # Build translation
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = eye

    # Camera-to-world = translation @ rotation
    return translation @ rotation


def compute_fov_x(image_width: int, focal_length: float) -> float:
    """Compute horizontal field of view in radians."""
    return 2 * math.atan(image_width / (2 * focal_length))


def create_nerf_dataset(
    views_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    image_size: int = 512,
    focal_length: float = 1111.0,  # Default for 512px with ~50 degree FOV
) -> Dict:
    """
    Convert multi-view images to NeRF synthetic dataset format.

    Args:
        views_dir: Directory with view_00.png ... view_05.png
        depth_dir: Directory with depth_00.png ... depth_05.png (for masking)
        output_dir: Where to write transforms_train.json and images
        image_size: Output image resolution (images will be resized)
        focal_length: Camera focal length in pixels

    Returns:
        Dict with status and paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images subdirectory
    images_out = output_dir / "images"
    images_out.mkdir(exist_ok=True)

    # Compute camera angle (FOV)
    camera_angle_x = compute_fov_x(image_size, focal_length)

    frames = []
    view_files = sorted(Path(views_dir).glob("view_*.png"))

    if len(view_files) != 6:
        return {
            "status": "failed",
            "error": f"Expected 6 view files, found {len(view_files)}"
        }

    for i, (view_file, view_config) in enumerate(zip(view_files, CANONICAL_VIEWS)):
        # Load and process image
        img = Image.open(view_file).convert("RGBA")

        # Resize if needed
        if img.size[0] != image_size or img.size[1] != image_size:
            img = img.resize((image_size, image_size), Image.LANCZOS)

        # Try to use depth for alpha mask
        depth_file = Path(depth_dir) / f"depth_{i:02d}.png"
        if depth_file.exists():
            depth = Image.open(depth_file).convert("L")
            depth = depth.resize((image_size, image_size), Image.LANCZOS)
            # Non-zero depth = object, zero depth = background
            depth_array = np.array(depth)
            alpha = (depth_array > 0).astype(np.uint8) * 255
            img_array = np.array(img)
            img_array[:, :, 3] = alpha
            img = Image.fromarray(img_array)

        # Save processed image
        out_name = f"view_{i:02d}.png"
        img.save(images_out / out_name)

        # Create camera transform matrix
        transform_matrix = look_at_matrix(
            eye=view_config["position"],
            target=[0, 0, 0],  # Look at origin
            up=view_config["up"]
        )

        frames.append({
            "file_path": f"./images/{out_name}",
            "transform_matrix": transform_matrix.tolist()
        })

        logger.debug(f"Processed view {i}: {view_config['name']}")

    # Write transforms_train.json
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }

    transforms_path = output_dir / "transforms_train.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    logger.info(f"Created NeRF dataset with {len(frames)} views at {output_dir}")

    return {
        "status": "success",
        "transforms_path": str(transforms_path),
        "image_count": len(frames),
        "image_size": image_size
    }


def validate_nerf_dataset(dataset_dir: Path) -> Dict:
    """
    Validate a NeRF-format dataset for nvdiffrec compatibility.

    Args:
        dataset_dir: Directory containing transforms_train.json

    Returns:
        Dict with validation status and details
    """
    dataset_dir = Path(dataset_dir)
    transforms_path = dataset_dir / "transforms_train.json"

    if not transforms_path.exists():
        return {"valid": False, "error": "transforms_train.json not found"}

    try:
        with open(transforms_path) as f:
            transforms = json.load(f)

        # Check required fields
        if "camera_angle_x" not in transforms:
            return {"valid": False, "error": "Missing camera_angle_x"}

        if "frames" not in transforms:
            return {"valid": False, "error": "Missing frames array"}

        frames = transforms["frames"]
        if len(frames) < 1:
            return {"valid": False, "error": "No frames in dataset"}

        # Validate each frame
        for i, frame in enumerate(frames):
            if "file_path" not in frame:
                return {"valid": False, "error": f"Frame {i} missing file_path"}
            if "transform_matrix" not in frame:
                return {"valid": False, "error": f"Frame {i} missing transform_matrix"}

            # Check image exists
            img_path = dataset_dir / frame["file_path"]
            if not img_path.exists():
                return {"valid": False, "error": f"Image not found: {frame['file_path']}"}

            # Check transform matrix shape
            matrix = frame["transform_matrix"]
            if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                return {"valid": False, "error": f"Frame {i} transform_matrix not 4x4"}

        return {
            "valid": True,
            "frame_count": len(frames),
            "camera_angle_x": transforms["camera_angle_x"]
        }

    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}
