"""
Result packaging service for job downloads.

Creates ZIP archives containing all job outputs:
- Mesh files (OBJ, PLY, GLB, MTL)
- Textures (mesh.png renamed to texture.png)
- Preview images (textured, wireframe)
- Quality reports (quality.json)
"""
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Tuple


class IncompleteResultsError(Exception):
    """Exception raised when required job outputs are missing."""

    def __init__(self, message: str, missing_items: list[str] | None = None):
        self.message = message
        self.missing_items = missing_items or []
        super().__init__(message)


def validate_job_outputs(output_dir: Path) -> Tuple[bool, list[str]]:
    """
    Validate that required job outputs exist.

    Checks:
    - Output directory exists
    - At least one model directory exists (reconviagen, nvdiffrec)
    - Each model directory has mesh.glb OR mesh.obj

    Args:
        output_dir: Path to job output directory

    Returns:
        Tuple of (is_valid, list of missing items)
    """
    missing: list[str] = []

    # Check output directory exists
    if not output_dir.exists():
        return False, ["output directory does not exist"]

    # Check for model directories
    model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    if not model_dirs:
        return False, ["no model output directories found"]

    # Check each model directory for required mesh file
    for model_dir in model_dirs:
        model_name = model_dir.name
        has_glb = (model_dir / "mesh.glb").exists()
        has_obj = (model_dir / "mesh.obj").exists()

        if not has_glb and not has_obj:
            missing.append(f"{model_name}: missing mesh.glb or mesh.obj")

    if missing:
        return False, missing

    return True, []


def create_result_zip(job_id: str, output_dir: Path) -> BytesIO:
    """
    Create ZIP archive containing all job outputs.

    ZIP structure per model directory:
    ```
    {model_name}/
        mesh.obj
        mesh.ply
        mesh.glb
        mesh.mtl (if exists)
        texture.png (from mesh.png)
        previews/
            textured_00.png ... textured_05.png
            wireframe_00.png ... wireframe_05.png
        quality.json
    ```

    Args:
        job_id: Job identifier (used for logging)
        output_dir: Path to job output directory

    Returns:
        BytesIO buffer containing ZIP file

    Raises:
        IncompleteResultsError: If required files are missing
    """
    # Validate outputs first
    is_valid, missing = validate_job_outputs(output_dir)
    if not is_valid:
        raise IncompleteResultsError(
            f"Job {job_id} has incomplete results",
            missing_items=missing
        )

    # Create ZIP in memory
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Process each model directory
        for model_dir in output_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Add mesh files (required: glb or obj, optional: ply, mtl)
            mesh_files = ["mesh.obj", "mesh.ply", "mesh.glb", "mesh.mtl"]
            for mesh_file in mesh_files:
                mesh_path = model_dir / mesh_file
                if mesh_path.exists():
                    arcname = f"{model_name}/{mesh_file}"
                    zf.write(mesh_path, arcname)

            # Add texture (mesh.png -> texture.png)
            texture_path = model_dir / "mesh.png"
            if texture_path.exists():
                arcname = f"{model_name}/texture.png"
                zf.write(texture_path, arcname)

            # Add preview images from previews/ subdirectory
            previews_dir = model_dir / "previews"
            if previews_dir.exists() and previews_dir.is_dir():
                for preview_file in previews_dir.glob("*.png"):
                    arcname = f"{model_name}/previews/{preview_file.name}"
                    zf.write(preview_file, arcname)

            # Add quality.json
            quality_path = model_dir / "quality.json"
            if quality_path.exists():
                arcname = f"{model_name}/quality.json"
                zf.write(quality_path, arcname)

    # CRITICAL: Seek to beginning for reading
    zip_buffer.seek(0)

    return zip_buffer
