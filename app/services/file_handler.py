"""
File handling service for upload validation and storage.
"""
import shutil
from pathlib import Path
from typing import BinaryIO

import aiofiles
from fastapi import UploadFile

# Constants
EXPECTED_VIEWS = 6
EXPECTED_DEPTH = 6
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB per file (2048x2048 PNGs can be large)
MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB total
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'

# Storage paths
STORAGE_ROOT = Path("storage/jobs")


class FileValidationError(Exception):
    """Exception raised when file validation fails."""

    def __init__(self, message: str, field: str | None = None):
        self.message = message
        self.field = field
        super().__init__(message)


async def validate_upload_files(
    views: list[UploadFile],
    depth_renders: list[UploadFile]
) -> None:
    """
    Validate uploaded files before processing.

    Checks:
    - Correct number of files (6 views, 6 depth)
    - PNG format (magic bytes)
    - File size limits (per-file and total)

    Raises:
        FileValidationError: If validation fails
    """
    # Check counts
    if len(views) != EXPECTED_VIEWS:
        raise FileValidationError(
            f"Expected {EXPECTED_VIEWS} view files, got {len(views)}",
            field="views"
        )

    if len(depth_renders) != EXPECTED_DEPTH:
        raise FileValidationError(
            f"Expected {EXPECTED_DEPTH} depth render files, got {len(depth_renders)}",
            field="depth_renders"
        )

    # Validate each file
    total_size = 0

    for i, view_file in enumerate(views):
        # Check PNG magic bytes
        magic = await view_file.read(8)
        if magic != PNG_MAGIC:
            await view_file.seek(0)
            raise FileValidationError(
                f"View file {i} is not a valid PNG (invalid magic bytes)",
                field=f"views[{i}]"
            )

        # Check file size
        await view_file.seek(0, 2)  # Seek to end
        size = await view_file.tell()

        if size > MAX_FILE_SIZE:
            await view_file.seek(0)
            raise FileValidationError(
                f"View file {i} exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
                field=f"views[{i}]"
            )

        total_size += size

        # CRITICAL: Reset file pointer for later use
        await view_file.seek(0)

    for i, depth_file in enumerate(depth_renders):
        # Check PNG magic bytes
        magic = await depth_file.read(8)
        if magic != PNG_MAGIC:
            await depth_file.seek(0)
            raise FileValidationError(
                f"Depth render file {i} is not a valid PNG (invalid magic bytes)",
                field=f"depth_renders[{i}]"
            )

        # Check file size
        await depth_file.seek(0, 2)  # Seek to end
        size = await depth_file.tell()

        if size > MAX_FILE_SIZE:
            await depth_file.seek(0)
            raise FileValidationError(
                f"Depth render file {i} exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
                field=f"depth_renders[{i}]"
            )

        total_size += size

        # CRITICAL: Reset file pointer for later use
        await depth_file.seek(0)

    # Check total size
    if total_size > MAX_TOTAL_SIZE:
        raise FileValidationError(
            f"Total upload size ({total_size / 1024 / 1024:.1f}MB) exceeds maximum of {MAX_TOTAL_SIZE / 1024 / 1024:.1f}MB",
            field="total_size"
        )


async def save_job_files(
    job_id: str,
    views: list[UploadFile],
    depth_renders: list[UploadFile]
) -> Path:
    """
    Save uploaded files to job directory.

    Directory structure:
        storage/jobs/{job_id}/
            views/
                view_00.png
                view_01.png
                ...
                view_05.png
            depth/
                depth_00.png
                depth_01.png
                ...
                depth_05.png

    Args:
        job_id: Job identifier
        views: List of 6 view image files
        depth_renders: List of 6 depth render files

    Returns:
        Path to job directory
    """
    # Create job directory structure
    job_dir = STORAGE_ROOT / job_id
    views_dir = job_dir / "views"
    depth_dir = job_dir / "depth"

    views_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Save view files
    for i, view_file in enumerate(views):
        file_path = views_dir / f"view_{i:02d}.png"
        async with aiofiles.open(file_path, 'wb') as f:
            content = await view_file.read()
            await f.write(content)

    # Save depth files
    for i, depth_file in enumerate(depth_renders):
        file_path = depth_dir / f"depth_{i:02d}.png"
        async with aiofiles.open(file_path, 'wb') as f:
            content = await depth_file.read()
            await f.write(content)

    return job_dir


def get_job_path(job_id: str) -> Path:
    """
    Get path to job directory (may not exist).

    Args:
        job_id: Job identifier

    Returns:
        Path to job directory
    """
    return STORAGE_ROOT / job_id


async def delete_job_files(job_id: str) -> bool:
    """
    Delete all files for a job.

    Args:
        job_id: Job identifier

    Returns:
        True if files were deleted, False if job directory didn't exist
    """
    job_dir = get_job_path(job_id)

    if job_dir.exists():
        shutil.rmtree(job_dir)
        return True

    return False
