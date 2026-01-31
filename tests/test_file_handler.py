"""
Unit tests for file_handler service.
"""
import io
from unittest.mock import Mock

import pytest

from app.services.file_handler import (
    EXPECTED_DEPTH,
    EXPECTED_VIEWS,
    MAX_FILE_SIZE,
    PNG_MAGIC,
    FileValidationError,
    validate_upload_files,
)


def create_mock_upload_file(content: bytes, filename: str = "test.png"):
    """Create a mock UploadFile for testing."""
    mock_file = Mock()
    mock_file.filename = filename

    # Create BytesIO buffer
    buffer = io.BytesIO(content)

    # Mock async read/seek/tell methods
    async def async_read(size=-1):
        return buffer.read(size)

    async def async_seek(offset, whence=0):
        return buffer.seek(offset, whence)

    async def async_tell():
        return buffer.tell()

    mock_file.read = async_read
    mock_file.seek = async_seek
    mock_file.tell = async_tell

    return mock_file


@pytest.mark.asyncio
async def test_validate_wrong_view_count():
    """Test validation fails with wrong number of view files."""
    # Create 5 views instead of 6
    views = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 100)
        for _ in range(5)
    ]
    depth_renders = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 100)
        for _ in range(6)
    ]

    with pytest.raises(FileValidationError) as exc_info:
        await validate_upload_files(views, depth_renders)

    assert "Expected 6 view files, got 5" in str(exc_info.value)
    assert exc_info.value.field == "views"


@pytest.mark.asyncio
async def test_validate_wrong_depth_count():
    """Test validation fails with wrong number of depth files."""
    # Create 7 depth renders instead of 6
    views = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 100)
        for _ in range(6)
    ]
    depth_renders = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 100)
        for _ in range(7)
    ]

    with pytest.raises(FileValidationError) as exc_info:
        await validate_upload_files(views, depth_renders)

    assert "Expected 6 depth render files, got 7" in str(exc_info.value)
    assert exc_info.value.field == "depth_renders"


@pytest.mark.asyncio
async def test_validate_non_png_file():
    """Test validation fails with non-PNG file."""
    # JPEG magic bytes: FF D8 FF
    jpeg_magic = b'\xFF\xD8\xFF\xE0'

    views = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 100)
        for _ in range(5)
    ]
    # Add one JPEG file
    views.append(create_mock_upload_file(jpeg_magic + b'\x00' * 100, "test.jpg"))

    depth_renders = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 100)
        for _ in range(6)
    ]

    with pytest.raises(FileValidationError) as exc_info:
        await validate_upload_files(views, depth_renders)

    assert "PNG" in str(exc_info.value)
    assert "magic bytes" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_oversized_file():
    """Test validation fails with oversized file."""
    # Create a file larger than MAX_FILE_SIZE
    large_content = PNG_MAGIC + b'\x00' * (MAX_FILE_SIZE + 1024)

    views = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 100)
        for _ in range(5)
    ]
    # Add one oversized file
    views.append(create_mock_upload_file(large_content))

    depth_renders = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 100)
        for _ in range(6)
    ]

    with pytest.raises(FileValidationError) as exc_info:
        await validate_upload_files(views, depth_renders)

    assert "exceeds maximum size" in str(exc_info.value)
    assert "MB" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_success():
    """Test validation succeeds with valid files."""
    # Create valid PNG files (small size, proper magic bytes)
    views = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 1024)
        for _ in range(6)
    ]
    depth_renders = [
        create_mock_upload_file(PNG_MAGIC + b'\x00' * 1024)
        for _ in range(6)
    ]

    # Should not raise any exception
    await validate_upload_files(views, depth_renders)


@pytest.mark.asyncio
async def test_file_pointer_reset_after_validation():
    """Test that file pointers are reset to beginning after validation."""
    # Create valid files
    content = PNG_MAGIC + b'test_content_here'
    views = [
        create_mock_upload_file(content)
        for _ in range(6)
    ]
    depth_renders = [
        create_mock_upload_file(content)
        for _ in range(6)
    ]

    # Validate files
    await validate_upload_files(views, depth_renders)

    # Verify we can read from beginning again
    for view in views:
        data = await view.read()
        assert data == content, "File pointer was not reset after validation"

    for depth in depth_renders:
        data = await depth.read()
        assert data == content, "File pointer was not reset after validation"
