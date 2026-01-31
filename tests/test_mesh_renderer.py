"""
Unit tests for mesh_renderer service.

Note: Most tests require torch/nvdiffrast which are only available in Docker.
GPU-dependent tests are skipped when CUDA is not available.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest


def _torch_available() -> bool:
    """Check if torch is available for testing."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    """Check if CUDA is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Marker for tests that require torch/nvdiffrast
requires_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="torch not available (tests run inside Docker)"
)

# Marker for tests that require CUDA
requires_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA not available"
)


@requires_torch
class TestLoadCameraPoses:
    """Tests for load_camera_poses function."""

    def test_load_camera_poses_basic(self, tmp_path):
        """Test loading camera poses from minimal transforms_train.json."""
        from app.services.mesh_renderer import load_camera_poses

        # Create minimal transforms_train.json
        transforms_data = {
            "camera_angle_x": 0.6911112070083618,
            "frames": [
                {
                    "file_path": "./images/view_00.png",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 2.5],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                },
                {
                    "file_path": "./images/view_01.png",
                    "transform_matrix": [
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, -2.5],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                }
            ]
        }

        # Write to temp directory
        transforms_path = tmp_path / "transforms_train.json"
        with open(transforms_path, "w") as f:
            json.dump(transforms_data, f)

        result = load_camera_poses(tmp_path)

        assert "camera_angle_x" in result
        assert result["camera_angle_x"] == pytest.approx(0.6911112070083618)

        assert "frames" in result
        assert len(result["frames"]) == 2

        # Check first frame
        frame0 = result["frames"][0]
        assert frame0["file_path"] == "./images/view_00.png"
        assert isinstance(frame0["transform_matrix"], np.ndarray)
        assert frame0["transform_matrix"].shape == (4, 4)
        assert frame0["transform_matrix"].dtype == np.float32

    def test_load_camera_poses_missing_file(self, tmp_path):
        """Test error when transforms_train.json not found."""
        from app.services.mesh_renderer import load_camera_poses

        with pytest.raises(FileNotFoundError) as exc_info:
            load_camera_poses(tmp_path)

        assert "transforms_train.json not found" in str(exc_info.value)

    def test_load_camera_poses_invalid_json(self, tmp_path):
        """Test error when JSON is malformed."""
        from app.services.mesh_renderer import load_camera_poses

        transforms_path = tmp_path / "transforms_train.json"
        with open(transforms_path, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            load_camera_poses(tmp_path)


@requires_torch
class TestBuildMvpMatrix:
    """Tests for build_mvp_matrix function."""

    def test_build_mvp_identity_transform(self):
        """Test MVP matrix with identity camera transform."""
        from app.services.mesh_renderer import build_mvp_matrix

        # Identity camera (at origin, looking along -Z)
        transform_matrix = np.eye(4, dtype=np.float32)

        # 60 degree FOV
        camera_angle_x = math.radians(60)
        resolution = (512, 512)

        mvp = build_mvp_matrix(transform_matrix, camera_angle_x, resolution)

        # Verify output shape and type
        assert mvp.shape == (4, 4)
        assert str(mvp.dtype).startswith("torch.float32")

    def test_build_mvp_camera_offset(self):
        """Test MVP matrix with camera at offset position."""
        from app.services.mesh_renderer import build_mvp_matrix

        # Camera at (0, 0, 2.5) looking at origin
        transform_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        camera_angle_x = 0.6911112070083618  # ~40 degrees
        resolution = (512, 512)

        mvp = build_mvp_matrix(transform_matrix, camera_angle_x, resolution)

        assert mvp.shape == (4, 4)

        # Origin point should project to center of screen
        origin = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        mvp_np = mvp.numpy()
        projected = mvp_np @ origin
        projected_ndc = projected[:3] / projected[3]

        # X and Y should be near 0 (center of NDC)
        assert abs(projected_ndc[0]) < 0.1
        assert abs(projected_ndc[1]) < 0.1

    def test_build_mvp_different_resolutions(self):
        """Test MVP matrix handles different aspect ratios."""
        from app.services.mesh_renderer import build_mvp_matrix

        transform_matrix = np.eye(4, dtype=np.float32)
        camera_angle_x = math.radians(60)

        # Square aspect
        mvp_square = build_mvp_matrix(transform_matrix, camera_angle_x, (512, 512))

        # Wide aspect
        mvp_wide = build_mvp_matrix(transform_matrix, camera_angle_x, (512, 1024))

        # Both should be 4x4
        assert mvp_square.shape == (4, 4)
        assert mvp_wide.shape == (4, 4)

        # Projection matrices should differ in aspect ratio term
        assert not np.allclose(mvp_square.numpy(), mvp_wide.numpy())

    def test_build_mvp_invalid_transform_shape(self):
        """Test error with invalid transform matrix shape."""
        from app.services.mesh_renderer import build_mvp_matrix

        # 3x3 matrix instead of 4x4
        transform_matrix = np.eye(3, dtype=np.float32)
        camera_angle_x = math.radians(60)

        with pytest.raises(ValueError) as exc_info:
            build_mvp_matrix(transform_matrix, camera_angle_x, (512, 512))

        assert "Expected 4x4 transform matrix" in str(exc_info.value)


@requires_torch
class TestMeshRendererInitialization:
    """Tests for MeshRenderer class initialization."""

    @requires_cuda
    def test_mesh_renderer_cuda_init(self):
        """Test MeshRenderer initialization with CUDA."""
        from app.services.mesh_renderer import MeshRenderer

        renderer = MeshRenderer(device="cuda")

        assert renderer.device.type == "cuda"
        assert renderer._glctx is not None


@requires_torch
class TestMeshRendererLoadMesh:
    """Tests for MeshRenderer.load_mesh method."""

    @requires_cuda
    def test_load_mesh_file_not_found(self, tmp_path):
        """Test load_mesh with non-existent file."""
        from app.services.mesh_renderer import MeshRenderer

        renderer = MeshRenderer(device="cuda")

        with pytest.raises(FileNotFoundError):
            renderer.load_mesh(tmp_path / "nonexistent.glb")


# Tests that don't require torch - run always
class TestVerticalFlipConvention:
    """Tests verifying vertical flip is applied consistently."""

    def test_rendered_array_is_flipped(self):
        """Verify rendered arrays are flipped vertically."""
        # This is a conceptual test - actual rendering requires GPU
        # Just verify the flip operation produces expected results

        # Original array with gradient from top to bottom
        original = np.arange(9).reshape(3, 3).astype(np.float32)
        # [[0, 1, 2],
        #  [3, 4, 5],
        #  [6, 7, 8]]

        # After vertical flip (bottom becomes top)
        flipped = original[::-1, :]
        # [[6, 7, 8],
        #  [3, 4, 5],
        #  [0, 1, 2]]

        assert flipped[0, 0] == 6
        assert flipped[2, 0] == 0
        np.testing.assert_array_equal(flipped, np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]))
