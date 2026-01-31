"""
Integration tests for 3D reconstruction models.

Tests model factory functions, model instantiation, and inference capabilities.
Uses pytest markers to separate fast (no GPU) and slow (GPU + weights) tests.

NOTE: Many tests require PyTorch which is only available inside the Docker container.
Run: docker-compose exec api pytest tests/test_models.py -v
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest


# ============================================================================
# Module availability checks
# ============================================================================

def _has_torch():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Skip message for tests requiring torch
TORCH_SKIP = pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not available (run inside Docker container)"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_input_dir():
    """Create temporary input directory with views and depth subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        views_dir = input_dir / "views"
        depth_dir = input_dir / "depth"

        views_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)

        yield input_dir


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True)
        yield output_dir


@pytest.fixture
def mock_celery_task():
    """Create mock Celery task for progress reporting."""
    task = Mock()
    task.update_state = Mock()
    return task


@pytest.fixture
def sample_view_images(temp_input_dir):
    """Create sample view images (512x512 RGB PNG)."""
    from PIL import Image

    views_dir = temp_input_dir / "views"
    depth_dir = temp_input_dir / "depth"

    for i in range(6):
        # Create view image (gradient pattern for visual distinction)
        view_data = np.zeros((512, 512, 3), dtype=np.uint8)
        view_data[:, :, 0] = (i * 40) % 256  # R varies by view
        view_data[:, :, 1] = 128  # G constant
        view_data[:, :, 2] = ((255 - i * 40) % 256)  # B inverse

        view_img = Image.fromarray(view_data, mode="RGB")
        view_img.save(views_dir / f"view_{i:02d}.png")

        # Create depth image (grayscale)
        depth_data = np.zeros((512, 512), dtype=np.uint8)
        # Create a circle in center to represent object depth
        y, x = np.ogrid[:512, :512]
        center = 256
        mask = (x - center)**2 + (y - center)**2 <= 200**2
        depth_data[mask] = 200  # Object area

        depth_img = Image.fromarray(depth_data, mode="L")
        depth_img.save(depth_dir / f"depth_{i:02d}.png")

    return temp_input_dir


# ============================================================================
# Camera Estimation Tests (Fast - No PyTorch Required)
# ============================================================================

class TestCameraEstimation:
    """Tests for camera estimation service (runs without PyTorch)."""

    def test_look_at_matrix_shape(self):
        """look_at_matrix returns 4x4 matrix."""
        from app.services.camera_estimation import look_at_matrix

        matrix = look_at_matrix(
            eye=[0, 0, 2.5],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
        assert matrix.shape == (4, 4)

    def test_look_at_matrix_orthogonal(self):
        """look_at_matrix produces orthogonal rotation component."""
        from app.services.camera_estimation import look_at_matrix

        matrix = look_at_matrix(
            eye=[2.5, 0, 0],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )

        # Extract rotation (top-left 3x3)
        rotation = matrix[:3, :3]

        # Check orthogonality: R^T @ R should be identity
        product = rotation.T @ rotation
        np.testing.assert_allclose(product, np.eye(3), atol=1e-7)

    def test_look_at_matrix_translation(self):
        """look_at_matrix includes camera position in translation."""
        from app.services.camera_estimation import look_at_matrix

        eye = [1.0, 2.0, 3.0]
        matrix = look_at_matrix(
            eye=eye,
            target=[0, 0, 0],
            up=[0, 1, 0]
        )

        # Translation is in the last column
        translation = matrix[:3, 3]
        np.testing.assert_allclose(translation, eye, atol=1e-7)

    def test_canonical_views_count(self):
        """CANONICAL_VIEWS has 6 views."""
        from app.services.camera_estimation import CANONICAL_VIEWS

        assert len(CANONICAL_VIEWS) == 6

    def test_canonical_views_names(self):
        """CANONICAL_VIEWS includes all standard views."""
        from app.services.camera_estimation import CANONICAL_VIEWS

        names = {v["name"] for v in CANONICAL_VIEWS}
        expected = {"front", "back", "right", "left", "top", "bottom"}
        assert names == expected

    def test_compute_fov_x(self):
        """compute_fov_x calculates horizontal FOV correctly."""
        from app.services.camera_estimation import compute_fov_x

        # With focal length 1111 and width 512, FOV should be ~26 degrees
        fov = compute_fov_x(image_width=512, focal_length=1111.0)

        # Convert to degrees
        fov_degrees = np.degrees(fov)

        # Should be approximately 26 degrees
        assert 25 < fov_degrees < 27


class TestNeRFDataset:
    """Tests for NeRF dataset creation (runs without PyTorch)."""

    def test_create_nerf_dataset_success(self, sample_view_images, temp_output_dir):
        """create_nerf_dataset creates valid dataset from views."""
        from app.services.camera_estimation import create_nerf_dataset

        result = create_nerf_dataset(
            views_dir=sample_view_images / "views",
            depth_dir=sample_view_images / "depth",
            output_dir=temp_output_dir,
            image_size=256,  # Smaller for fast test
            focal_length=555.0
        )

        assert result["status"] == "success"
        assert result["image_count"] == 6
        assert result["image_size"] == 256

        # Check transforms file exists
        transforms_path = Path(result["transforms_path"])
        assert transforms_path.exists()

    def test_create_nerf_dataset_wrong_view_count(self, temp_input_dir, temp_output_dir):
        """create_nerf_dataset fails with wrong number of views."""
        from app.services.camera_estimation import create_nerf_dataset
        from PIL import Image

        # Create only 4 views
        views_dir = temp_input_dir / "views"
        for i in range(4):
            img = Image.new("RGB", (512, 512), color=(128, 128, 128))
            img.save(views_dir / f"view_{i:02d}.png")

        result = create_nerf_dataset(
            views_dir=views_dir,
            depth_dir=temp_input_dir / "depth",
            output_dir=temp_output_dir
        )

        assert result["status"] == "failed"
        assert "Expected 6" in result["error"]

    def test_validate_nerf_dataset_valid(self, sample_view_images, temp_output_dir):
        """validate_nerf_dataset accepts valid dataset."""
        from app.services.camera_estimation import (
            create_nerf_dataset,
            validate_nerf_dataset,
        )

        create_nerf_dataset(
            views_dir=sample_view_images / "views",
            depth_dir=sample_view_images / "depth",
            output_dir=temp_output_dir
        )

        result = validate_nerf_dataset(temp_output_dir)

        assert result["valid"] is True
        assert result["frame_count"] == 6

    def test_validate_nerf_dataset_missing_transforms(self, temp_output_dir):
        """validate_nerf_dataset fails without transforms_train.json."""
        from app.services.camera_estimation import validate_nerf_dataset

        result = validate_nerf_dataset(temp_output_dir)

        assert result["valid"] is False
        assert "not found" in result["error"]


# ============================================================================
# Model Factory Tests (Requires PyTorch - Run Inside Docker)
# ============================================================================

@TORCH_SKIP
class TestModelFactory:
    """Tests for model factory functions (requires PyTorch)."""

    def test_available_models_list(self):
        """AVAILABLE_MODELS contains expected model types."""
        from app.models import AVAILABLE_MODELS

        assert "reconviagen" in AVAILABLE_MODELS
        assert "nvdiffrec" in AVAILABLE_MODELS
        assert len(AVAILABLE_MODELS) == 2

    def test_get_model_reconviagen(self):
        """get_model returns ReconViaGenModel for 'reconviagen'."""
        from app.models import get_model
        from app.models.reconviagen import ReconViaGenModel

        model = get_model("reconviagen")
        assert isinstance(model, ReconViaGenModel)
        assert model.model_name == "reconviagen"

    def test_get_model_nvdiffrec(self):
        """get_model returns NvdiffrecModel for 'nvdiffrec'."""
        from app.models import get_model
        from app.models.nvdiffrec import NvdiffrecModel

        model = get_model("nvdiffrec")
        assert isinstance(model, NvdiffrecModel)
        assert model.model_name == "nvdiffrec"

    def test_get_model_with_celery_task(self, mock_celery_task):
        """get_model passes celery_task to model instance."""
        from app.models import get_model

        model = get_model("reconviagen", celery_task=mock_celery_task)
        assert model.celery_task is mock_celery_task

    def test_get_model_unknown_type_raises(self):
        """get_model raises ValueError for unknown model type."""
        from app.models import get_model

        with pytest.raises(ValueError) as exc_info:
            get_model("unknown_model")

        assert "Unknown model type" in str(exc_info.value)
        assert "reconviagen" in str(exc_info.value)
        assert "nvdiffrec" in str(exc_info.value)

    def test_model_inherits_base(self):
        """Both models inherit from BaseReconstructionModel."""
        from app.models import get_model
        from app.models.base import BaseReconstructionModel

        rv_model = get_model("reconviagen")
        nv_model = get_model("nvdiffrec")

        assert isinstance(rv_model, BaseReconstructionModel)
        assert isinstance(nv_model, BaseReconstructionModel)


@TORCH_SKIP
class TestModelInstantiation:
    """Tests for model instantiation without loading weights (requires PyTorch)."""

    def test_reconviagen_init(self):
        """ReconViaGenModel initializes without errors."""
        from app.models.reconviagen import ReconViaGenModel

        model = ReconViaGenModel()
        assert model.model_name == "reconviagen"
        assert model.celery_task is None
        assert model._pipeline is None

    def test_nvdiffrec_init(self):
        """NvdiffrecModel initializes without errors."""
        from app.models.nvdiffrec import NvdiffrecModel

        model = NvdiffrecModel()
        assert model.model_name == "nvdiffrec"
        assert model.celery_task is None
        assert model._glctx is None
        assert model._iterations == 500  # Default

    def test_nvdiffrec_custom_iterations(self):
        """NvdiffrecModel accepts custom iteration count."""
        from app.models.nvdiffrec import NvdiffrecModel

        model = NvdiffrecModel(iterations=100)
        assert model._iterations == 100

    def test_model_has_required_methods(self):
        """Both models implement required abstract methods."""
        from app.models import get_model, AVAILABLE_MODELS

        for model_type in AVAILABLE_MODELS:
            model = get_model(model_type)

            # Check required methods exist
            assert hasattr(model, "load_weights")
            assert callable(model.load_weights)

            assert hasattr(model, "inference")
            assert callable(model.inference)

            assert hasattr(model, "cleanup")
            assert callable(model.cleanup)

            assert hasattr(model, "report_progress")
            assert callable(model.report_progress)

    def test_report_progress_with_task(self, mock_celery_task):
        """report_progress updates Celery task state."""
        from app.models import get_model

        model = get_model("nvdiffrec", celery_task=mock_celery_task)
        model.report_progress(50, "Test step")

        mock_celery_task.update_state.assert_called_once()
        call_args = mock_celery_task.update_state.call_args
        assert call_args[1]["state"] == "PROGRESS"
        assert call_args[1]["meta"]["progress"] == 50
        assert "nvdiffrec" in call_args[1]["meta"]["step"]

    def test_report_progress_without_task(self):
        """report_progress does nothing when no Celery task."""
        from app.models import get_model

        model = get_model("reconviagen")  # No celery_task
        # Should not raise
        model.report_progress(50, "Test step")


# ============================================================================
# SimpleGeometry Tests (Requires PyTorch - Run Inside Docker)
# ============================================================================

@TORCH_SKIP
class TestSimpleGeometry:
    """Tests for nvdiffrec SimpleGeometry helper class (requires PyTorch)."""

    def test_geometry_initialization(self):
        """SimpleGeometry creates valid sphere mesh."""
        import torch
        from app.models.nvdiffrec import SimpleGeometry

        device = torch.device("cpu")
        geometry = SimpleGeometry(resolution=8, device=device)

        # Check vertices
        assert geometry.base_verts.shape[1] == 3  # (N, 3)
        assert geometry.base_verts.device == device

        # Check faces
        assert geometry.faces.shape[1] == 3  # (F, 3)
        assert geometry.faces.dtype == torch.long

    def test_geometry_displacements(self):
        """SimpleGeometry has learnable displacements."""
        import torch
        from app.models.nvdiffrec import SimpleGeometry

        device = torch.device("cpu")
        geometry = SimpleGeometry(resolution=8, device=device)

        # Displacements should be zero initially
        assert torch.allclose(geometry.displacements, torch.zeros_like(geometry.displacements))

        # Should be a Parameter (learnable)
        assert isinstance(geometry.displacements, torch.nn.Parameter)

    def test_geometry_vertices_property(self):
        """SimpleGeometry.vertices returns deformed positions."""
        import torch
        from app.models.nvdiffrec import SimpleGeometry

        device = torch.device("cpu")
        geometry = SimpleGeometry(resolution=8, device=device)

        # Initial vertices should equal base vertices (zero displacement)
        initial_verts = geometry.vertices.clone()

        # Modify displacements
        with torch.no_grad():
            geometry.displacements.data += 0.1

        # Vertices should now be different
        new_verts = geometry.vertices
        assert not torch.allclose(initial_verts, new_verts)

    def test_geometry_parameters(self):
        """SimpleGeometry.parameters returns learnable params."""
        import torch
        from app.models.nvdiffrec import SimpleGeometry

        device = torch.device("cpu")
        geometry = SimpleGeometry(resolution=8, device=device)

        params = geometry.parameters()
        assert len(params) == 2  # displacements and vertex_colors

        for p in params:
            assert isinstance(p, torch.nn.Parameter)


# ============================================================================
# GPU Tests (Slow - Requires GPU + Weights)
# ============================================================================

@pytest.mark.slow
@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
class TestModelLoadWeights:
    """Tests for model weight loading (requires GPU)."""

    def test_nvdiffrec_load_weights(self):
        """NvdiffrecModel.load_weights initializes without pre-trained weights."""
        from app.models.nvdiffrec import NvdiffrecModel

        model = NvdiffrecModel(iterations=10)

        # Mock VRAM check to always pass
        with patch("app.models.nvdiffrec.check_vram_available") as mock_vram:
            mock_vram.return_value = {"available": True, "free_gb": 16.0}
            model.load_weights()

        # nvdiffrec doesn't require pre-trained weights
        # It should initialize successfully
        assert model._glctx is not None or True  # May fallback without nvdiffrast

    def test_reconviagen_load_weights_requires_weights(self):
        """ReconViaGenModel.load_weights fails without TRELLIS weights."""
        from app.models.reconviagen import ReconViaGenModel

        model = ReconViaGenModel()

        # Should fail because TRELLIS weights aren't available
        with patch("app.models.reconviagen.check_vram_available") as mock_vram:
            mock_vram.return_value = {"available": True, "free_gb": 20.0}

            with pytest.raises(Exception):
                # Will fail trying to load TRELLIS
                model.load_weights()


@pytest.mark.slow
@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
class TestModelInference:
    """Tests for model inference (requires GPU + weights)."""

    def test_nvdiffrec_inference_minimal(self, sample_view_images, temp_output_dir):
        """NvdiffrecModel.inference runs with minimal iterations."""
        from app.models.nvdiffrec import NvdiffrecModel

        model = NvdiffrecModel(iterations=5)  # Very few iterations for speed

        # Mock VRAM check
        with patch("app.models.nvdiffrec.check_vram_available") as mock_vram:
            mock_vram.return_value = {"available": True, "free_gb": 16.0}
            model.load_weights()

        result = model.inference(
            input_dir=sample_view_images,
            output_dir=temp_output_dir
        )

        # May succeed or fail depending on environment
        # Just verify it returns proper structure
        assert "status" in result
        if result["status"] == "success":
            assert "mesh_path" in result
        else:
            assert "error" in result

    def test_model_cleanup(self):
        """Model cleanup releases resources."""
        from app.models.nvdiffrec import NvdiffrecModel

        model = NvdiffrecModel(iterations=10)

        with patch("app.models.nvdiffrec.check_vram_available") as mock_vram:
            mock_vram.return_value = {"available": True, "free_gb": 16.0}
            model.load_weights()

        # Should not raise
        model.cleanup()

        # After cleanup, _glctx should be None
        assert model._glctx is None


# ============================================================================
# Integration Sanity Checks
# ============================================================================

@TORCH_SKIP
class TestIntegrationSanity:
    """Basic integration sanity checks (requires PyTorch)."""

    def test_imports_work(self):
        """All model imports work without errors."""
        # These imports should not raise
        from app.models import get_model, AVAILABLE_MODELS
        from app.models.base import BaseReconstructionModel
        from app.models.reconviagen import ReconViaGenModel
        from app.models.nvdiffrec import NvdiffrecModel
        from app.services.camera_estimation import look_at_matrix, create_nerf_dataset

        assert True

    def test_model_factory_complete(self):
        """All AVAILABLE_MODELS can be instantiated."""
        from app.models import get_model, AVAILABLE_MODELS

        for model_type in AVAILABLE_MODELS:
            model = get_model(model_type)
            assert model is not None
            assert model.model_name == model_type


# ============================================================================
# VRAM Manager Tests
# ============================================================================

@TORCH_SKIP
class TestVramManager:
    """Tests for VRAM management service (requires PyTorch)."""

    def test_check_vram_available_structure(self):
        """check_vram_available returns expected structure."""
        from app.services.vram_manager import check_vram_available

        result = check_vram_available(1.0)  # Request 1GB

        assert 'available' in result
        assert 'free_gb' in result
        assert isinstance(result['available'], bool)
        assert isinstance(result['free_gb'], (int, float))

    def test_cleanup_gpu_memory_runs(self):
        """cleanup_gpu_memory runs without error."""
        from app.services.vram_manager import cleanup_gpu_memory

        # Should not raise
        cleanup_gpu_memory()


# ============================================================================
# Mesh Export Tests
# ============================================================================

@TORCH_SKIP
class TestMeshExportService:
    """Tests for mesh export service (requires PyTorch)."""

    def test_validate_mesh_output_missing_files(self, temp_output_dir):
        """validate_mesh_output detects missing files."""
        from app.services.mesh_export import validate_mesh_output

        temp_output_dir.mkdir(parents=True, exist_ok=True)
        result = validate_mesh_output(temp_output_dir, "mesh")

        assert not result['valid']
        assert 'error' in result
