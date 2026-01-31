"""Integration tests for quality pipeline.

Tests quality metrics computation, classification, and report generation.
These tests focus on the quality_metrics and preview_generator modules.

Note: Many tests require skimage which depends on scipy/numpy.
Imports are lazy to avoid issues when torch/nvdiffrast are not installed.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _skimage_available() -> bool:
    """Check if skimage (scikit-image) is available for testing."""
    try:
        from skimage.metrics import peak_signal_noise_ratio
        return True
    except ImportError:
        return False


# Marker for tests that require skimage
requires_skimage = pytest.mark.skipif(
    not _skimage_available(),
    reason="skimage not available (tests run inside Docker)"
)


# Threshold constants (replicated here to avoid import issues)
PSNR_NORMAL_THRESHOLD = 25.0
PSNR_WARNING_THRESHOLD = 20.0
SSIM_NORMAL_THRESHOLD = 0.85
SSIM_WARNING_THRESHOLD = 0.75


@requires_skimage
class TestQualityMetricsComputation:
    """Test PSNR and SSIM computation."""

    def test_psnr_identical_images(self):
        """PSNR of identical images should be infinite."""
        from app.services.quality_metrics import compute_psnr
        img = np.random.rand(64, 64, 3).astype(np.float32)
        psnr = compute_psnr(img, img)
        assert psnr == float('inf'), "PSNR of identical images should be inf"

    def test_psnr_different_images(self):
        """PSNR of different images should be finite and positive."""
        from app.services.quality_metrics import compute_psnr
        img1 = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        img2 = np.ones((64, 64, 3), dtype=np.float32) * 0.6
        psnr = compute_psnr(img1, img2)
        assert np.isfinite(psnr), "PSNR should be finite for different images"
        assert psnr > 0, "PSNR should be positive"

    def test_psnr_random_images(self):
        """PSNR with random noise should be in typical range."""
        from app.services.quality_metrics import compute_psnr
        img1 = np.random.rand(64, 64, 3).astype(np.float32)
        # Add small noise
        noise = np.random.randn(64, 64, 3).astype(np.float32) * 0.05
        img2 = np.clip(img1 + noise, 0.0, 1.0)
        psnr = compute_psnr(img1, img2)
        assert np.isfinite(psnr), "PSNR should be finite"
        assert 10 < psnr < 50, f"PSNR {psnr:.2f} dB should be in reasonable range"

    def test_ssim_identical_images(self):
        """SSIM of identical images should be 1.0."""
        from app.services.quality_metrics import compute_ssim
        img = np.random.rand(64, 64, 3).astype(np.float32)
        ssim = compute_ssim(img, img)
        assert ssim == pytest.approx(1.0), "SSIM of identical images should be 1.0"

    def test_ssim_different_images(self):
        """SSIM of different images should be in [0, 1]."""
        from app.services.quality_metrics import compute_ssim
        img1 = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        img2 = np.ones((64, 64, 3), dtype=np.float32) * 0.7
        ssim = compute_ssim(img1, img2)
        assert 0.0 <= ssim <= 1.0, f"SSIM {ssim:.4f} should be in [0, 1]"

    def test_ssim_random_images_with_noise(self):
        """SSIM with small noise should still be high."""
        from app.services.quality_metrics import compute_ssim
        img1 = np.random.rand(64, 64, 3).astype(np.float32)
        # Add very small noise
        noise = np.random.randn(64, 64, 3).astype(np.float32) * 0.01
        img2 = np.clip(img1 + noise, 0.0, 1.0)
        ssim = compute_ssim(img1, img2)
        assert ssim > 0.9, f"SSIM {ssim:.4f} should be high with small noise"

    def test_shape_mismatch_psnr(self):
        """PSNR should raise ValueError for shape mismatch."""
        from app.services.quality_metrics import compute_psnr
        img1 = np.random.rand(64, 64, 3).astype(np.float32)
        img2 = np.random.rand(32, 32, 3).astype(np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_psnr(img1, img2)

    def test_shape_mismatch_ssim(self):
        """SSIM should raise ValueError for shape mismatch."""
        from app.services.quality_metrics import compute_ssim
        img1 = np.random.rand(64, 64, 3).astype(np.float32)
        img2 = np.random.rand(32, 32, 3).astype(np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_ssim(img1, img2)


@requires_skimage
class TestQualityStatusClassification:
    """Test quality status classification based on thresholds."""

    def test_normal_status(self):
        """High PSNR and SSIM should result in 'normal' status."""
        from app.services.quality_metrics import classify_quality_status
        assert classify_quality_status(30.0, 0.90) == "normal"
        assert classify_quality_status(25.0, 0.85) == "normal"
        assert classify_quality_status(40.0, 0.99) == "normal"

    def test_warning_status(self):
        """Medium PSNR and SSIM should result in 'warning' status."""
        from app.services.quality_metrics import classify_quality_status
        assert classify_quality_status(22.0, 0.80) == "warning"
        assert classify_quality_status(20.0, 0.75) == "warning"
        assert classify_quality_status(24.9, 0.84) == "warning"

    def test_failure_status(self):
        """Low PSNR or SSIM should result in 'failure' status."""
        from app.services.quality_metrics import classify_quality_status
        assert classify_quality_status(15.0, 0.60) == "failure"
        assert classify_quality_status(19.9, 0.75) == "failure"  # PSNR too low
        assert classify_quality_status(25.0, 0.74) == "failure"  # SSIM too low
        assert classify_quality_status(10.0, 0.90) == "failure"  # PSNR too low

    def test_threshold_boundaries(self):
        """Test exact threshold boundaries."""
        from app.services.quality_metrics import classify_quality_status
        # Exactly at normal thresholds -> normal
        assert classify_quality_status(PSNR_NORMAL_THRESHOLD, SSIM_NORMAL_THRESHOLD) == "normal"

        # Just below normal but above warning -> warning
        assert classify_quality_status(PSNR_NORMAL_THRESHOLD - 0.1, SSIM_NORMAL_THRESHOLD) == "warning"
        assert classify_quality_status(PSNR_NORMAL_THRESHOLD, SSIM_NORMAL_THRESHOLD - 0.01) == "warning"

        # Exactly at warning thresholds -> warning
        assert classify_quality_status(PSNR_WARNING_THRESHOLD, SSIM_WARNING_THRESHOLD) == "warning"

        # Just below warning -> failure
        assert classify_quality_status(PSNR_WARNING_THRESHOLD - 0.1, SSIM_WARNING_THRESHOLD) == "failure"
        assert classify_quality_status(PSNR_WARNING_THRESHOLD, SSIM_WARNING_THRESHOLD - 0.01) == "failure"

    def test_and_logic(self):
        """Both PSNR and SSIM must pass for a status level (AND logic)."""
        from app.services.quality_metrics import classify_quality_status
        # High PSNR but low SSIM -> failure
        assert classify_quality_status(30.0, 0.5) == "failure"

        # Low PSNR but high SSIM -> failure
        assert classify_quality_status(15.0, 0.95) == "failure"

        # Medium PSNR high SSIM -> warning (PSNR determines)
        assert classify_quality_status(22.0, 0.95) == "warning"


@requires_skimage
class TestQualityReportStructure:
    """Test quality report generation and structure."""

    def test_report_contains_required_fields(self):
        """Quality report should contain all required fields."""
        from app.services.preview_generator import generate_quality_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quality.json"
            metrics = {
                "psnr": 28.5,
                "ssim": 0.88,
                "depth_mae": 0.05,
                "depth_rmse": 0.08,
            }
            metadata = {
                "input_file": "test.zip",
                "duration": 125.3,
                "resolution": [512, 512],
            }

            report = generate_quality_report(
                job_id="test-job-123",
                model="reconviagen",
                metrics=metrics,
                metadata=metadata,
                output_path=output_path,
            )

            # Check required top-level fields
            assert "job_id" in report
            assert "model" in report
            assert "status" in report
            assert "summary" in report
            assert "metrics" in report
            assert "thresholds" in report
            assert "metadata" in report

            # Check values
            assert report["job_id"] == "test-job-123"
            assert report["model"] == "reconviagen"
            assert report["status"] == "normal"  # PSNR 28.5, SSIM 0.88 -> normal

            # Check metrics structure
            assert report["metrics"]["psnr"] == 28.5
            assert report["metrics"]["ssim"] == 0.88
            assert report["metrics"]["depth_mae"] == 0.05
            assert report["metrics"]["depth_rmse"] == 0.08

    def test_report_thresholds_present(self):
        """Report should include threshold values for reference."""
        from app.services.preview_generator import generate_quality_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quality.json"
            report = generate_quality_report(
                job_id="test-job",
                model="nvdiffrec",
                metrics={"psnr": 25.0, "ssim": 0.85, "depth_mae": 0.0, "depth_rmse": 0.0},
                metadata={},
                output_path=output_path,
            )

            thresholds = report["thresholds"]
            assert "psnr_normal" in thresholds
            assert "psnr_warning" in thresholds
            assert "ssim_normal" in thresholds
            assert "ssim_warning" in thresholds

            assert thresholds["psnr_normal"] == 25.0
            assert thresholds["psnr_warning"] == 20.0
            assert thresholds["ssim_normal"] == 0.85
            assert thresholds["ssim_warning"] == 0.75

    def test_report_saved_to_file(self):
        """Report should be saved as JSON file."""
        from app.services.preview_generator import generate_quality_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quality.json"
            generate_quality_report(
                job_id="test-job",
                model="reconviagen",
                metrics={"psnr": 30.0, "ssim": 0.90, "depth_mae": 0.0, "depth_rmse": 0.0},
                metadata={},
                output_path=output_path,
            )

            assert output_path.exists(), "quality.json should be created"

            with open(output_path) as f:
                saved_report = json.load(f)

            assert saved_report["job_id"] == "test-job"
            assert saved_report["status"] == "normal"

    def test_report_summary_human_readable(self):
        """Report summary should be human readable."""
        from app.services.preview_generator import generate_quality_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quality.json"
            report = generate_quality_report(
                job_id="test-job",
                model="nvdiffrec",
                metrics={"psnr": 22.0, "ssim": 0.80, "depth_mae": 0.0, "depth_rmse": 0.0},
                metadata={},
                output_path=output_path,
            )

            summary = report["summary"]
            assert "Warning" in summary or "warning" in summary.lower()
            assert "22" in summary or "22.0" in summary  # PSNR value
            assert "0.80" in summary  # SSIM value

    def test_report_metadata_includes_timestamp(self):
        """Report metadata should include timestamp."""
        from app.services.preview_generator import generate_quality_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quality.json"
            report = generate_quality_report(
                job_id="test-job",
                model="reconviagen",
                metrics={"psnr": 25.0, "ssim": 0.85, "depth_mae": 0.0, "depth_rmse": 0.0},
                metadata={"input_file": "images.zip", "duration": 60.0},
                output_path=output_path,
            )

            assert "timestamp" in report["metadata"]
            assert report["metadata"]["timestamp"].endswith("Z")  # UTC format


@requires_skimage
class TestDepthErrorComputation:
    """Test depth error metrics (MAE/RMSE)."""

    def test_depth_error_identical_maps(self):
        """Identical depth maps should have zero error."""
        from app.services.quality_metrics import compute_depth_error
        depth = np.random.rand(64, 64).astype(np.float32)
        depth[depth < 0.1] = 0  # Some background pixels
        result = compute_depth_error(depth, depth)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["valid_pixels"] > 0

    def test_depth_error_different_maps(self):
        """Different depth maps should have non-zero error."""
        from app.services.quality_metrics import compute_depth_error
        depth1 = np.ones((64, 64), dtype=np.float32) * 0.5
        depth2 = np.ones((64, 64), dtype=np.float32) * 0.6
        result = compute_depth_error(depth1, depth2)
        assert result["mae"] > 0
        assert result["rmse"] > 0
        assert result["mae"] == pytest.approx(0.1, rel=0.01)

    def test_depth_error_non_negative(self):
        """MAE and RMSE should always be non-negative."""
        from app.services.quality_metrics import compute_depth_error
        depth1 = np.random.rand(64, 64).astype(np.float32)
        depth1[depth1 < 0.2] = 0  # Some zeros
        depth2 = np.random.rand(64, 64).astype(np.float32)
        depth2[depth2 < 0.2] = 0  # Matching zeros pattern

        # Ensure at least some overlap
        depth1[30:40, 30:40] = 0.5
        depth2[30:40, 30:40] = 0.6

        result = compute_depth_error(depth1, depth2)
        assert result["mae"] >= 0
        assert result["rmse"] >= 0

    def test_depth_error_excludes_background(self):
        """Valid pixels count should exclude background (zero depth)."""
        from app.services.quality_metrics import compute_depth_error
        depth1 = np.ones((64, 64), dtype=np.float32)
        depth2 = np.ones((64, 64), dtype=np.float32)

        # Set half to background (zero)
        depth1[:32, :] = 0
        depth2[:32, :] = 0

        result = compute_depth_error(depth1, depth2)
        # Only non-zero pixels should be counted
        assert result["valid_pixels"] == 32 * 64

    def test_depth_error_no_valid_pixels(self):
        """Should raise error if no valid pixels for comparison."""
        from app.services.quality_metrics import compute_depth_error
        depth1 = np.zeros((64, 64), dtype=np.float32)
        depth2 = np.zeros((64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="No valid depth pixels"):
            compute_depth_error(depth1, depth2)

    def test_depth_error_shape_mismatch(self):
        """Should raise error for shape mismatch."""
        from app.services.quality_metrics import compute_depth_error
        depth1 = np.ones((64, 64), dtype=np.float32)
        depth2 = np.ones((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_depth_error(depth1, depth2)


@requires_skimage
class TestOverallMetrics:
    """Test overall metrics computation from multiple pairs."""

    def test_overall_metrics_single_pair(self):
        """Single pair should return its own metrics."""
        from app.services.quality_metrics import compute_overall_metrics
        img1 = np.random.rand(64, 64, 3).astype(np.float32)
        img2 = np.clip(img1 + 0.01, 0.0, 1.0)  # Tiny difference

        result = compute_overall_metrics([(img1, img2)])

        assert "psnr" in result
        assert "ssim" in result
        assert result["psnr"] > 30  # Should be high for tiny difference
        assert result["ssim"] > 0.95

    def test_overall_metrics_multiple_pairs(self):
        """Multiple pairs should be averaged."""
        from app.services.quality_metrics import compute_overall_metrics
        pairs = []
        for _ in range(4):
            img1 = np.random.rand(64, 64, 3).astype(np.float32)
            noise = np.random.randn(64, 64, 3).astype(np.float32) * 0.05
            img2 = np.clip(img1 + noise, 0.0, 1.0)
            pairs.append((img1, img2))

        result = compute_overall_metrics(pairs)

        assert np.isfinite(result["psnr"])
        assert 0.0 <= result["ssim"] <= 1.0

    def test_overall_metrics_empty_list(self):
        """Empty list should raise ValueError."""
        from app.services.quality_metrics import compute_overall_metrics
        with pytest.raises(ValueError, match="No image pairs"):
            compute_overall_metrics([])
