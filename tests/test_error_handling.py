"""Tests for structured error handling.

Tests validate that API endpoints return structured error responses
with code, message, details, and suggestion fields per ERR-01/02/03 requirements.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO

from fastapi.testclient import TestClient
from app.main import app
from app.api.error_codes import ErrorCode

client = TestClient(app, raise_server_exceptions=False)


class TestValidationErrors:
    """Test upload validation error responses."""

    def _create_png_content(self, size: int = 100) -> bytes:
        """Create fake PNG content with magic bytes."""
        # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
        return b'\x89PNG\r\n\x1a\n' + b'x' * size

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.validate_upload_files')
    @patch('app.api.jobs.save_job_files')
    @patch('app.api.jobs.process_reconstruction')
    def test_valid_upload_success(self, mock_task, mock_save, mock_validate):
        """Test that valid uploads succeed."""
        mock_validate.return_value = None
        mock_save.return_value = None
        mock_task.apply_async = MagicMock()

        # Create 6 views + 6 depths
        files = [("views", ("view.png", self._create_png_content(), "image/png")) for _ in range(6)]
        files += [("depth_renders", ("depth.png", self._create_png_content(), "image/png")) for _ in range(6)]

        response = client.post("/jobs/", files=files, data={"model_type": "reconviagen"})

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    def test_wrong_view_count_returns_structured_error(self):
        """Test submitting wrong number of view files returns structured error."""
        # Submit with 5 views instead of 6
        files = [("views", ("view.png", self._create_png_content(), "image/png")) for _ in range(5)]
        files += [("depth_renders", ("depth.png", self._create_png_content(), "image/png")) for _ in range(6)]

        response = client.post("/jobs/", files=files, data={"model_type": "reconviagen"})

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        error = data["error"]
        assert error["code"] == ErrorCode.INVALID_FILE_COUNT.value
        assert "suggestion" in error
        assert "6" in error["message"]

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    def test_wrong_depth_count_returns_structured_error(self):
        """Test submitting wrong number of depth files returns structured error."""
        # Submit with 6 views and 5 depths
        files = [("views", ("view.png", self._create_png_content(), "image/png")) for _ in range(6)]
        files += [("depth_renders", ("depth.png", self._create_png_content(), "image/png")) for _ in range(5)]

        response = client.post("/jobs/", files=files, data={"model_type": "reconviagen"})

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        error = data["error"]
        assert error["code"] == ErrorCode.INVALID_FILE_COUNT.value

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    def test_invalid_png_returns_structured_error(self):
        """Test submitting non-PNG file returns structured error."""
        # Create files with one being non-PNG (JPEG magic bytes)
        files = [("views", ("view.png", self._create_png_content(), "image/png")) for _ in range(5)]
        files.append(("views", ("view.png", b'\xFF\xD8\xFF\xE0' + b'x' * 100, "image/png")))  # JPEG
        files += [("depth_renders", ("depth.png", self._create_png_content(), "image/png")) for _ in range(6)]

        response = client.post("/jobs/", files=files, data={"model_type": "reconviagen"})

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        error = data["error"]
        assert error["code"] == ErrorCode.INVALID_FILE_FORMAT.value
        assert "suggestion" in error

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    def test_file_too_large_returns_structured_error(self):
        """Test submitting oversized file returns structured error."""
        # Create file > 20MB
        large_content = b'\x89PNG\r\n\x1a\n' + b'x' * (21 * 1024 * 1024)
        files = [("views", ("view.png", large_content, "image/png"))]
        files += [("views", ("view.png", self._create_png_content(), "image/png")) for _ in range(5)]
        files += [("depth_renders", ("depth.png", self._create_png_content(), "image/png")) for _ in range(6)]

        response = client.post("/jobs/", files=files, data={"model_type": "reconviagen"})

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        error = data["error"]
        assert error["code"] == ErrorCode.FILE_TOO_LARGE.value
        assert "suggestion" in error
        assert "20MB" in error["suggestion"]


class TestJobStateErrors:
    """Test job state error responses."""

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_cancel_completed_job_returns_error(self, mock_celery):
        """Test cancelling already completed job returns structured error."""
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_celery.AsyncResult.return_value = mock_result

        response = client.post("/jobs/test123/cancel", json={"confirm": True})

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        error = data["error"]
        assert error["code"] == ErrorCode.JOB_NOT_READY.value
        assert "SUCCESS" in error["message"]

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_cancel_failed_job_returns_error(self, mock_celery):
        """Test cancelling already failed job returns structured error."""
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_celery.AsyncResult.return_value = mock_result

        response = client.post("/jobs/test123/cancel", json={"confirm": True})

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.JOB_NOT_READY.value

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    @patch('app.api.jobs.confirm_cancellation')
    @patch('app.api.jobs.request_cancellation')
    def test_confirm_cancel_without_request_returns_error(self, mock_request, mock_confirm, mock_celery):
        """Test confirming cancellation without prior request returns error."""
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_celery.AsyncResult.return_value = mock_result
        mock_confirm.return_value = False  # No pending request

        response = client.post("/jobs/test123/cancel", json={"confirm": True})

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.VALIDATION_FAILED.value
        assert "pending" in data["error"]["message"].lower()


class TestJobStatusErrors:
    """Test job status endpoint error responses."""

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_failed_job_status_includes_error(self, mock_celery):
        """Test that failed job status includes error information."""
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.info = {"error": "Model crashed", "error_code": "MODEL_FAILED"}
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/failed123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] is not None
        assert "crashed" in data["error"].lower() or "model" in data["error"].lower()

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_processing_job_status_includes_progress(self, mock_celery):
        """Test that processing job status includes progress information."""
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_result.info = {"progress": 50, "model": "reconviagen", "step": "inference"}
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/processing123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] == 50
        assert data["current_model"] == "reconviagen"
