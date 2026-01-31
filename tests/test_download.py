"""Tests for download endpoint.

Tests validate that GET /jobs/{job_id}/download returns appropriate
status codes and structured error responses for various job states.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import zipfile
from io import BytesIO

from fastapi.testclient import TestClient
from app.main import app
from app.api.error_codes import ErrorCode

client = TestClient(app, raise_server_exceptions=False)


class TestDownloadEndpoint:
    """Test GET /jobs/{job_id}/download."""

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_download_not_found_job(self, mock_celery):
        """Test downloading non-existent job returns 404 with structured error."""
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/nonexistent/download")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.JOB_NOT_FOUND.value
        assert "suggestion" in data["error"]

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_download_started_job(self, mock_celery):
        """Test downloading just-started job returns 409 with structured error."""
        mock_result = MagicMock()
        mock_result.state = "STARTED"
        mock_result.info = None
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/started123/download")

        assert response.status_code == 409
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.JOB_NOT_READY.value
        assert "processing" in data["error"]["message"].lower()

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_download_progress_job(self, mock_celery):
        """Test downloading in-progress job returns 409 with progress info."""
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_result.info = {"progress": 50, "step": "inference"}
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/processing123/download")

        assert response.status_code == 409
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.JOB_NOT_READY.value
        assert "50%" in data["error"]["message"]
        assert data["error"]["details"]["progress"] == 50

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_download_failed_job(self, mock_celery):
        """Test downloading failed job returns 500 with error details."""
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.info = "Model inference failed: CUDA OOM"
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/failed123/download")

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.MODEL_FAILED.value
        assert "failed" in data["error"]["message"].lower()

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_download_cancelled_job(self, mock_celery):
        """Test downloading cancelled job returns 500."""
        mock_result = MagicMock()
        mock_result.state = "REVOKED"
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/cancelled123/download")

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.MODEL_FAILED.value
        assert "cancelled" in data["error"]["message"].lower()

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    @patch('app.api.jobs.get_job_path')
    def test_download_expired_job(self, mock_get_path, mock_celery):
        """Test downloading expired job (results deleted) returns 410."""
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_celery.AsyncResult.return_value = mock_result

        # Return path that doesn't exist
        mock_get_path.return_value = Path("/nonexistent/path/that/does/not/exist")

        response = client.get("/jobs/expired123/download")

        assert response.status_code == 410
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.JOB_EXPIRED.value
        assert "suggestion" in data["error"]

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    @patch('app.api.jobs.get_job_path')
    @patch('app.api.jobs.validate_job_outputs')
    def test_download_incomplete_results(self, mock_validate, mock_get_path, mock_celery):
        """Test downloading job with incomplete outputs returns 500."""
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_celery.AsyncResult.return_value = mock_result

        # Create temp directory with output dir but missing files
        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = Path(tmpdir)
            output_dir = job_path / "output"
            output_dir.mkdir(parents=True)

            mock_get_path.return_value = job_path
            mock_validate.return_value = (False, ["mesh.glb", "quality.json"])

            response = client.get("/jobs/incomplete123/download")

            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == ErrorCode.INCOMPLETE_RESULTS.value

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    @patch('app.api.jobs.get_job_path')
    @patch('app.api.jobs.validate_job_outputs')
    @patch('app.api.jobs.create_result_zip')
    def test_download_success(self, mock_zip, mock_validate, mock_get_path, mock_celery):
        """Test successful download returns ZIP file."""
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_celery.AsyncResult.return_value = mock_result

        # Create temp directory with output structure
        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = Path(tmpdir)
            output_dir = job_path / "output" / "reconviagen"
            output_dir.mkdir(parents=True)
            (output_dir / "mesh.glb").write_bytes(b"glb content")
            (output_dir / "quality.json").write_text('{"status": "normal"}')

            mock_get_path.return_value = job_path
            mock_validate.return_value = (True, [])

            # Return a valid ZIP buffer
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("reconviagen/mesh.glb", b"glb content")
                zf.writestr("reconviagen/quality.json", '{"status": "normal"}')
            zip_buffer.seek(0)
            mock_zip.return_value = zip_buffer

            response = client.get("/jobs/success123/download")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/zip"
            assert "attachment" in response.headers["content-disposition"]
            assert "success123.zip" in response.headers["content-disposition"]

            # Verify ZIP content is valid
            zip_content = BytesIO(response.content)
            with zipfile.ZipFile(zip_content, 'r') as zf:
                assert "reconviagen/mesh.glb" in zf.namelist()

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    @patch('app.api.jobs.get_job_path')
    @patch('app.api.jobs.validate_job_outputs')
    @patch('app.api.jobs.create_result_zip')
    def test_download_zip_creation_failure(self, mock_zip, mock_validate, mock_get_path, mock_celery):
        """Test ZIP creation failure returns 500 with structured error."""
        from app.services.result_packager import IncompleteResultsError

        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_celery.AsyncResult.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = Path(tmpdir)
            output_dir = job_path / "output"
            output_dir.mkdir(parents=True)

            mock_get_path.return_value = job_path
            mock_validate.return_value = (True, [])

            # Simulate ZIP creation failure
            mock_zip.side_effect = IncompleteResultsError(
                "ZIP creation failed",
                missing_items=["texture.png"]
            )

            response = client.get("/jobs/zipfail123/download")

            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == ErrorCode.INCOMPLETE_RESULTS.value


class TestDownloadErrorCodes:
    """Test that download errors include proper error codes and suggestions."""

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_error_response_structure(self, mock_celery):
        """Test that all error responses follow consistent structure."""
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/test/download")

        data = response.json()
        error = data["error"]

        # Verify all required fields present
        assert "code" in error
        assert "message" in error
        assert "details" in error
        assert "suggestion" in error

        # Verify code is from ErrorCode enum
        valid_codes = [e.value for e in ErrorCode]
        assert error["code"] in valid_codes

        # Verify suggestion is not empty
        assert len(error["suggestion"]) > 0

    @patch('app.main.gpu_state', {'initialized': True, 'name': 'Test GPU', 'driver_version': '1.0', 'memory_total_gb': 16})
    @patch('app.api.jobs.celery_app')
    def test_job_not_ready_includes_poll_suggestion(self, mock_celery):
        """Test JOB_NOT_READY error suggests polling for status."""
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_result.info = {"progress": 25}
        mock_celery.AsyncResult.return_value = mock_result

        response = client.get("/jobs/inprogress/download")

        data = response.json()
        suggestion = data["error"]["suggestion"]

        # Suggestion should mention polling status endpoint
        assert "poll" in suggestion.lower() or "status" in suggestion.lower()
