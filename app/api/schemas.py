"""
Pydantic schemas for API requests and responses.
"""
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    """Available reconstruction model types."""
    RECONVIAGEN = "reconviagen"
    NVDIFFREC = "nvdiffrec"
    BOTH = "both"


class JobSubmitResponse(BaseModel):
    """Response when a job is submitted."""
    job_id: str = Field(..., description="8-character job identifier")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Job status (always 'queued' on submit)")
    model_type: ModelType = Field(..., description="Model type for reconstruction")
    created_at: datetime = Field(..., description="Job creation timestamp (ISO 8601)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "abc123xy",
                "status": "queued",
                "model_type": "reconviagen",
                "created_at": "2026-01-31T14:30:00Z"
            }
        }
    }


class JobStatusResponse(BaseModel):
    """Response for job status queries."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Processing progress (0-100, only when processing)")
    current_model: Optional[str] = Field(None, description="Currently running model (when processing)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    error: Optional[str] = Field(None, description="Error message (only when failed)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "abc123xy",
                "status": "processing",
                "progress": 45,
                "current_model": "reconviagen",
                "created_at": "2026-01-31T14:30:00Z",
                "updated_at": "2026-01-31T14:32:15Z",
                "error": None
            }
        }
    }


class CancelRequest(BaseModel):
    """Request to cancel a job."""
    confirm: bool = Field(default=False, description="Must be true to confirm cancellation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "confirm": True
            }
        }
    }


class CancelResponse(BaseModel):
    """Response for cancel requests."""
    job_id: str = Field(..., description="Job identifier")
    status: Literal["cancel_requested", "cancelled"] = Field(..., description="Cancellation status")
    message: str = Field(..., description="Human-readable message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "abc123xy",
                "status": "cancelled",
                "message": "Job cancelled successfully"
            }
        }
    }


class ErrorResponse(BaseModel):
    """Error response format."""
    error: str = Field(..., description="Error type or code")
    detail: Optional[str] = Field(None, description="Detailed error message")
    fields: Optional[dict[str, str]] = Field(None, description="Field-level validation errors")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid file upload",
                "fields": {
                    "views": "Expected 6 files, got 5"
                }
            }
        }
    }
