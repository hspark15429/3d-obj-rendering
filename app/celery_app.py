"""Celery application factory for async task processing."""
from celery import Celery
from app.config import settings


def create_celery_app() -> Celery:
    """
    Create and configure Celery application instance.

    Uses factory pattern to avoid circular imports.
    Tasks should use @shared_task decorator, not @celery_app.task.
    """
    celery_app = Celery("3d_reconstruction")

    # Configure Celery
    celery_app.conf.update(
        broker_url=settings.CELERY_BROKER_URL,
        result_backend=settings.CELERY_RESULT_BACKEND,
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Long-running inference tasks need extended visibility timeout
        broker_transport_options={
            "visibility_timeout": 14400,  # 4 hours
        },
        # Fair task distribution - one task at a time per worker
        worker_prefetch_multiplier=1,
    )

    # Auto-discover tasks in app.tasks module
    celery_app.autodiscover_tasks(["app.tasks"])

    return celery_app


# Export singleton instance
celery_app = create_celery_app()
