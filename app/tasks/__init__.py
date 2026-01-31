"""
Celery tasks module.

Tasks are auto-discovered by celery_app.autodiscover_tasks(['app.tasks']).
"""
# Import tasks to ensure they're registered with Celery
from app.tasks.reconstruction import process_reconstruction

__all__ = ["process_reconstruction"]
