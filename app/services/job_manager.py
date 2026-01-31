"""
Job manager for tracking cancellation requests via Redis.

Implements two-step cancellation:
1. request_cancellation() - Sets cancel_request:{job_id} flag
2. confirm_cancellation() - Confirms request and sets cancel:{job_id} flag

Workers check is_job_cancelled() before each processing step.
"""
import redis
from app.config import settings


# Module-level cache for Redis client
_redis_client = None


def get_redis_client() -> redis.Redis:
    """
    Get Redis client for state storage (DB 1).

    Uses lazy initialization - creates client on first call and caches.
    Separate from Celery broker (DB 0).

    Returns:
        redis.Redis: Redis client connected to state DB
    """
    global _redis_client

    if _redis_client is None:
        # Parse Redis URL and connect to state DB (DB 1)
        _redis_client = redis.from_url(
            settings.REDIS_STATE_DB,
            decode_responses=True  # Return strings instead of bytes
        )

    return _redis_client


def request_cancellation(job_id: str) -> bool:
    """
    Request cancellation for a job (step 1 of 2).

    Sets cancel_request:{job_id} flag with 1 hour TTL.
    Does not immediately cancel - requires confirmation.

    Args:
        job_id: Job identifier

    Returns:
        True (always succeeds)
    """
    client = get_redis_client()
    client.setex(f"cancel_request:{job_id}", 3600, "pending")
    return True


def confirm_cancellation(job_id: str) -> bool:
    """
    Confirm cancellation request (step 2 of 2).

    Checks if cancel_request:{job_id} exists.
    If yes: sets cancel:{job_id} flag and deletes request flag.
    If no: returns False (nothing to confirm).

    Args:
        job_id: Job identifier

    Returns:
        True if cancellation confirmed, False if no pending request
    """
    client = get_redis_client()

    # Check if there's a pending request
    if client.exists(f"cancel_request:{job_id}"):
        # Confirm the cancellation
        client.setex(f"cancel:{job_id}", 3600, "1")
        # Remove the request flag
        client.delete(f"cancel_request:{job_id}")
        return True

    return False


def is_job_cancelled(job_id: str) -> bool:
    """
    Check if job is cancelled.

    Workers call this before each processing step to check if they should abort.
    Only returns True after cancellation has been confirmed.

    Args:
        job_id: Job identifier

    Returns:
        True if job is cancelled (confirmed), False otherwise
    """
    client = get_redis_client()
    return client.exists(f"cancel:{job_id}") > 0


def cancel_pending(job_id: str) -> bool:
    """
    Check if cancellation is requested but not confirmed.

    Useful for API to show "cancellation pending" status.

    Args:
        job_id: Job identifier

    Returns:
        True if cancel requested but not confirmed, False otherwise
    """
    client = get_redis_client()
    return client.exists(f"cancel_request:{job_id}") > 0


def clear_cancellation(job_id: str) -> None:
    """
    Clear all cancellation flags for a job.

    Called after job completes (success or failure) to clean up.
    Removes both cancel_request and cancel keys.

    Args:
        job_id: Job identifier
    """
    client = get_redis_client()
    client.delete(f"cancel_request:{job_id}", f"cancel:{job_id}")
