#!/usr/bin/env python3
"""
Static verification for reconstruction task structure (Plan 02-03).

Verifies code structure without requiring Docker runtime:
1. Job manager has all required functions
2. Task has correct decorator and imports
3. Task includes cancellation checks
4. Task includes progress updates
5. Docker configuration is valid
"""

import ast
import sys
from pathlib import Path

def check_file_exists(path: str) -> bool:
    """Check if file exists."""
    if not Path(path).exists():
        print(f"✗ Missing: {path}")
        return False
    print(f"✓ Exists: {path}")
    return True

def check_job_manager():
    """Verify job_manager.py structure."""
    print("\n=== Job Manager (app/services/job_manager.py) ===")

    if not check_file_exists("app/services/job_manager.py"):
        return False

    with open("app/services/job_manager.py") as f:
        content = f.read()

    required_functions = [
        "get_redis_client",
        "request_cancellation",
        "confirm_cancellation",
        "is_job_cancelled",
        "cancel_pending",
        "clear_cancellation"
    ]

    for func in required_functions:
        if f"def {func}" in content:
            print(f"✓ Function: {func}")
        else:
            print(f"✗ Missing function: {func}")
            return False

    # Check Redis DB 1 usage
    if "REDIS_STATE_DB" in content:
        print("✓ Uses REDIS_STATE_DB (separate DB 1)")
    else:
        print("✗ Missing REDIS_STATE_DB reference")
        return False

    # Check TTL usage
    if "3600" in content or "setex" in content:
        print("✓ Uses TTL (setex or explicit timeout)")
    else:
        print("✗ Missing TTL for cancellation keys")
        return False

    return True

def check_reconstruction_task():
    """Verify reconstruction.py structure."""
    print("\n=== Reconstruction Task (app/tasks/reconstruction.py) ===")

    if not check_file_exists("app/tasks/reconstruction.py"):
        return False

    with open("app/tasks/reconstruction.py") as f:
        content = f.read()

    # Check shared_task decorator
    if '@shared_task(bind=True, name="reconstruction.process")' in content:
        print("✓ Uses @shared_task(bind=True, name='reconstruction.process')")
    else:
        print("✗ Missing or incorrect @shared_task decorator")
        return False

    # Check imports
    required_imports = [
        ("is_job_cancelled", "app.services.job_manager"),
        ("clear_cancellation", "app.services.job_manager"),
        ("delete_job_files", "app.services.file_handler"),
    ]

    for item, source in required_imports:
        if item in content and source in content:
            print(f"✓ Imports: {item} from {source}")
        else:
            print(f"✗ Missing import: {item} from {source}")
            return False

    # Check cancellation checking
    if "is_job_cancelled(job_id)" in content:
        print("✓ Checks for cancellation")
    else:
        print("✗ Missing cancellation check")
        return False

    # Check progress updates
    if 'self.update_state(state="PROGRESS"' in content or "self.update_state(\n            state=\"PROGRESS\"" in content:
        print("✓ Reports progress via update_state")
    else:
        print("✗ Missing progress updates")
        return False

    # Check file cleanup
    if "delete_job_files(job_id)" in content:
        print("✓ Cleans up files on cancellation")
    else:
        print("✗ Missing file cleanup")
        return False

    # Check clear_cancellation
    if "clear_cancellation(job_id)" in content:
        print("✓ Clears cancellation flags after completion")
    else:
        print("✗ Missing clear_cancellation call")
        return False

    return True

def check_docker_config():
    """Verify Docker configuration includes worker service."""
    print("\n=== Docker Configuration (docker-compose.yml) ===")

    if not check_file_exists("docker-compose.yml"):
        return False

    with open("docker-compose.yml") as f:
        content = f.read()

    # Check worker service exists
    if "worker:" in content:
        print("✓ Worker service defined")
    else:
        print("✗ Missing worker service")
        return False

    # Check worker command includes celery
    if "celery -A app.celery_app worker" in content:
        print("✓ Worker command includes Celery worker")
    else:
        print("✗ Missing Celery worker command")
        return False

    # Check worker has Redis connection
    if "REDIS_STATE_DB" in content:
        print("✓ Worker has REDIS_STATE_DB environment variable")
    else:
        print("✗ Missing REDIS_STATE_DB for worker")
        return False

    return True

def main():
    """Run all verifications."""
    print("=== Static Verification: Reconstruction Task (02-03) ===\n")

    results = [
        check_job_manager(),
        check_reconstruction_task(),
        check_docker_config(),
    ]

    print("\n=== Summary ===")
    if all(results):
        print("✓ All static checks passed")
        print("\nNext step: Run verify-reconstruction-task.sh in Docker environment")
        return 0
    else:
        print("✗ Some checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
