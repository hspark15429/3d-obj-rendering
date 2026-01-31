#!/usr/bin/env python3
"""Verify Celery configuration can be imported (requires dependencies installed)."""

import sys

try:
    print("Verifying app.config...")
    from app.config import settings
    print(f"  ✓ Settings loaded")
    print(f"  - CELERY_BROKER_URL: {settings.CELERY_BROKER_URL}")
    print(f"  - CELERY_RESULT_BACKEND: {settings.CELERY_RESULT_BACKEND}")
    print(f"  - REDIS_STATE_DB: {settings.REDIS_STATE_DB}")
    print(f"  - JOB_STORAGE_PATH: {settings.JOB_STORAGE_PATH}")

    print("\nVerifying app.celery_app...")
    from app.celery_app import celery_app
    print(f"  ✓ Celery app created")
    print(f"  - App name: {celery_app.main}")
    print(f"  - Broker URL: {celery_app.conf.broker_url}")
    print(f"  - Result backend: {celery_app.conf.result_backend}")
    print(f"  - Task track started: {celery_app.conf.task_track_started}")
    print(f"  - Worker prefetch multiplier: {celery_app.conf.worker_prefetch_multiplier}")

    print("\n✓ Configuration verified successfully")
    sys.exit(0)

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  (This is expected if dependencies aren't installed)")
    print("  Run 'pip install -r requirements.txt' first")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
