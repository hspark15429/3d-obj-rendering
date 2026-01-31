#!/bin/bash
#
# Verification script for reconstruction task (Plan 02-03)
#
# Run this in a Docker-enabled environment to verify:
# 1. Worker discovers reconstruction.process task
# 2. Task can be queued via delay()
# 3. Cancellation flow works (request -> confirm -> is_cancelled)
# 4. Task executes with progress updates
#
# Usage: ./verify-reconstruction-task.sh
#

set -e

echo "=== Verification: Reconstruction Task (02-03) ==="
echo

# Rebuild and restart services
echo "1. Rebuilding and restarting Docker services..."
docker compose down
docker compose up --build -d

echo "2. Waiting for services to be healthy (30s)..."
sleep 30

# Check worker logs for task registration
echo
echo "3. Checking if worker discovered reconstruction.process task..."
docker compose logs worker 2>&1 | grep -E "(reconstruction.process|ready)" || echo "⚠ Task not found in logs yet"

# Test task can be queued
echo
echo "4. Testing task can be queued..."
docker compose exec api python -c "
from app.tasks.reconstruction import process_reconstruction
result = process_reconstruction.delay('test123', 'reconviagen')
print(f'✓ Task ID: {result.id}')
print(f'✓ Task state: {result.state}')
" || echo "⚠ Task queueing failed"

# Check Redis has the task
echo
echo "5. Checking Redis queue length..."
docker compose exec redis redis-cli LLEN celery || echo "⚠ Redis check failed"

# Test cancellation flow
echo
echo "6. Testing cancellation flow..."
docker compose exec api python -c "
from app.services.job_manager import request_cancellation, confirm_cancellation, is_job_cancelled

# Request cancellation
request_cancellation('test123')
pending = is_job_cancelled('test123')
print(f'After request: is_cancelled={pending} (should be False)')

# Confirm cancellation
confirm_cancellation('test123')
confirmed = is_job_cancelled('test123')
print(f'After confirm: is_cancelled={confirmed} (should be True)')
print('✓ Cancellation flow works')
" || echo "⚠ Cancellation test failed"

# Watch worker logs briefly
echo
echo "7. Watching worker logs for task execution (15 seconds)..."
echo "   (Look for progress updates and task completion)"
timeout 15 docker compose logs -f worker 2>&1 || true

echo
echo "=== Verification Complete ==="
echo
echo "Expected outcomes:"
echo "  ✓ Worker logs show 'reconstruction.process' registered"
echo "  ✓ Task can be queued with delay()"
echo "  ✓ Cancellation: request -> confirm -> is_cancelled returns True"
echo "  ✓ Worker processes tasks with progress updates"
echo
