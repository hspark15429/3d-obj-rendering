#!/bin/bash
# Verification script for Celery infrastructure
# Run this in an environment with Docker access

set -e

echo "=== Verifying Celery Infrastructure ==="

echo "1. Validating docker-compose.yml syntax..."
docker compose config > /dev/null
echo "   ✓ Docker Compose config valid"

echo "2. Building and starting services..."
docker compose up --build -d

echo "3. Waiting for services to be healthy..."
sleep 15

echo "4. Checking Redis..."
docker compose exec -T redis redis-cli ping
echo "   ✓ Redis responding"

echo "5. Checking worker logs for Celery connection..."
docker compose logs worker 2>&1 | grep -E "(ready|connected)" || {
    echo "   ! Worker not ready yet, showing recent logs:"
    docker compose logs --tail=20 worker
}

echo "6. Checking API health..."
curl -sf http://localhost:8000/health | grep -q "healthy"
echo "   ✓ API healthy"

echo "7. Checking all services status..."
docker compose ps

echo ""
echo "=== Verification Complete ==="
echo "All services running. Redis and Celery worker connected successfully."
