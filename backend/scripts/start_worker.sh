#!/bin/bash
# Start the detection worker process
#
# This worker polls the database for pending detection jobs and processes them.
# Run this separately from the FastAPI backend to offload ML processing.
#
# Usage:
#   ./scripts/start_worker.sh
#
# Environment variables:
#   WORKER_POLL_INTERVAL - Seconds between polls (default: 2.0)
#   WORKER_MAX_ERRORS - Max consecutive errors before stopping (default: 5)
#   DATABASE_URL - Database connection URL (default: uses .env)

set -e

# Change to the backend directory
cd "$(dirname "$0")/.."

# Load environment from .env if it exists
if [ -f .env ]; then
    echo "Loading environment from .env"
    set -a
    source .env
    set +a
fi

# Activate virtual environment if it exists
if [ -d .venv ]; then
    echo "Activating virtual environment"
    source .venv/bin/activate
fi

echo "Starting detection worker..."
echo "  Database: ${DATABASE_URL:-sqlite+aiosqlite:///./basketball_analyzer.db}"
echo "  Poll interval: ${WORKER_POLL_INTERVAL:-2.0}s"
echo ""
echo "Press Ctrl+C to stop the worker gracefully."
echo ""

# Run the worker
python -m worker
