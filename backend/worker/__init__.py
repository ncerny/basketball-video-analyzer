"""Detection worker module.

Run as a separate process to handle ML processing jobs (SAM3 detection, etc.)
independently from the FastAPI backend.

Usage:
    python -m worker

Or use the convenience script:
    ./scripts/start_worker.sh
"""

from worker.config import WorkerConfig
from worker.job_processor import JobProcessor

__all__ = ["JobProcessor", "WorkerConfig"]
