"""Detection worker module.

Run as a separate process to handle ML processing jobs (SAM3 detection, etc.)
independently from the FastAPI backend.

Usage (local):
    python -m worker

Usage (cloud CLI):
    python -m worker.cli submit --video-id 1 --video-path /path/to/video.mp4
    python -m worker.cli status
    python -m worker.cli import-all

Or use the convenience script:
    ./scripts/start_worker.sh
"""

from worker.config import WorkerConfig
from worker.job_processor import JobProcessor

__all__ = ["JobProcessor", "WorkerConfig"]
