"""Entry point for running the worker as a module.

Usage:
    cd backend
    python -m worker
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add app directory to path so we can import from app.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from worker.config import WorkerConfig
from worker.job_processor import JobProcessor


def setup_logging() -> None:
    """Configure logging for the worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main() -> None:
    """Main entry point for the worker."""
    setup_logging()
    logger = logging.getLogger("worker")

    config = WorkerConfig.from_env()
    logger.info(f"Starting worker {config.worker_id}")
    logger.info(f"Database: {config.database_url}")
    logger.info(f"Poll interval: {config.poll_interval_seconds}s")

    processor = JobProcessor(config)

    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig: int, frame) -> None:
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await processor.run(shutdown_event)
    except Exception as e:
        logger.exception(f"Worker crashed: {e}")
        sys.exit(1)

    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
