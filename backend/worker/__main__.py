"""Entry point for running the worker as a module.

Usage:
    # Local worker (polls SQLite DB)
    python -m worker

    # Cloud worker (polls R2)
    python -m worker --cloud

    # CLI commands
    python -m worker.cli submit --video-id 1 --video-path video.mp4
    python -m worker.cli status
    python -m worker.cli import-all
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add app directory to path so we can import from app.*
sys.path.insert(0, str(Path(__file__).parent.parent))


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


async def run_local_worker() -> None:
    """Run local worker that polls SQLite DB."""
    from worker.config import WorkerConfig
    from worker.job_processor import JobProcessor

    logger = logging.getLogger("worker")
    config = WorkerConfig.from_env()

    logger.info(f"Starting LOCAL worker {config.worker_id}")
    logger.info(f"Database: {config.database_url}")
    logger.info(f"Poll interval: {config.poll_interval_seconds}s")

    processor = JobProcessor(config)
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

    logger.info("Local worker shutdown complete")


async def run_cloud_worker() -> None:
    """Run cloud worker that polls R2."""
    from worker.cloud_worker import main as cloud_main
    await cloud_main()


def main() -> None:
    """Main entry point."""
    setup_logging()

    # Check for --cloud flag
    cloud_mode = "--cloud" in sys.argv

    if cloud_mode:
        asyncio.run(run_cloud_worker())
    else:
        asyncio.run(run_local_worker())


if __name__ == "__main__":
    main()
