"""Worker configuration."""

import os
from dataclasses import dataclass


@dataclass
class WorkerConfig:
    """Configuration for the detection worker."""

    # Polling settings
    poll_interval_seconds: float = 2.0
    max_consecutive_errors: int = 5

    # Worker identification
    worker_id: str = ""

    # Database connection (reuse backend's database)
    database_url: str = ""

    # Graceful shutdown timeout
    shutdown_timeout_seconds: float = 30.0

    # Idle shutdown (for cloud GPU cost savings)
    # Stop pod after this many seconds of no jobs (0 = disabled)
    idle_shutdown_seconds: float = 300.0  # 5 minutes default

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """Load config from environment variables."""
        import socket
        import uuid

        # Generate a unique worker ID if not provided
        worker_id = os.environ.get(
            "WORKER_ID",
            f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}",
        )

        # Use same database as backend
        database_url = os.environ.get(
            "DATABASE_URL",
            "sqlite+aiosqlite:///./basketball_analyzer.db",
        )

        return cls(
            poll_interval_seconds=float(os.environ.get("WORKER_POLL_INTERVAL", "2.0")),
            max_consecutive_errors=int(os.environ.get("WORKER_MAX_ERRORS", "5")),
            worker_id=worker_id,
            database_url=database_url,
            shutdown_timeout_seconds=float(os.environ.get("WORKER_SHUTDOWN_TIMEOUT", "30.0")),
            idle_shutdown_seconds=float(os.environ.get("WORKER_IDLE_SHUTDOWN", "300.0")),
        )
