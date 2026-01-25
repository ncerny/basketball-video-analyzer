"""RunPod API service for cloud GPU auto-scaling.

Manages RunPod pods for on-demand GPU processing:
- Start pod when jobs are submitted
- Stop pod when idle (no jobs for X minutes)
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PodStatus:
    """Status of a RunPod pod."""

    pod_id: str
    name: str
    status: str  # RUNNING, EXITED, CREATED, etc.
    gpu_type: str | None = None
    cost_per_hour: float | None = None


class RunPodService:
    """Service for managing RunPod GPU pods."""

    def __init__(
        self,
        api_key: str | None = None,
        template_name: str = "basketball-video-analyzer",
    ) -> None:
        """Initialize RunPod service.

        Args:
            api_key: RunPod API key. If None, reads from RUNPOD_API_KEY env var.
            template_name: Name of the pod template to use.
        """
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self._template_name = template_name
        self._initialized = False

        if not self._api_key:
            logger.warning(
                "RUNPOD_API_KEY not set. RunPod auto-scaling will be disabled."
            )

    def _ensure_initialized(self) -> bool:
        """Lazy initialize RunPod SDK."""
        if self._initialized:
            return True

        if not self._api_key:
            return False

        try:
            import runpod

            runpod.api_key = self._api_key
            self._initialized = True
            return True
        except ImportError:
            logger.warning("runpod package not installed")
            return False

    def get_pod_by_name(self, name: str | None = None) -> PodStatus | None:
        """Find a pod by name.

        Args:
            name: Pod name to search for. Defaults to template_name.

        Returns:
            PodStatus if found, None otherwise.
        """
        if not self._ensure_initialized():
            return None

        import runpod

        search_name = name or self._template_name

        try:
            pods = runpod.get_pods()
            for pod in pods:
                if pod.get("name") == search_name:
                    return PodStatus(
                        pod_id=pod["id"],
                        name=pod["name"],
                        status=pod.get("desiredStatus", "UNKNOWN"),
                        gpu_type=pod.get("machine", {}).get("gpuDisplayName"),
                        cost_per_hour=pod.get("costPerHr"),
                    )
        except Exception as e:
            logger.error(f"Failed to list RunPod pods: {e}")

        return None

    def get_pod_status(self, pod_id: str) -> PodStatus | None:
        """Get status of a specific pod.

        Args:
            pod_id: RunPod pod ID.

        Returns:
            PodStatus if found, None otherwise.
        """
        if not self._ensure_initialized():
            return None

        import runpod

        try:
            pod = runpod.get_pod(pod_id)
            if pod:
                return PodStatus(
                    pod_id=pod["id"],
                    name=pod["name"],
                    status=pod.get("desiredStatus", "UNKNOWN"),
                    gpu_type=pod.get("machine", {}).get("gpuDisplayName"),
                    cost_per_hour=pod.get("costPerHr"),
                )
        except Exception as e:
            logger.error(f"Failed to get pod {pod_id}: {e}")

        return None

    def is_pod_running(self, pod_id: str | None = None) -> bool:
        """Check if a pod is running.

        Args:
            pod_id: Specific pod ID, or None to search by template name.

        Returns:
            True if pod is running.
        """
        if pod_id:
            status = self.get_pod_status(pod_id)
        else:
            status = self.get_pod_by_name()

        return status is not None and status.status == "RUNNING"

    def start_pod(self, pod_id: str | None = None) -> bool:
        """Start (resume) a stopped pod.

        Args:
            pod_id: Specific pod ID, or None to find by template name.

        Returns:
            True if pod was started successfully.
        """
        if not self._ensure_initialized():
            logger.warning("RunPod not initialized, cannot start pod")
            return False

        import runpod

        # Find pod if not specified
        if not pod_id:
            pod = self.get_pod_by_name()
            if not pod:
                logger.warning(
                    f"No pod found with name '{self._template_name}'. "
                    "Create one manually in RunPod console first."
                )
                return False
            pod_id = pod.pod_id

        try:
            # Check current status
            status = self.get_pod_status(pod_id)
            if status and status.status == "RUNNING":
                logger.info(f"Pod {pod_id} is already running")
                return True

            logger.info(f"Starting RunPod pod {pod_id}...")
            runpod.resume_pod(pod_id)
            logger.info(f"Pod {pod_id} start requested")
            return True

        except Exception as e:
            logger.error(f"Failed to start pod {pod_id}: {e}")
            return False

    def stop_pod(self, pod_id: str | None = None) -> bool:
        """Stop a running pod.

        Args:
            pod_id: Specific pod ID, or None to find by template name.

        Returns:
            True if pod was stopped successfully.
        """
        if not self._ensure_initialized():
            return False

        import runpod

        # Find pod if not specified
        if not pod_id:
            pod = self.get_pod_by_name()
            if not pod:
                logger.debug(f"No pod found with name '{self._template_name}'")
                return False
            pod_id = pod.pod_id

        try:
            status = self.get_pod_status(pod_id)
            if status and status.status != "RUNNING":
                logger.info(f"Pod {pod_id} is already stopped (status: {status.status})")
                return True

            logger.info(f"Stopping RunPod pod {pod_id}...")
            runpod.stop_pod(pod_id)
            logger.info(f"Pod {pod_id} stop requested")
            return True

        except Exception as e:
            logger.error(f"Failed to stop pod {pod_id}: {e}")
            return False

    def ensure_pod_running(self) -> bool:
        """Ensure a pod is running, starting it if needed.

        Returns:
            True if a pod is running (or was started).
        """
        if self.is_pod_running():
            logger.debug("Pod is already running")
            return True

        return self.start_pod()

    def get_status_summary(self) -> dict[str, Any]:
        """Get a summary of RunPod status for API responses.

        Returns:
            Dict with pod status info.
        """
        if not self._ensure_initialized():
            return {
                "enabled": False,
                "reason": "RUNPOD_API_KEY not configured",
            }

        pod = self.get_pod_by_name()
        if not pod:
            return {
                "enabled": True,
                "pod_found": False,
                "template_name": self._template_name,
            }

        return {
            "enabled": True,
            "pod_found": True,
            "pod_id": pod.pod_id,
            "pod_name": pod.name,
            "status": pod.status,
            "gpu_type": pod.gpu_type,
            "cost_per_hour": pod.cost_per_hour,
        }


# Singleton instance for easy access
_runpod_service: RunPodService | None = None


def get_runpod_service() -> RunPodService:
    """Get the singleton RunPod service instance."""
    global _runpod_service
    if _runpod_service is None:
        _runpod_service = RunPodService()
    return _runpod_service
