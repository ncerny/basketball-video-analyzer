"""RunPod API service for cloud GPU auto-scaling.

Manages RunPod pods for on-demand GPU processing:
- Create pod when jobs are submitted (tries GPU types in preference order)
- Terminate pod when idle (worker self-terminates after no jobs)
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from app.config import settings

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

    # Default GPU preferences if not configured
    DEFAULT_GPU_PREFERENCES = [
        "NVIDIA RTX 4090",
        "NVIDIA RTX A5000",
        "NVIDIA RTX 3090",
        "NVIDIA RTX A4000",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        template_id: str | None = None,
    ) -> None:
        """Initialize RunPod service.

        Args:
            api_key: RunPod API key. If None, reads from settings/env.
            template_id: RunPod template ID with our Docker image.
        """
        self._api_key = api_key or settings.runpod_api_key or os.environ.get("RUNPOD_API_KEY")
        self._template_id = template_id or settings.runpod_template_id or os.environ.get("RUNPOD_TEMPLATE_ID")
        self._initialized = False

        # Parse GPU preferences
        if settings.runpod_gpu_preferences:
            self._gpu_preferences = [
                g.strip() for g in settings.runpod_gpu_preferences.split(",")
            ]
        else:
            self._gpu_preferences = self.DEFAULT_GPU_PREFERENCES

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

    def get_our_pods(self) -> list[PodStatus]:
        """Get all pods created by this service.

        Returns:
            List of PodStatus for pods with our naming pattern.
        """
        if not self._ensure_initialized():
            return []

        import runpod

        pods = []
        try:
            all_pods = runpod.get_pods()
            for pod in all_pods:
                name = pod.get("name", "")
                # Our pods are named "bva-worker-{timestamp}"
                if name.startswith("bva-worker-"):
                    # Determine actual pod status from runtime and desiredStatus
                    runtime = pod.get("runtime")
                    desired = pod.get("desiredStatus", "UNKNOWN")

                    if runtime:
                        # Pod has runtime info - it's alive
                        uptime = runtime.get("uptimeInSeconds", 0)
                        actual_status = "RUNNING" if uptime > 0 else "STARTING"
                    elif desired == "RUNNING":
                        # Desired RUNNING but no runtime yet = starting up or crashed
                        # Check if pod was recently created (within last 5 min)
                        # For now, treat as STARTING to prevent duplicate pod creation
                        actual_status = "STARTING"
                    elif desired == "EXITED":
                        actual_status = "EXITED"
                    else:
                        actual_status = desired

                    logger.debug(
                        f"Pod {pod['id']}: desired={desired}, "
                        f"runtime={runtime is not None}, actual={actual_status}"
                    )

                    pods.append(PodStatus(
                        pod_id=pod["id"],
                        name=name,
                        status=actual_status,
                        gpu_type=pod.get("machine", {}).get("gpuDisplayName"),
                        cost_per_hour=pod.get("costPerHr"),
                    ))
        except Exception as e:
            logger.error(f"Failed to list RunPod pods: {e}")

        return pods

    def is_pod_running(self) -> bool:
        """Check if any of our pods are running or starting.

        Returns:
            True if at least one pod is running or being started.
        """
        pods = self.get_our_pods()
        # Include STARTING to prevent race condition where multiple pods get created
        return any(p.status in ("RUNNING", "STARTING") for p in pods)

    def get_running_pod(self) -> PodStatus | None:
        """Get a running pod if one exists.

        Returns:
            PodStatus of a running pod, or None.
        """
        pods = self.get_our_pods()
        for p in pods:
            if p.status == "RUNNING":
                return p
        return None

    def start_pod(self) -> bool:
        """Create and start a new pod.

        Tries GPU types in preference order until one is available.

        Returns:
            True if pod was created successfully.
        """
        if not self._ensure_initialized():
            logger.warning("RunPod not initialized, cannot start pod")
            return False

        if not self._template_id:
            logger.warning(
                "RUNPOD_TEMPLATE_ID not set. Create a template in RunPod console "
                "with your Docker image and set the template ID."
            )
            return False

        # Check if we already have a running pod
        if self.is_pod_running():
            logger.info("A worker pod is already running")
            return True

        import runpod

        # Generate unique pod name
        pod_name = f"bva-worker-{int(time.time())}"

        # Try each GPU type in preference order
        for gpu_type in self._gpu_preferences:
            try:
                logger.info(f"Attempting to create pod with {gpu_type}...")

                # All env vars (R2 credentials, SAM3 settings) are configured in the
                # RunPod template. This keeps secrets out of the API and gives the
                # template full control over pod configuration.
                pod = runpod.create_pod(
                    name=pod_name,
                    image_name="",  # Uses template's image
                    gpu_type_id=gpu_type,
                    template_id=self._template_id,
                    container_disk_in_gb=settings.runpod_container_disk_gb,
                    volume_in_gb=settings.runpod_volume_disk_gb if settings.runpod_volume_disk_gb > 0 else None,
                )

                if pod and pod.get("id"):
                    logger.info(
                        f"Created pod {pod['id']} with {gpu_type} "
                        f"(${pod.get('costPerHr', '?')}/hr)"
                    )
                    return True

            except Exception as e:
                error_msg = str(e).lower()
                if "no available" in error_msg or "insufficient" in error_msg:
                    logger.info(f"{gpu_type} not available, trying next...")
                    continue
                else:
                    logger.error(f"Failed to create pod with {gpu_type}: {e}")
                    continue

        logger.error("Failed to create pod - no GPU types available")
        return False

    def stop_pod(self, pod_id: str | None = None) -> bool:
        """Terminate a pod (not just stop - fully delete it).

        Args:
            pod_id: Specific pod ID, or None to terminate all our pods.

        Returns:
            True if pod(s) were terminated successfully.
        """
        if not self._ensure_initialized():
            return False

        import runpod

        if pod_id:
            pods_to_terminate = [pod_id]
        else:
            pods_to_terminate = [p.pod_id for p in self.get_our_pods()]

        if not pods_to_terminate:
            logger.debug("No pods to terminate")
            return True

        success = True
        for pid in pods_to_terminate:
            try:
                logger.info(f"Terminating pod {pid}...")
                runpod.terminate_pod(pid)
                logger.info(f"Pod {pid} terminated")
            except Exception as e:
                logger.error(f"Failed to terminate pod {pid}: {e}")
                success = False

        return success

    def cleanup_old_pods(self) -> int:
        """Terminate any pods that are not running (stuck, exited, etc).

        Returns:
            Number of pods cleaned up.
        """
        if not self._ensure_initialized():
            return 0

        import runpod

        cleaned = 0
        for pod in self.get_our_pods():
            if pod.status not in ("RUNNING", "CREATED"):
                try:
                    logger.info(f"Cleaning up {pod.status} pod {pod.pod_id}")
                    runpod.terminate_pod(pod.pod_id)
                    cleaned += 1
                except Exception as e:
                    logger.error(f"Failed to cleanup pod {pod.pod_id}: {e}")

        return cleaned

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

        if not self._template_id:
            return {
                "enabled": False,
                "reason": "RUNPOD_TEMPLATE_ID not configured",
            }

        pods = self.get_our_pods()
        running = [p for p in pods if p.status == "RUNNING"]

        return {
            "enabled": True,
            "template_id": self._template_id,
            "gpu_preferences": self._gpu_preferences,
            "total_pods": len(pods),
            "running_pods": len(running),
            "pods": [
                {
                    "pod_id": p.pod_id,
                    "name": p.name,
                    "status": p.status,
                    "gpu_type": p.gpu_type,
                    "cost_per_hour": p.cost_per_hour,
                }
                for p in pods
            ],
        }


# Singleton instance for easy access
_runpod_service: RunPodService | None = None


def get_runpod_service() -> RunPodService:
    """Get the singleton RunPod service instance."""
    global _runpod_service
    if _runpod_service is None:
        _runpod_service = RunPodService()
    return _runpod_service
