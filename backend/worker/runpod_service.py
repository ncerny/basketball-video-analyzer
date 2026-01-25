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
                    # Check actual runtime status, not just desiredStatus
                    # Runtime contains actual state info when pod is live
                    runtime = pod.get("runtime")
                    if runtime:
                        # Pod has runtime info - check actual uptime
                        uptime = runtime.get("uptimeInSeconds", 0)
                        actual_status = "RUNNING" if uptime > 0 else "STARTING"
                    else:
                        # No runtime = not actually running (crashed, exited, etc.)
                        actual_status = pod.get("desiredStatus", "UNKNOWN")
                        if actual_status == "RUNNING":
                            # Desired is RUNNING but no runtime = crashed/not started
                            actual_status = "NOT_RUNNING"

                    logger.debug(
                        f"Pod {pod['id']}: desired={pod.get('desiredStatus')}, "
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
        """Check if any of our pods are running.

        Returns:
            True if at least one pod is running.
        """
        pods = self.get_our_pods()
        return any(p.status == "RUNNING" for p in pods)

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

                # Build env vars - R2 credentials + SAM3 settings
                # Note: This may override template env vars, so include everything needed
                pod_env = {
                    # R2 credentials (secrets - not in template)
                    "R2_ACCOUNT_ID": settings.r2_account_id,
                    "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
                    "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
                    "R2_BUCKET_NAME": settings.r2_bucket_name,
                    # SAM3 settings
                    "SAM3_MEMORY_WINDOW_SIZE": str(settings.sam3_memory_window_size),
                    "SAM3_USE_TORCH_COMPILE": str(settings.sam3_use_torch_compile).lower(),
                    "SAM3_CONFIDENCE_THRESHOLD": str(settings.sam3_confidence_threshold),
                }

                pod = runpod.create_pod(
                    name=pod_name,
                    image_name="",  # Uses template's image
                    gpu_type_id=gpu_type,
                    template_id=self._template_id,
                    container_disk_in_gb=settings.runpod_container_disk_gb,
                    volume_in_gb=settings.runpod_volume_disk_gb if settings.runpod_volume_disk_gb > 0 else None,
                    env=pod_env,
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
