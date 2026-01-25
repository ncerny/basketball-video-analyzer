"""Job result importer for cloud GPU processing."""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.detection import PlayerDetection
from app.models.video import Video
from worker.cloud_storage import CloudStorage, JobManifest

logger = logging.getLogger(__name__)


def validate_detection(det: dict, index: int) -> dict | None:
    """Validate detection dict has required fields with correct types.

    Returns validated detection dict or None if invalid.
    """
    required_fields = ["frame", "track_id", "bbox", "confidence"]
    for field in required_fields:
        if field not in det:
            logger.warning(f"Detection {index} missing required field '{field}', skipping")
            return None

    bbox = det.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        logger.warning(f"Detection {index} has invalid bbox format, skipping")
        return None

    try:
        return {
            "frame": int(det["frame"]),
            "track_id": int(det["track_id"]) if det["track_id"] is not None else None,
            "bbox_x": float(bbox[0]),
            "bbox_y": float(bbox[1]),
            "bbox_width": float(bbox[2]),
            "bbox_height": float(bbox[3]),
            "confidence": float(det["confidence"]),
        }
    except (ValueError, TypeError) as e:
        logger.warning(f"Detection {index} has invalid data types ({e}), skipping")
        return None


async def import_job_results(
    storage: CloudStorage,
    manifest: JobManifest,
    session: AsyncSession,
    cleanup: bool = True,
) -> int:
    """Import completed job results into the database.

    Args:
        storage: CloudStorage instance.
        manifest: Job manifest (must be completed status).
        session: Database session.
        cleanup: Whether to delete R2 files after import.

    Returns:
        Number of detections imported.

    Raises:
        ValueError: If job is not completed or video not found.
    """
    if manifest.status != "completed":
        raise ValueError(f"Job {manifest.job_id} is not completed (status: {manifest.status})")

    # Verify video exists
    video_result = await session.execute(
        select(Video).where(Video.id == manifest.video_id)
    )
    video = video_result.scalar_one_or_none()
    if not video:
        raise ValueError(f"Video {manifest.video_id} not found in database")

    # Download results
    logger.info(f"Downloading results for job {manifest.job_id}...")
    results = storage.download_results(manifest.job_id)
    if not results:
        raise ValueError(f"No results found for job {manifest.job_id}")

    detections = results.get("detections", [])
    logger.info(f"Found {len(detections)} detections for job {manifest.job_id}")

    # Validate all detections
    validated_detections = []
    for i, det in enumerate(detections):
        validated = validate_detection(det, i)
        if validated:
            validated_detections.append(validated)

    if len(validated_detections) < len(detections):
        logger.warning(
            f"Validated {len(validated_detections)}/{len(detections)} detections "
            f"for job {manifest.job_id}"
        )

    # Insert detections
    for det in validated_detections:
        detection = PlayerDetection(
            video_id=manifest.video_id,
            frame_number=det["frame"],
            tracking_id=det["track_id"],
            bbox_x=det["bbox_x"],
            bbox_y=det["bbox_y"],
            bbox_width=det["bbox_width"],
            bbox_height=det["bbox_height"],
            confidence_score=det["confidence"],
        )
        session.add(detection)

    await session.commit()
    logger.info(f"Imported {len(validated_detections)} detections for job {manifest.job_id}")

    # Update manifest status
    manifest.status = "imported"
    storage.upload_job_manifest(manifest)

    # Cleanup R2 files
    if cleanup:
        logger.info(f"Cleaning up R2 files for job {manifest.job_id}...")
        storage.delete_job_files(manifest.job_id)

    return len(validated_detections)
