"""Tests for SAM3 detection pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ml.types import BoundingBox, Detection, FrameDetections


class TestSAM3DetectionPipeline:
    """Tests for SAM3DetectionPipeline class."""

    def test_init_creates_pipeline(self) -> None:
        """Test that initialization creates pipeline instance."""
        from app.services.sam3_detection_pipeline import SAM3DetectionPipeline

        pipeline = SAM3DetectionPipeline()
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_process_video_yields_detections(self) -> None:
        """Test that process_video yields FrameDetections."""
        from app.services.sam3_detection_pipeline import SAM3DetectionPipeline

        pipeline = SAM3DetectionPipeline()

        # Mock the tracker
        mock_detections = [
            FrameDetections(
                frame_number=0,
                detections=[
                    Detection(
                        bbox=BoundingBox(10, 20, 100, 200),
                        confidence=0.9,
                        class_id=0,
                        class_name="person",
                        tracking_id=1,
                    )
                ],
                frame_width=1920,
                frame_height=1080,
            )
        ]

        with patch.object(
            pipeline, "_tracker"
        ) as mock_tracker:
            mock_tracker.process_video.return_value = iter(mock_detections)

            results = []
            async for frame_det in pipeline.process_video(Path("fake.mp4")):
                results.append(frame_det)

            assert len(results) == 1
            assert results[0].frame_number == 0
            assert results[0].detections[0].tracking_id == 1
