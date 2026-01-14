"""Tests for SAM3 configuration."""

import pytest
from pathlib import Path


class TestSAM3Config:
    """Tests for SAM3 configuration settings."""

    def test_sam3_settings_exist(self) -> None:
        """Test that SAM3 settings are defined in config."""
        from app.config import settings

        assert hasattr(settings, "sam3_prompt")
        assert hasattr(settings, "sam3_confidence_threshold")
        assert hasattr(settings, "sam3_use_half_precision")
        assert hasattr(settings, "sam3_temp_frames_dir")

    def test_sam3_default_values(self) -> None:
        """Test SAM3 default configuration values."""
        from app.config import settings

        assert settings.sam3_prompt == "basketball player"
        assert settings.sam3_confidence_threshold == 0.25
        assert settings.sam3_use_half_precision is True
        assert isinstance(settings.sam3_temp_frames_dir, Path)

    def test_tracking_backend_includes_sam3(self) -> None:
        """Test that sam3 is a valid tracking backend option."""
        from app.config import Settings

        # Check the type annotation includes sam3
        tracking_field = Settings.model_fields["tracking_backend"]
        assert "sam3" in str(tracking_field.annotation)
