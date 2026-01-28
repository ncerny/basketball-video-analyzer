"""R2 cloud storage service for video files.

Provides permanent video storage in Cloudflare R2 with presigned URL
generation for direct browser streaming.
"""

import logging
from pathlib import Path

import boto3
from botocore.config import Config

from app.config import settings

logger = logging.getLogger(__name__)


class R2StorageService:
    """Service for managing video files in Cloudflare R2 storage."""

    def __init__(
        self,
        account_id: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket_name: str | None = None,
    ) -> None:
        """Initialize R2 storage service.

        Args:
            account_id: Cloudflare account ID. Defaults to settings.
            access_key_id: R2 access key ID. Defaults to settings.
            secret_access_key: R2 secret access key. Defaults to settings.
            bucket_name: R2 bucket name. Defaults to settings.
        """
        self._account_id = account_id or settings.r2_account_id
        self._access_key_id = access_key_id or settings.r2_access_key_id
        self._secret_access_key = secret_access_key or settings.r2_secret_access_key
        self._bucket = bucket_name or settings.r2_bucket_name

        if not self._account_id:
            logger.warning("R2 account ID not configured - R2 storage disabled")
            self._client = None
            return

        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{self._account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            config=Config(signature_version="s3v4"),
        )
        logger.info(f"R2StorageService initialized for bucket: {self._bucket}")

    @property
    def is_configured(self) -> bool:
        """Check if R2 storage is configured."""
        return self._client is not None

    def _generate_key(self, game_id: int, filename: str) -> str:
        """Generate R2 key for a video file.

        Key structure: videos/game_{game_id}/{filename}
        Preserves original file extension.

        Args:
            game_id: Game ID the video belongs to.
            filename: Original filename with extension.

        Returns:
            R2 object key.
        """
        return f"videos/game_{game_id}/{filename}"

    def upload_video(self, local_path: Path, game_id: int, filename: str) -> str:
        """Upload a video file to R2.

        Args:
            local_path: Local path to the video file.
            game_id: Game ID this video belongs to.
            filename: Filename to use in R2 (with extension).

        Returns:
            R2 key for the uploaded video.

        Raises:
            RuntimeError: If R2 is not configured.
            Exception: If upload fails.
        """
        if not self._client:
            raise RuntimeError("R2 storage is not configured")

        key = self._generate_key(game_id, filename)

        # Determine content type based on extension
        extension = Path(filename).suffix.lower()
        content_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".flv": "video/x-flv",
            ".wmv": "video/x-ms-wmv",
        }
        content_type = content_types.get(extension, "video/mp4")

        logger.info(f"Uploading video to R2: {key} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")

        try:
            self._client.upload_file(
                str(local_path),
                self._bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info(f"Video uploaded successfully: {key}")
            return key
        except Exception as e:
            logger.error(f"Failed to upload video to R2: {e}")
            raise

    def generate_presigned_url(self, r2_key: str, expires_in: int = 14400) -> str:
        """Generate a presigned URL for streaming a video.

        Args:
            r2_key: R2 object key for the video.
            expires_in: URL expiration time in seconds (default 4 hours).

        Returns:
            Presigned URL for direct video access.

        Raises:
            RuntimeError: If R2 is not configured.
        """
        if not self._client:
            raise RuntimeError("R2 storage is not configured")

        url = self._client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self._bucket,
                "Key": r2_key,
            },
            ExpiresIn=expires_in,
        )
        logger.debug(f"Generated presigned URL for {r2_key} (expires in {expires_in}s)")
        return url

    def delete_video(self, r2_key: str) -> None:
        """Delete a video from R2.

        Args:
            r2_key: R2 object key for the video.

        Raises:
            RuntimeError: If R2 is not configured.
        """
        if not self._client:
            raise RuntimeError("R2 storage is not configured")

        try:
            self._client.delete_object(Bucket=self._bucket, Key=r2_key)
            logger.info(f"Deleted video from R2: {r2_key}")
        except Exception as e:
            logger.error(f"Failed to delete video from R2: {e}")
            raise

    def video_exists(self, r2_key: str) -> bool:
        """Check if a video exists in R2.

        Args:
            r2_key: R2 object key for the video.

        Returns:
            True if video exists, False otherwise.
        """
        if not self._client:
            return False

        try:
            self._client.head_object(Bucket=self._bucket, Key=r2_key)
            return True
        except self._client.exceptions.ClientError:
            return False


# Singleton instance for dependency injection
_r2_service: R2StorageService | None = None


def get_r2_storage_service() -> R2StorageService:
    """Get the R2 storage service singleton.

    Returns:
        R2StorageService instance.
    """
    global _r2_service
    if _r2_service is None:
        _r2_service = R2StorageService()
    return _r2_service
