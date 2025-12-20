"""add processing_batches table

Revision ID: 96312ae6fc68
Revises: c7d8a3dfd4cb
Create Date: 2025-12-20 15:54:52.146526

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "96312ae6fc68"
down_revision: Union[str, Sequence[str], None] = "c7d8a3dfd4cb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "processing_batches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("batch_index", sa.Integer(), nullable=False),
        sa.Column("frame_start", sa.Integer(), nullable=False),
        sa.Column("frame_end", sa.Integer(), nullable=False),
        sa.Column(
            "detection_status",
            sa.Enum("PENDING", "PROCESSING", "COMPLETED", "FAILED", "SKIPPED", name="batchstatus"),
            nullable=False,
        ),
        sa.Column(
            "ocr_status",
            sa.Enum("PENDING", "PROCESSING", "COMPLETED", "FAILED", "SKIPPED", name="batchstatus"),
            nullable=False,
        ),
        sa.Column("detection_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ocr_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_batch_video_index", "processing_batches", ["video_id", "batch_index"], unique=False
    )
    op.create_index(
        "ix_batch_video_status",
        "processing_batches",
        ["video_id", "detection_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_batch_video_status", table_name="processing_batches")
    op.drop_index("ix_batch_video_index", table_name="processing_batches")
    op.drop_table("processing_batches")
