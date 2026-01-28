"""add processing_jobs table

Revision ID: 882ba92dc242
Revises: 96312ae6fc68
Create Date: 2026-01-21

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "882ba92dc242"
down_revision: Union[str, Sequence[str], None] = "96312ae6fc68"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.String(36), nullable=False),  # UUID
        sa.Column(
            "job_type",
            sa.Enum("VIDEO_DETECTION", "TRACK_REPROCESS", name="jobtype"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("PENDING", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED", name="jobstatus"),
            nullable=False,
        ),
        sa.Column("video_id", sa.Integer(), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("progress_current", sa.Integer(), nullable=False, default=0),
        sa.Column("progress_total", sa.Integer(), nullable=False, default=0),
        sa.Column("progress_message", sa.String(500), nullable=False, default=""),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_id", sa.String(100), nullable=True),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_job_status_created",
        "processing_jobs",
        ["status", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_job_video_id",
        "processing_jobs",
        ["video_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_job_video_id", table_name="processing_jobs")
    op.drop_index("ix_job_status_created", table_name="processing_jobs")
    op.drop_table("processing_jobs")
    # Drop the enums
    op.execute("DROP TYPE IF EXISTS jobstatus")
    op.execute("DROP TYPE IF EXISTS jobtype")
