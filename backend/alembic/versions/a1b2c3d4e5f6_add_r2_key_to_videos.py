"""add r2_key to videos

Revision ID: a1b2c3d4e5f6
Revises: 882ba92dc242
Create Date: 2026-01-25

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "882ba92dc242"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("videos", sa.Column("r2_key", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("videos", "r2_key")
