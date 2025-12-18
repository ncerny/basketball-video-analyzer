"""add_title_description_to_annotations

Revision ID: 481788e17285
Revises: 3c2f03621f12
Create Date: 2025-12-14 15:01:06.189371

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '481788e17285'
down_revision: Union[str, Sequence[str], None] = '3c2f03621f12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Only add the new columns - skip alter_column for SQLite compatibility
    op.add_column('annotations', sa.Column('title', sa.String(), nullable=True))
    op.add_column('annotations', sa.Column('description', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('annotations', 'description')
    op.drop_column('annotations', 'title')
