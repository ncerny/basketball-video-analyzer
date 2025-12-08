"""Add timestamps to annotations table

Revision ID: f278c83bdd7b
Revises: 05191b487fe1
Create Date: 2025-12-07 21:56:27.153224

"""
from typing import Sequence, Union
from datetime import datetime

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f278c83bdd7b'
down_revision: Union[str, Sequence[str], None] = '05191b487fe1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # SQLite doesn't support adding NOT NULL columns with defaults directly
    # Use batch operations to recreate the table
    with op.batch_alter_table('annotations', schema=None) as batch_op:
        batch_op.add_column(sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.current_timestamp()))
        batch_op.add_column(sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.current_timestamp()))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('annotations', schema=None) as batch_op:
        batch_op.drop_column('updated_at')
        batch_op.drop_column('created_at')
