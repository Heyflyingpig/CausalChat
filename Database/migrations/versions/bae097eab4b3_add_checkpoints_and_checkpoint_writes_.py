"""add checkpoints and checkpoint_writes tables

Revision ID: bae097eab4b3
Revises: 
Create Date: 2025-10-26 14:30:34.309644

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bae097eab4b3'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    """Upgrade schema."""
    op.execute("""
        CREATE TABLE checkpoints (
            thread_id VARCHAR(255) NOT NULL,
            checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
            checkpoint_id VARCHAR(255) NOT NULL,
            parent_checkpoint_id VARCHAR(255) DEFAULT NULL,
            checkpoint LONGBLOB NOT NULL,
            metadata_data JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id),
            INDEX idx_thread_time (thread_id, created_at DESC),
            INDEX idx_parent (parent_checkpoint_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    op.execute("""
        CREATE TABLE checkpoint_writes (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            thread_id VARCHAR(255) NOT NULL,
            checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
            checkpoint_id VARCHAR(255) NOT NULL,
            task_id VARCHAR(255) NOT NULL,
            idx INT NOT NULL,
            channel VARCHAR(255) NOT NULL,
            value LONGBLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            INDEX idx_checkpoint (thread_id, checkpoint_ns, checkpoint_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS checkpoint_writes")
    op.execute("DROP TABLE IF EXISTS checkpoints")
    pass
