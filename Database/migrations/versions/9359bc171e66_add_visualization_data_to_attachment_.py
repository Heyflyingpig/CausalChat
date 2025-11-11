"""add_visualization_data_to_attachment_type

Revision ID: 9359bc171e66
Revises: bae097eab4b3
Create Date: 2025-11-10 13:05:20.074853

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9359bc171e66'
down_revision: Union[str, Sequence[str], None] = 'bae097eab4b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
    ALTER TABLE chat_attachments
    MODIFY COLUMN attachment_type
    ENUM('causal_graph', 'analysis_result', 'file_content', 'other', 'visualization')
    NOT NULL
    """)
    pass


def downgrade() -> None:
    """先移除数据，再移除结构

    """
     # 先检查是否有使用 visualization_data 的记录
    op.execute("""
        DELETE FROM chat_attachments 
        WHERE attachment_type = 'visualization'
    """)
    
    # 然后移除该枚举值
    op.execute("""
        ALTER TABLE chat_attachments 
        MODIFY COLUMN attachment_type 
        ENUM('causal_graph', 'analysis_result', 'file_content', 'other') 
        NOT NULL
    """)
    pass
