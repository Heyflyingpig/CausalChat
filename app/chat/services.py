'''

app.chat.services - 聊天服务

- 获取聊天记录
'''
from app.db import get_db_connection
import mysql.connector
import logging
import json
from datetime import datetime
def get_chat_history(session_id: str, user_id: int, limit: int) -> list:
    """从数据库获取指定会话的最近聊天记录。"""
    history = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            # 获取最近的 'limit' 条记录
            # 为什么这里需要先反转，再反转排序呢？
            # 我需要获取一个子集，也就是所有记录中的最新的子集，然后在从老到新进行排序
            # 最后通过一个append,从老到新进行添加
            
            cursor.execute("""
                SELECT message_type, content FROM chat_messages
                WHERE session_id = %s AND user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (session_id, user_id, limit))
            recent_chats = cursor.fetchall()

            # 按时间倒序获取，所以要反转回来才是正确的对话顺序
            for row in reversed(recent_chats):
                role = "user" if row['message_type'] == 'user' else "assistant"
                history.append({"role": role, "content": row['content']})
            
            logging.info(f"为会话 {session_id} 获取了 {len(history)} 条历史消息。")
            return history
            
    except mysql.connector.Error as e:
        logging.error(f"为会话 {session_id} 获取历史记录时数据库出错: {e}")
        return []
    except Exception as e:
        logging.error(f"为会话 {session_id} 获取历史记录时发生未知错误: {e}")
        return []
    


## 保存历史文件     
def save_chat(user_id, session_id, user_msg, ai_response):
    """
    将用户和AI的交互保存到新的优化数据库结构中。
    - 采用延迟创建策略：如果session不存在，则在第一条消息时创建
    - 在 chat_messages 中为用户和AI分别创建记录。
    - 如果AI响应包含附件，则在 chat_attachments 中创建记录。
    - 更新 sessions 表的元数据。
    """
    timestamp_dt = datetime.now()

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            #  核心修改：实现延迟session创建逻辑 
            # 0. 检查session是否存在，如果不存在则创建
            cursor.execute("SELECT message_count, title FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            session_data = cursor.fetchone()
            
            if not session_data:
                # Session不存在，创建新的session记录（延迟创建）
                new_title = user_msg[:8] + ("..." if len(user_msg) > 8 else "")
                cursor.execute("""
                    INSERT INTO sessions (id, user_id, title, created_at, last_activity_at, message_count)
                    VALUES (%s, %s, %s, %s, %s, 0)
                """, (session_id, user_id, new_title, timestamp_dt, timestamp_dt))
                is_first_message = True
                logging.info(f"延迟创建session记录: {session_id} (用户: {user_id}, 标题: '{new_title}')")
            else:
                # Session已存在，判断是否为第一条消息
                is_first_message = session_data['message_count'] == 0
            
            # 保存用户消息
            sql_user = """
                INSERT INTO chat_messages (session_id, user_id, message_type, content, created_at)
                VALUES (%s, %s, 'user', %s, %s)
            """
            cursor.execute(sql_user, (session_id, user_id, user_msg, timestamp_dt))
            
            # 保存AI消息
            ai_content = ""
            attachment_content = None
            attachment_type = 'other'
            attachment_to_save = []

            if isinstance(ai_response, dict):
                # 确保 summary 键存在且不为 None
                ai_content = ai_response.get('summary')
                
                if ai_content is None:
                    # 如果 summary 为空，将整个响应序列化为字符串作为备用
                    ai_content = json.dumps(ai_response, ensure_ascii=False)
                
                if ai_response.get('type') == 'causal_graph' and 'data' in ai_response:
                    attachment_type = 'causal_graph'
                    attachment_content = json.dumps(ai_response, ensure_ascii=False) # 保存完整响应
                    attachment_to_save.append({
                        "type": attachment_type,
                        "content": attachment_content
                    })
                
                if ai_response.get('visualization_mapping'):
                    attachment_type = 'visualization'
                    attachment_content = json.dumps(ai_response.get('visualization_mapping'), ensure_ascii=False)
                    attachment_to_save.append({
                        "type": attachment_type,
                        "content": attachment_content
                    })

            elif isinstance(ai_response, str):
                ai_content = ai_response
            else:
                ai_content = json.dumps(ai_response, ensure_ascii=False)
            
            # 处理数据库保存格式
            sql_ai = """
                INSERT INTO chat_messages (session_id, user_id, message_type, content, has_attachment, created_at)
                VALUES (%s, %s, 'ai', %s, %s, %s)
            """
            has_attachment = len(attachment_to_save) > 0
            cursor.execute(sql_ai, (session_id, user_id, ai_content, has_attachment, timestamp_dt))
            ai_message_id = cursor.lastrowid # 获取AI消息的ID，用于关联附件

            # 3. 如果有附件，保存到 chat_attachments
            if has_attachment and attachment_to_save:
                for attachment in attachment_to_save:
                    sql_attachment = """
                    INSERT INTO chat_attachments (message_id, attachment_type, content, created_at)
                    VALUES (%s, %s, %s, %s)
                    """
                    logging.info(f"准备保存附件: type={attachment['type']}, content_size={len(attachment['content'])} 字节")
                    cursor.execute(sql_attachment,
                    (ai_message_id, attachment['type'], attachment['content'], timestamp_dt))
                    logging.info(f"成功保存附件: {attachment['type']}")
            

            # 根据是否为第一条消息，决定是否更新标题
            if is_first_message:
                #  更新会话，包括新标题（或确认创建时的标题）
                new_title = user_msg[:8] # 截取前8个字符作为标题
                new_title = new_title + "..." if len(user_msg) > 8 else new_title
                sql_update_session = """
                    UPDATE sessions 
                    SET title = %s, last_activity_at = %s, message_count = message_count + 2
                    WHERE id = %s AND user_id = %s
                """
                cursor.execute(sql_update_session, (new_title, timestamp_dt, session_id, user_id))
            else:
                #  只更新活动时间和消息数
                sql_update_session = """
                    UPDATE sessions 
                    SET last_activity_at = %s, message_count = message_count + 2
                    WHERE id = %s AND user_id = %s
                """
                cursor.execute(sql_update_session, (timestamp_dt, session_id, user_id))
            
            conn.commit()
    except mysql.connector.Error as e:
        logging.error(f"保存聊天记录到数据库时出错 (用户 ID: {user_id}, 会话: {session_id}): {e}")
    except Exception as e:
        logging.error(f"保存聊天时发生未知错误: {e}")

