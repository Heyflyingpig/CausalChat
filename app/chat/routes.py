'''
app.chat.routes - 聊天路由
'''
from flask import Blueprint, request, jsonify, session
from app.chat.services import get_chat_history
from app.db import get_db_connection
import logging
import mysql.connector
import json
from Agent.Report.Metadata_sum import replace_placeholders

chat_bp = Blueprint('chat', __name__, url_prefix='/api')
import uuid

# 新对话
@chat_bp.route('/new_chat',methods=['POST'])
def new_chat():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '用户未登录'}), 401
    
    user_id = session['user_id']
    username = session.get('username', '未知用户')
    new_session_id = str(uuid.uuid4())

    # 核心修改：不再立即创建数据库记录，只生成session_id
    # 会话记录将在用户发送第一条消息时通过 save_chat() 函数创建
    logging.info(f"用户 {username} (ID: {user_id}) 生成新会话ID: {new_session_id} ")
    return jsonify({'success': True, 'new_session_id': new_session_id})

# 会话管理接口,获取会话
@chat_bp.route('/sessions')
def get_sessions():
    if 'user_id' not in session:
        return jsonify({"error": "用户未登录或会话已过期"}), 401
    
    user_id = session['user_id']
    logging.info(f"用户 {user_id} 请求会话列表 (新版逻辑)")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            # 高效地直接从 sessions 表查询
            cursor.execute("""
                SELECT id, title, last_activity_at
                FROM sessions
                WHERE user_id = %s AND is_archived = FALSE
                ORDER BY last_activity_at DESC
            """, (user_id,)) 
            session_rows = cursor.fetchall()

        if not session_rows:
            logging.info(f"用户 {user_id} 没有会话记录")
            return jsonify([])

        # 格式化以适应前端期望的 (id, {preview, last_time}) 结构
        session_list_for_frontend = [
            (
                row["id"], 
                {
                    "preview": row["title"], 
                    "last_time": row["last_activity_at"].strftime("%m-%d %H:%M")
                }
            )
            for row in session_rows
        ]

    except mysql.connector.Error as e:
        logging.error(f"为用户 {user_id} 读取会话列表时数据库出错: {e}")
        return jsonify({"error": f"读取历史记录时出错: {e}"}), 500
    
    logging.info(f"为用户 {user_id} 返回 {len(session_list_for_frontend)} 个会话")
    return jsonify(session_list_for_frontend)

# 加载特定会话内容 
@chat_bp.route('/load_session')
def load_session_content():
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "用户未登录或会话已过期"}), 401
    
    user_id = session['user_id']
    username = session['username']

    session_id = request.args.get('session')

    if not session_id:
        return jsonify({"success": False, "error": "缺少 session ID"}), 400

    logging.info(f"用户 {username} (ID: {user_id}) 请求加载会话: {session_id} (延迟创建模式)")

    messages = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            # 处理延迟创建的session
            # 首先检查session是否存在
            cursor.execute("SELECT id FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            session_exists = cursor.fetchone()

            if not session_exists:
                # Session还不存在（用户还没发送第一条消息），返回空消息列表
                logging.info(f"会话 {session_id} 尚未创建，返回空消息列表")
                return jsonify({"success": True, "messages": []})


            # 按照id获取所有消息和其附件，并且顺序排序，时间由早到晚
            cursor.execute("""
                SELECT
                    id,message_type,content,has_attachment
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (session_id,))
            chat_rows = cursor.fetchall()

            # 处理每条消息（在 with 语句内部）
            for row in chat_rows:
                sender = "user" if row["message_type"] == 'user' else "ai"

                # 如果是AI消息，且有附件，则优先使用附件内容
                if sender == "ai" and row["has_attachment"]:
                    cursor.execute("""
                        SELECT attachment_type, content
                         FROM chat_attachments
                        WHERE message_id = %s
                    """, (row["id"],))
                    attachments = cursor.fetchall()

                    causal_graph_data = None
                    visualization_mapping = None

                    ## attachment格式：{"type": "causal_graph", "content": {...}}
                    for attachment in attachments:
                        if attachment["attachment_type"] == "causal_graph":
                            try:
                                causal_graph_data = json.loads(attachment["content"])
                            except json.JSONDecodeError:
                                logging.warning(f"无法解析 causal_graph 附件，Message ID: {row['id']}")

                        elif attachment["attachment_type"] == "visualization":
                            try:
                                visualization_mapping = json.loads(attachment["content"])
                            except json.JSONDecodeError:
                                logging.warning(f"无法解析 visualization 附件，Message ID: {row['id']}")

                    if causal_graph_data:
                        message_content = causal_graph_data

                        if visualization_mapping and "summary" in message_content:
                            message_content["summary"] = replace_placeholders(message_content["summary"], visualization_mapping)

                        messages.append({"sender": "ai", "text": message_content})
                    else:
                        message_text = row["content"]

                        if visualization_mapping:
                            message_text = replace_placeholders(message_text, visualization_mapping)

                        messages.append({"sender": "ai", "text": message_text})

                else:
                    # 对于用户消息或没有附件的AI消息，直接使用content
                    messages.append({"sender": sender, "text": row["content"]})

        logging.info(f"用户 {username} 成功加载会话 {session_id} ({len(messages)} 条消息)")
        return jsonify({"success": True, "messages": messages})

    except mysql.connector.Error as e:
        logging.error(f"加载会话 {session_id} (用户 {username}) 时数据库出错: {e}")
        return jsonify({"success": False, "error": f"加载会话时出错: {e}"}), 500
    except Exception as e:
        logging.error(f"加载会话 {session_id} (用户 {username}) 时发生未知错误: {e}")
        return jsonify({"success": False, "error": f"加载会话时出错: {e}"}), 500

## 更改会话
@chat_bp.route('/change_session', methods=['POST'])
def change_session():
    #  用户认证检查 
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "用户未登录或会话已过期"}), 401
    
    user_id = session['user_id']
    
    #  修改：从 POST 请求的 JSON body 中获取数据 
    data = request.json
    title = data.get('title')
    session_id = data.get('session_id')

    if not title or not session_id:
        return jsonify({"success": False, "error": "缺少标题或会话ID"}), 400

    try:
        with get_db_connection() as conn:
            #  增加 user_id 条件以确保安全，并处理延迟创建的session 
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET title = %s WHERE id = %s AND user_id = %s",
                (title, session_id, user_id)
            )
            conn.commit()
            
            #  更详细的错误处理，区分延迟创建的session 
            if cursor.rowcount == 0:
                # 检查session是否因为延迟创建而不存在
                cursor.execute("SELECT 1 FROM chat_messages WHERE session_id = %s AND user_id = %s LIMIT 1", (session_id, user_id))
                has_messages = cursor.fetchone()
                
                if not has_messages:
                    # session确实不存在且没有消息，可能是延迟创建的session
                    logging.info(f"用户 {user_id} 尝试修改尚未创建的会话标题 {session_id}（延迟创建模式）")
                    return jsonify({"success": False, "error": "无法修改标题，请先发送一条消息来创建会话"}), 400
                else:
                    # 有消息但session记录不存在，这是一个数据不一致的问题
                    logging.warning(f"用户 {user_id} 的会话 {session_id} 存在消息但session记录缺失")
                    return jsonify({"success": False, "error": "会话数据异常，请联系管理员"}), 500
            
        logging.info(f"用户 {user_id} 成功将会话 {session_id} 的标题更新为 '{title}'")
        return jsonify({"success": True, "message": "会话标题已更新"})
    except mysql.connector.Error as e:
        logging.error(f"更新会话标题时数据库出错 (用户ID: {user_id}, 会话ID: {session_id}): {e}")
        return jsonify({"success": False, "error": "更新会话标题时数据库出错"}), 500

## 删除会话
@chat_bp.route('/delete_session', methods=['POST'])
def delete_session():
    #  核心修改：安全和完整的删除逻辑，支持延迟创建 
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "用户未登录或会话已过期"}), 401
    
    user_id = session['user_id']
    data = request.json
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({"success": False, "error": "缺少会话ID"}), 400

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 开启事务
            conn.start_transaction()
            
            #  修改：处理延迟创建的session 
            # 0. 检查session是否存在
            cursor.execute("SELECT id FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            session_exists = cursor.fetchone()
            
            if not session_exists:
                # 检查是否有相关的消息（可能是数据不一致的情况）
                cursor.execute("SELECT 1 FROM chat_messages WHERE session_id = %s AND user_id = %s LIMIT 1", (session_id, user_id))
                has_messages = cursor.fetchone()
                
                if not has_messages:
                    # session不存在且没有消息，这是正常的延迟创建情况
                    conn.rollback()
                    logging.info(f"用户 {user_id} 尝试删除尚未创建的会话 {session_id}（延迟创建模式），视为成功")
                    return jsonify({"success": True, "message": "会话删除成功（会话尚未创建）"})
                else:
                    # 有消息但session记录不存在，清理孤立的消息
                    logging.warning(f"发现用户 {user_id} 的会话 {session_id} 有孤立消息，正在清理")
            # 

            # 1. 删除与该会话相关的附件 (通过连接 chat_messages)
            # 这是为了处理 chat_attachments 和 chat_messages 之间没有直接外键的情况
            sql_delete_attachments = """
                DELETE ca FROM chat_attachments ca
                JOIN chat_messages cm ON ca.message_id = cm.id
                WHERE cm.session_id = %s AND cm.user_id = %s
            """
            cursor.execute(sql_delete_attachments, (session_id, user_id))
            deleted_attachments = cursor.rowcount
            logging.info(f"为会话 {session_id} 删除了 {deleted_attachments} 个附件")

            # 2. 删除该会话的所有聊天记录
            cursor.execute("DELETE FROM chat_messages WHERE session_id = %s AND user_id = %s", (session_id, user_id))
            deleted_messages = cursor.rowcount
            logging.info(f"为会话 {session_id} 删除了 {deleted_messages} 条聊天记录")

            # 3. 删除会话本身（如果存在）
            if session_exists:
                cursor.execute("DELETE FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
                logging.info(f"删除了会话记录 {session_id}")
            
            # 提交事务
            conn.commit()
            
            logging.info(f"用户 {user_id} 成功删除了会话 {session_id} 及其所有数据")
            return jsonify({"success": True, "message": "会话已成功删除"})

    except mysql.connector.Error as e:
        conn.rollback() # 确保出错时回滚
        logging.error(f"删除会话 {session_id} (用户 {user_id}) 时数据库出错: {e}")
        return jsonify({"success": False, "error": "删除会话时数据库出错"}), 500
    except Exception as e:
        conn.rollback() # 确保出错时回滚
        logging.error(f"删除会话 {session_id} (用户 {user_id}) 时发生未知错误: {e}")
        return jsonify({"success": False, "error": "删除会话时发生未知错误"}), 500
