'''
app.files.routes - 文件路由
'''
from flask import Blueprint, request, jsonify, session
import logging
from app.db import get_db_connection
import mysql.connector
import os
import hashlib
from app.chat.services import save_chat

files_bp = Blueprint('files', __name__, url_prefix='/api')

# 获取文件列表
@files_bp.route('/files')
def get_file_list():
    if 'user_id' not in session:
        return jsonify({"error": "用户未登录或会话已过期"}), 401
    
    user_id = session['user_id']
    logging.info(f"用户 {user_id} 请求文件列表")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, filename, last_accessed_at FROM uploaded_files WHERE user_id = %s ORDER BY last_accessed_at DESC", (user_id,))
            file_rows = cursor.fetchall()
        if not file_rows:
            logging.info(f"用户 {user_id} 没有文件记录")
            return jsonify([])
        file_list_for_frontend = [
            (
                row["id"], 
                {
                    "preview": row["filename"], 
                    "last_time": row["last_accessed_at"].strftime("%m-%d %H:%M")
                }
            )
            for row in file_rows
        ]

    except mysql.connector.Error as e:
        logging.error(f"为用户 {user_id} 读取文件列表时数据库出错: {e}")
        return jsonify({"error": f"读取文件列表时出错: {e}"}), 500
    
    logging.info(f"为用户 {user_id} 返回 {len(file_list_for_frontend)} 个文件")
    return jsonify(file_list_for_frontend)

## 上传文件
@files_bp.route('/upload_file', methods=['POST'])
def upload_file():
    #  重构：从 Session 获取用户身份 
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '用户未登录或会话已过期'}), 401
    
    user_id = session['user_id']
    username = session.get('username', '未知用户') # 用于日志
    
    session_id = request.form.get('session_id')
    if not session_id:
        logging.warning(f"用户 {username} 上传文件请求缺少 session_id")
        return jsonify({'success': False, 'error': '请求无效，缺少会话ID'}), 400
    # 

    if 'file' not in request.files:
        logging.warning(f"用户 {username} 上传CSV请求中没有文件部分")
        return jsonify({'success': False, 'error': '没有文件被上传'}), 400
    
    file = request.files['file'] # 获取上传的文件对象

    # 3. 检查文件名是否为空
    if file.filename == '':
        logging.warning(f"用户 {username} 上传了但未选择文件")
        return jsonify({'success': False, 'error': '没有选择文件'}), 400
    
    allowed_extensions = {'.csv'}
    allowed_mimetypes = {'text/csv', 'application/vnd.ms-excel'} # 有些浏览器对csv的mimetype可能是后者

    # 4. 检查文件扩展名和MIME类型
    original_filename = file.filename
    file_ext = os.path.splitext(original_filename)[1].lower() # 获取文件扩展名并转为小写

    if not (file_ext in allowed_extensions and file.mimetype in allowed_mimetypes):
        logging.warning(f"用户 {username} 尝试上传非法文件类型: {original_filename} (MIME: {file.mimetype})")
        return jsonify({'success': False, 'error': '只允许上传 CSV 文件。请检查文件格式和扩展名。'}), 400

    # 5. 读取文件内容并计算哈希
    try:
        file.seek(0) # 确保从文件开头读取
        file_content = file.read() # 将整个文件内容读取为 bytes
        file_hash = hashlib.sha256(file_content).hexdigest()
        file_size = len(file_content)
    except Exception as e:
        logging.error(f"用户 {username} 上传文件 {original_filename} 时读取内容或计算哈希失败: {e}")
        return jsonify({'success': False, 'error': '处理文件内容失败'}), 500
    
    # 6. 检查重复文件并保存到数据库 (使用哈希)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True) # 使用字典游标
            
            # 检查是否已存在相同哈希的文件
            cursor.execute("""
                SELECT id, filename FROM uploaded_files 
                WHERE user_id = %s AND file_hash = %s
            """, (user_id, file_hash))
            existing_file = cursor.fetchone()
            
            if existing_file:
                # 文件内容已存在，更新访问时间戳和计数
                cursor.execute("""
                    UPDATE uploaded_files 
                    SET last_accessed_at = NOW(), access_count = access_count + 1
                    WHERE id = %s
                """, (existing_file['id'],))
                conn.commit()
                # 使用原始文件名进行提示
                action_message = f'您之前已上传过内容相同的文件 (名为 "{existing_file["filename"]}")。无需重复上传。'
                logging.info(f"用户 {username} (ID: {user_id}) 上传了重复内容的文件: {original_filename} (Hash: {file_hash[:10]}...)")
            else:
                # 文件不存在，插入新记录
                cursor.execute("""
                    INSERT INTO uploaded_files (user_id, filename, original_filename, mime_type, file_size, file_hash, file_content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (user_id, original_filename, original_filename, file.mimetype, file_size, file_hash, file_content))
                conn.commit()
                action_message = f'文件 "{original_filename}" 上传成功！'
                logging.info(f"用户 {username} (ID: {user_id}) 成功上传新文件: {original_filename}")
        
        # 保存文件上传的聊天记录
        user_message = f"上传文件: {original_filename}"
        # 修改AI响应，使其更清晰
        ai_message_text = f"已接收您的文件：`{original_filename}`。\n\n{action_message}\n\n您现在可以对我提问，例如：请对`{original_filename}`进行因果分析"
        ai_response = {"type": "text", "summary": ai_message_text}
        
        save_chat(user_id, session_id, user_message, ai_response)
        
        return jsonify({'success': True, 'message': action_message, 'ai_response': ai_response})
    except mysql.connector.Error as e:
        logging.error(f"用户 {username} 保存文件 {original_filename} 到数据库时出错: {e}")
        return jsonify({'success': False, 'error': '保存文件到数据库失败'}), 500
    except Exception as e: # 捕获其他可能的未知错误
        logging.error(f"用户 {username} 上传文件 {original_filename} 时发生未知服务器错误: {e}")
        return jsonify({'success': False, 'error': '上传文件时发生服务器内部错误'}), 500


# delete file
@files_bp.route('/delete_file', methods=['POST'])
def delete_file():
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "用户未登录或会话已过期"}), 401
    
    user_id = session['user_id']
    data = request.json
    file_id = data.get('file_id')

    if not file_id:
        return jsonify({"success": False, "error": "缺少文件ID"}), 400

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 执行删除，确保文件属于用户
            cursor.execute("DELETE FROM uploaded_files WHERE id = %s AND user_id = %s", (file_id, user_id))
            
            conn.commit()
            
            if cursor.rowcount == 0:
                logging.warning(f"用户 {user_id} 尝试删除无权或不存在的文件 {file_id}")
                return jsonify({"success": False, "error": "无法删除该文件，权限不足或文件不存在"}), 404
            
            logging.info(f"用户 {user_id} 成功删除了文件 {file_id}")
            return jsonify({"success": True, "message": "文件已成功删除"})

    except mysql.connector.Error as e:
        logging.error(f"删除文件 {file_id} (用户 {user_id}) 时数据库出错: {e}")
        return jsonify({"success": False, "error": "删除文件时数据库出错"}), 500
    except Exception as e:
        logging.error(f"删除文件 {file_id} (用户 {user_id}) 时发生未知错误: {e}")
        return jsonify({"success": False, "error": "删除文件时发生未知错误"}), 500
