'''
app.auth.service - 用户认证服务

- 用户查询
- 密码哈希
- 注册用户

'''
from app.db import get_db_connection
import mysql.connector
import logging
import bcrypt
from mysql.connector import errorcode       
# 查找用户
def find_user(username):
    try:
        # with提供一个临时变量，储存这个函数
        with get_db_connection() as conn:
            # 使用 dictionary=True 使 cursor 返回字典而不是元组，方便按列名访问
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
            user_row = cursor.fetchone()
            # cursor.close() # 'with' 语句会自动关闭游标和连接
            if user_row:
                # 返回一个字典，包含 id, username 和 password_hash
                return user_row # user_row 已经是字典了
            return None
    except mysql.connector.Error as e: 
        logging.error(f"查找用户 '{username}' 时数据库出错: {e}")
        return None
    except Exception as e:
        logging.error(f"查找用户 '{username}' 时发生未知错误: {e}")
        return None
# 哈希密码
def hash_password(password):
    hashed_password = bcrypt.hashpw(
        password.encode('utf-8'), 
        bcrypt.gensalt(rounds=12)).decode('utf-8')
    
    return hashed_password

# 注册用户
def register_user(username, plain_password):
    """
    注册新用户，使用 bcrypt 对明文密码进行哈希。
    
    Args:
        username: 用户名
        plain_password: 前端发送的明文密码（通过HTTPS保护）
    
    Returns:
        (success: bool, message: str)
    """
    if find_user(username): # 首先检查用户是否存在
        return False, "用户名已被注册。"

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 使用 bcrypt 对明文密码进行哈希（包含自动生成的盐值）
            hashed_password = hash_password(plain_password)
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                           (username, hashed_password))
            conn.commit()
            # user_id = cursor.lastrowid # 如果需要获取新用户的ID
            # cursor.close()
        logging.info(f"新用户注册成功: {username}")
        return True, "注册成功！"
    except mysql.connector.Error as e: # <-- 修改异常类型
        # MySQL 的 IntegrityError 对于 UNIQUE 约束冲突通常是 ER_DUP_ENTRY (errno 1062)
        # 这里对应的是mysql报错文档
        if e.errno == errorcode.ER_DUP_ENTRY:
            logging.warning(f"尝试注册已存在的用户名 (数据库约束): {username}")
            return False, "用户名已被注册。"
        logging.error(f"注册用户 '{username}' 时数据库出错: {e}")
        return False, "注册过程中发生服务器错误。"
    except Exception as e:
        logging.error(f"注册用户 '{username}' 时发生未知错误: {e}")
        return False, "注册过程中发生服务器错误。"
