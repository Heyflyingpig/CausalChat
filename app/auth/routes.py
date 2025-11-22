"""
用户认证路由
"""
from flask import Blueprint, request, jsonify, session
from app.auth.service import find_user, hash_password, register_user
import bcrypt
import logging

auth_bp = Blueprint('auth', __name__, url_prefix='/api')

# 获取注册值,检查注册值
@auth_bp.route('/register', methods=['POST'])
def handle_register():
    """
    处理用户注册请求。
    接收前端通过HTTPS发送的明文密码，在后端使用bcrypt进行哈希。
    """
    data = request.json
    username = data.get('username')
    plain_password = data.get('password')  # 接收明文密码（HTTPS保护传输）

    if not username or not plain_password:
        return jsonify({'success': False, 'error': '缺少用户名或密码'}), 400

    # 基本的用户名和密码格式验证
    if len(username) < 3:
        return jsonify({'success': False, 'error': '用户名至少需要3个字符'}), 400
    
    # 密码长度验证（建议至少6位，可根据需求调整）
    if len(plain_password) < 6:
        return jsonify({'success': False, 'error': '密码至少需要6个字符'}), 400
    
    # 可选：添加密码强度验证
    # if not any(c.isdigit() for c in plain_password):
    #     return jsonify({'success': False, 'error': '密码必须包含至少一个数字'}), 400

    success, message = register_user(username, plain_password)
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': message}), 400 # 用户名已存在等是客户端错误

# 检查登录值
@auth_bp.route('/login', methods=['POST'])
def handle_login():
    """
    处理用户登录请求。
    接收前端通过HTTPS发送的明文密码，使用bcrypt进行验证。
    """


    data = request.json
    username = data.get('username')
    plain_password = data.get('password')  # 接收明文密码（HTTPS保护传输）

    if not username or not plain_password:
        return jsonify({'success': False, 'error': '缺少用户名或密码'}), 400

    user_data = find_user(username)

    if not user_data:
        return jsonify({'success': False, 'error': '用户名不存在'}), 401 # 401 Unauthorized

    # 使用 bcrypt 验证密码
    # bcrypt.checkpw() 会自动从存储的哈希中提取盐值进行验证
    stored_hashed_password = user_data["password_hash"].encode('utf-8')
    if bcrypt.checkpw(plain_password.encode('utf-8'), stored_hashed_password):
        logging.info(f"用户登录成功: {username}")
        
        #  核心修改：在 Session 中存储用户信息 
        session.clear() # 先清除旧的会话数据
        session['user_id'] = user_data['id']
        session['username'] = user_data['username']
        # Session 会自动通过浏览器 cookie 维护状态，不再需要文件
        
        return jsonify({'success': True, 'username': username})
    else:
        logging.warning(f"用户登录失败（密码错误）: {username}")
        return jsonify({'success': False, 'error': '密码错误'}), 401 # 401 Unauthorized

# 登出
@auth_bp.route('/logout', methods=['POST'])
def handle_logout():

    
    # 从会话中获取用户名用于日志记录
    username = session.get('username', '未知用户')
    logging.info(f"用户 {username} 请求退出登录")

    #  核心修改：清除会话 
    session.clear()

    return jsonify({'success': True})

#  检查认证状态 API 端点 
@auth_bp.route('/check_auth', methods=['GET'])
def check_auth():
    """检查当前后端记录的登录状态"""

    if 'user_id' in session and 'username' in session:
        username = session['username']
        logging.debug(f"检查认证状态：用户 '{username}' (通过会话) 已登录")
        return jsonify({'isLoggedIn': True, 'username': username})
    else:
        logging.debug("检查认证状态：无有效会话")
        return jsonify({'isLoggedIn': False})
