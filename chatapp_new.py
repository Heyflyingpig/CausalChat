# app.py (Flask后端)
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI
import subprocess
import os
import sqlite3 
import uuid
from datetime import datetime
import logging
import json 

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__, static_folder='static')
current_session = str(uuid.uuid4()) ## 全局会话 ID，现在主要由前端在加载历史时设置
BASE_DIR = os.path.dirname(__file__)
SETTING_DIR = os.path.join(BASE_DIR, "setting")

DATABASE_PATH = os.path.join(BASE_DIR, "chat_app.db") # <-- 新增：数据库文件路径
CONFIG_PATH = os.path.join(BASE_DIR, "config.json") # <-- 新增：配置文件路径

current_logged_in_user = None # <-- 新增：全局变量
current_api = "zhipuai"
current_model = "glm-4-flash"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
apikey = "49520ba1c63a183b4c6333b4b4d523fe.9C6LVcy8IPz80pyf"



def load_config():
    """从 config.json 加载配置"""
    global current_logged_in_user
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                current_logged_in_user = config_data.get("logged_in_user") # 读取已登录用户
                logging.info(f"从配置文件加载登录用户: {current_logged_in_user}")
                return config_data
        else:
            logging.info(f"配置文件 {CONFIG_PATH} 不存在，将创建。")
            # 文件不存在，返回默认空配置，并初始化 current_logged_in_user 为 None
            current_logged_in_user = None
            return {"logged_in_user": None}
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"读取配置文件 {CONFIG_PATH} 时出错: {e}。将使用默认配置。")
        # 文件损坏或读取错误，同样返回默认
        current_logged_in_user = None
        return {"logged_in_user": None}

def save_config(config_data):
    """将配置数据保存到 config.json"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False) # indent=4 使文件更易读
        logging.info(f"配置已保存到 {CONFIG_PATH}")
    except IOError as e:
        logging.error(f"保存配置文件 {CONFIG_PATH} 时出错: {e}")

# --- 程序启动时加载配置 ---
config = load_config() # 加载初始配置并设置 current_logged_in_user


# --- 新增：数据库辅助函数 ---
def get_db_connection():
    """创建并返回一个数据库连接。"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row # 允许通过列名访问数据
    return conn

def initialize_database():
    try:
        with get_db_connection() as conn: # 使用 with 语句确保连接自动关闭
            cursor = conn.cursor()
            # 创建用户表
            # 自动增加id
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 创建聊天记录表
            # 这里的username未来可能需要更换为id
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    user_msg TEXT,
                    ai_msg TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            # 为 chat_messages 表的 session_id, user_id, timestamp 创建索引以提高查询效率
            # 这里的命名规则是人为定的
            cursor.execute("DROP INDEX IF EXISTS idx_chat_messages_session_user_time")
            cursor.execute("DROP INDEX IF EXISTS idx_chat_messages_user_session_time")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_userid_time ON chat_messages (session_id, user_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_userid_session_time ON chat_messages (user_id, session_id, timestamp DESC)")

            conn.commit()# 提交事务，保存
        
        logging.info(f"数据库已初始化: {DATABASE_PATH}")
    except sqlite3.Error as e:
        logging.error(f"数据库初始化失败: {e}")
        raise # 重新抛出异常，以便应用启动时能感知到严重错误

initialize_database() # 程序启动时检查并初始化数据库

# 全局状态

# --- 新增：用户认证相关函数 ---

# 查找用户
def find_user(username):
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 同时选择 id, username, 和 password_hash
            # 这里的数据库语法是说：？是占位符，username是变量，可以防止恶意注入
            cursor.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
            user_row = cursor.fetchone()
            if user_row:
                # 返回一个字典，包含 id, username 和 password_hash
                return {"id": user_row["id"], "username": user_row["username"], "password_hash": user_row["password_hash"]}
            return None
    except sqlite3.Error as e:
        logging.error(f"查找用户 '{username}' 时数据库出错: {e}")
        return None


# 注册用户
def register_user(username, hashed_password):
    if find_user(username): # 首先检查用户是否存在
        return False, "用户名已被注册。"

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                           (username, hashed_password))
            conn.commit()
        logging.info(f"新用户注册成功: {username}")
        return True, "注册成功！"
    except sqlite3.IntegrityError: # 捕获 UNIQUE 约束冲突，虽然 find_user 应该先捕获
        logging.warning(f"尝试注册已存在的用户名 (数据库约束): {username}")
        return False, "用户名已被注册。"
    except sqlite3.Error as e:
        logging.error(f"注册用户 '{username}' 时数据库出错: {e}")
        return False, "注册过程中发生服务器错误。"

# 获取注册值
@app.route('/api/register', methods=['POST'])
def handle_register():
    data = request.json
    username = data.get('username')
    hashed_password = data.get('password') # 前端已经哈希过了

    if not username or not hashed_password:
        return jsonify({'success': False, 'error': '缺少用户名或密码'}), 400

    # 基本的用户名和密码格式验证 (可选)
    if len(username) < 3:
         return jsonify({'success': False, 'error': '用户名至少需要3个字符'}), 400
    # 密码哈希的长度通常是固定的 (SHA256 是 64 个十六进制字符)
    if len(hashed_password) != 64:
         return jsonify({'success': False, 'error': '密码格式无效'}), 400


    success, message = register_user(username, hashed_password)
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': message}), 400 # 用户名已存在等是客户端错误


@app.route('/api/login', methods=['POST'])
def handle_login():
    global current_logged_in_user # 声明要修改全局变量
    data = request.json
    username = data.get('username')
    hashed_password_from_client = data.get('password') # 前端已经哈希过了

    if not username or not hashed_password_from_client:
        return jsonify({'success': False, 'error': '缺少用户名或密码'}), 400

    user_data = find_user(username)

    if not user_data:
        return jsonify({'success': False, 'error': '用户名不存在'}), 401 # 401 Unauthorized

    # 比较哈希值
    stored_hashed_password = user_data["password_hash"]
    if stored_hashed_password == hashed_password_from_client:
        logging.info(f"用户登录成功: {username}")
        # --- 修改：更新并保存配置 ---
        # 这里是检测目前登录的用户
        current_logged_in_user = username # 更新全局变量
        config["logged_in_user"] = username # 更新配置字典
        save_config(config) # 保存到文件
        # --------------------------
        return jsonify({'success': True, 'username': username}) # 保持现有返回结构
    else:
        logging.warning(f"用户登录失败（密码错误）: {username}")
        return jsonify({'success': False, 'error': '密码错误'}), 401 # 401 Unauthorized

# 登出
@app.route('/api/logout', methods=['POST'])
def handle_logout():
    global current_logged_in_user
    logged_out_user = current_logged_in_user # 获取当前登录用户，用于日志记录
    logging.info(f"用户 {logged_out_user} 请求退出登录")

    # --- 清除配置 ---
    current_logged_in_user = None # 清除全局变量
    config["logged_in_user"] = None # 更新配置字典
    save_config(config) # 保存到文件
    # ----------------

    return jsonify({'success': True})

# --- 新增：检查认证状态 API 端点 ---
@app.route('/api/check_auth', methods=['GET'])
def check_auth():
    """检查当前后端记录的登录状态"""
    if current_logged_in_user:
        logging.debug(f"检查认证状态：用户 '{current_logged_in_user}' 已登录")
        return jsonify({'isLoggedIn': True, 'username': current_logged_in_user})
    else:
        logging.debug("检查认证状态：无用户登录")
        return jsonify({'isLoggedIn': False})



# 利用flask的jsonify的框架，将后端处理转发到前端
@app.route('/api/send', methods=['POST'])
def handle_message():
    data = request.json
    user_input = data.get('message', '')
    username = data.get('username') # **新增：** 获取用户名

    if not username:
         logging.warning("收到发送消息请求，但缺少用户名")
         return jsonify({'success': False, 'error': '用户未登录或请求无效'}), 401

    user_data = find_user(username) # 获取用户数据，包括 id
    if not user_data:
        logging.error(f"处理消息时用户 '{username}' 未找到。")
        return jsonify({'success': False, 'error': '用户认证失败'}), 401
    
    user_id = user_data['id'] # 提取 user_id

    logging.info(f"用户 {username} (ID: {user_id}) 发送消息: {user_input[:50]}...") # 日志记录

    try:
        response = ai_call(user_input)
        #传递 user_id 给 save_chat
        save_chat(user_id, user_input, response)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        logging.error(f"处理用户 {username} (ID: {user_id}) 消息时出错: {e}")
        return jsonify({'success': False, 'error': f'处理消息时出错: {e}'}), 500

@app.route('/api/new_chat',methods=['POST'])
def new_chat():
  
    global current_session
    old_session = current_session
    current_session = str(uuid.uuid4()) # 生成新的全局 session_id
    logging.info(f"创建新会话 ID: {current_session} (旧: {old_session})")
    # 注意：这个全局 current_session 可能在多用户场景下不是最佳实践
    # 但对于单用户本地运行或前端驱动会话切换的场景是可行的
    return jsonify({'success': True})

# 回答函数 (不变)
def ai_call(text):
    client = OpenAI(base_url=BASE_URL, api_key= apikey)
    response = client.chat.completions.create(
        model=current_model,
        messages=[
            {"role": "system", "content": "你是一个工程师"},
            {"role": "user", "content": text}
            ],
    )
    return response.choices[0].message.content


## 保存历史文件 ( 添加 user_id 参数和列)
def save_chat(user_id, user_msg, ai_msg): # 修改：username -> user_id
    global current_session # 需要访问全局会话ID
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 使用 datetime 获取更精确的时间

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_messages (session_id, user_id, user_msg, ai_msg, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (current_session, user_id, user_msg, ai_msg, timestamp)) # 修改：使用 user_id
            conn.commit()
        # logging.info(f"聊天记录已保存 (用户 ID: {user_id}, 会话: {current_session})")
    except sqlite3.Error as e:
        logging.error(f"保存聊天记录到数据库时出错 (用户 ID: {user_id}, 会话: {current_session}): {e}")
    except Exception as e: # 捕获其他可能的未知错误
        logging.error(f"保存聊天时发生未知错误: {e}")


# 会话管理接口 (**修改：** 过滤用户)
@app.route('/api/sessions')
def get_sessions():
    username = request.args.get('user') # **新增：** 从查询参数获取用户名
    if not username:
        logging.warning("获取会话列表请求缺少用户名")
        return jsonify({"error": "需要提供用户名"}), 400

    user_data = find_user(username) # 获取用户数据
    if not user_data:
        logging.warning(f"用户 {username} 请求会话列表，但用户不存在")
        return jsonify({"error": "用户不存在"}), 404 # 404 Not Found 似乎更合适
    
    user_id = user_data['id'] # 提取 user_id

    logging.info(f"用户 {username} (ID: {user_id}) 请求会话列表")
    sessions = {}
    # if os.path.exists(HISTORY_PATH): # <-- 移除对 HISTORY_PATH 的检查
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 查询每个会话的最新一条用户消息作为预览，并按最新时间排序
            # 这里运用到了内外表，我们的session_id是外表，chat_messages（cm_outer）是内表
            # 我们所做的就是在外表查询的基础上，进行内表的查询
            # 由于内表的查询是基于chat的uuser_msg的，所以要进行新一轮的筛选
            cursor.execute("""
                SELECT
                    session_id,
                    MAX(timestamp) as last_time,
                    (SELECT COALESCE(user_msg, '') FROM chat_messages cm_inner
                     WHERE cm_inner.session_id = cm_outer.session_id
                       AND cm_inner.user_id = cm_outer.user_id 
                       AND cm_inner.user_msg IS NOT NULL AND LENGTH(cm_inner.user_msg) > 0
                     ORDER BY cm_inner.timestamp DESC LIMIT 1) as preview_user_msg,
                    
                    (SELECT COALESCE(ai_msg, '') FROM chat_messages cm_inner
                     WHERE cm_inner.session_id = cm_outer.session_id
                       AND cm_inner.user_id = cm_outer.user_id 
                       AND cm_inner.ai_msg IS NOT NULL AND LENGTH(cm_inner.ai_msg) > 0
                     ORDER BY cm_inner.timestamp DESC LIMIT 1) as preview_ai_msg
                FROM chat_messages cm_outer --cm_outer是chat_messages表的别名
                WHERE user_id = ?
                GROUP BY session_id -- 按session_id分组
                ORDER BY last_time DESC
            """, (user_id,)) 
            session_rows = cursor.fetchall()

        if not session_rows:
            logging.info(f"用户 {username} (ID: {user_id}) 没有会话记录")
            return jsonify([])

        for row in session_rows:
            session_id = row["session_id"]
            last_time_str = row["last_time"] # 已经是字符串格式 "YYYY-MM-DD HH:MM:SS"
            
            # 优先使用用户消息作为预览，如果用户消息为空，则尝试AI消息
            preview_msg = row["preview_user_msg"]
            if not preview_msg or preview_msg.strip() == "":
                preview_msg = row["preview_ai_msg"]
            if not preview_msg: # 如果两者都为空
                preview_msg = "无预览内容"


            sessions[session_id] = {
                "last_time": last_time_str,
                "preview": preview_msg[:30] + "..." if len(preview_msg) > 30 else preview_msg
            }

    except sqlite3.Error as e:
        logging.error(f"为用户 {username} 读取会话列表时数据库出错: {e}")
        return jsonify({"error": f"读取历史记录时出错: {e}"}), 500
    except Exception as e: # 捕获其他可能的未知错误
            logging.error(f"处理历史记录时发生未知错误 (用户 {username}): {e}")
            return jsonify({"error": f"处理历史记录时出错: {e}"}), 500

    session_list_for_frontend = list(sessions.items())
    logging.info(f"为用户 {username} 返回 {len(session_list_for_frontend)} 个会话")
    return jsonify(session_list_for_frontend)


# 加载特定会话内容 (**修改：** 增加用户验证)
@app.route('/api/load_session')
def load_session_content():
    global current_session # 声明我们要修改全局变量
    session_id = request.args.get('session')
    username = request.args.get('user') #  获取用户名

    if not session_id or not username:
        logging.warning("加载会话请求缺少 session_id 或 username")
        return jsonify({"success": False, "error": "缺少 session ID 或用户名"}), 400

    user_data = find_user(username) # 获取用户数据
    if not user_data:
        logging.warning(f"用户 {username} 请求加载会话，但用户不存在")
        return jsonify({"success": False, "error": "用户不存在或无法加载会话"}), 404

    user_id = user_data['id'] # 提取 user_id

    logging.info(f"用户 {username} (ID: {user_id}) 请求加载会话: {session_id}")

    messages = []
    session_found_for_user = False
    # if os.path.exists(HISTORY_PATH): # <-- 移除对 HISTORY_PATH 的检查
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_msg, ai_msg FROM chat_messages
                WHERE session_id = ? AND user_id = ? 
                ORDER BY timestamp ASC
            """, (session_id, user_id)) # 修改：传递 user_id
            chat_rows = cursor.fetchall()

        if chat_rows:
            session_found_for_user = True
            for row in chat_rows:
                # 根据存储的逻辑，一条记录可能同时有 user_msg 和 ai_msg (如果AI是紧接着用户的回复)
                # 或者某一条只有 user_msg (用户刚发送，AI还没回)
                # 或者某一条只有 ai_msg (如果允许AI主动发起，但目前设计不是这样)
                # 当前端 addMessage 的逻辑是分开处理 user 和 ai，所以我们也分开添加
                if row["user_msg"]:
                    messages.append({"sender": "user", "text": row["user_msg"]})
                if row["ai_msg"]:
                    messages.append({"sender": "ai", "text": row["ai_msg"]})
        
        # 后端CSV版本中，如果session_id存在但不属于该用户，会继续遍历完再判断
        # 在数据库版本中，WHERE子句直接处理了权限，如果chat_rows为空，
        # 意味着要么session_id不存在，要么不属于该用户。
        if not session_found_for_user:
                logging.warning(f"用户 {username} 尝试加载的会话 {session_id} 不存在或不属于该用户")
                return jsonify({"success": False, "error": "无法加载该会话或会话不存在"}), 404

            # 如果找到了属于该用户的会话记录
        current_session = session_id # 切换后端的当前会话 ID
        logging.info(f"用户 {username} 成功加载会话 {session_id}，后端会话已切换")
        return jsonify({"success": True, "messages": messages})

    except sqlite3.Error as e:
        logging.error(f"加载会话 {session_id} (用户 {username}) 时数据库出错: {e}")
        return jsonify({"success": False, "error": f"加载会话时出错: {e}"}), 500
    except Exception as e: # 捕获其他可能的未知错误
            logging.error(f"加载会话 {session_id} (用户 {username}) 时发生未知错误: {e}")
            return jsonify({"success": False, "error": f"加载会话时出错: {e}"}), 500
 

## 设置
@app.route('/api/setting')
def setting():
    topic = request.args.get('topic') # 从查询参数获取 topic
    request.args.get('topic')
    topic_to_file = {
            "userAgreement": "Userprivacy.txt",
            "userManual": "manual.txt"
        }
    filename = topic_to_file.get(topic)
    file_path = os.path.join(SETTING_DIR, filename)
    with open(file_path,'r',encoding = 'utf-8') as f:
        content = f.read()
        if not os.path.exists(file_path):
            logging.error(f"设置文件未找到: {file_path}")
            # 返回更具体的错误信息给前端
            return jsonify({"success": False, "error": f"请求的内容文件 '{filename}' 未找到"}), 404 # 返回 404 Not Found

    return jsonify({"success": True, "messages": content})

# 根路由 (不变)
@app.route('/')
def index():
    # 总是返回 chat.html，由前端 JS 决定显示登录还是主界面
    return send_from_directory('static', 'chat.html')

# 主程序入口 (修改 webview.start)
if __name__ == '__main__':
    import webview
    # 启动 Flask app (webview 会处理)
    logging.info("启动 Flask 应用和 webview 窗口...")
    window = webview.create_window('FLYINGPIG-AI', app, width=1000, height=700)

    webview.start(debug=True) # <-- 移除 storage_path
