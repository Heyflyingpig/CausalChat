# app.py (Flask后端)
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI
import mysql.connector
from mysql.connector import errorcode

import os
import uuid
from datetime import datetime
import logging
import json 
import asyncio
import atexit
import subprocess
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__, static_folder='static')
current_session = str(uuid.uuid4()) ## 全局会话 ID，现在主要由前端在加载历史时设置
BASE_DIR = os.path.dirname(__file__)
SETTING_DIR = os.path.join(BASE_DIR, "setting")

DATABASE_PATH = None # <-- 修改：不再直接使用 SQLite 文件路径
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")

current_logged_in_user = None

# --- 修改：从 secrets.json 加载 API 配置 ---

current_model = None
BASE_URL = None
apikey = None

MYSQL_HOST = None
MYSQL_USER = None
MYSQL_PASSWORD = None
MYSQL_DATABASE = None


def load_api_config():
        """从 secrets.json 加载 API 和数据库配置。如果缺少关键配置则会失败。"""
        ## 这里的raise是用来手动抛出错误，将捕获的错误传递和返回给调用者
        global current_api, current_model, BASE_URL, apikey
        global MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

        required_keys_zhipu = ["API_KEY", "BASE_URL", "MODEL"] # 假设你的 secrets.json 用的是这些键
        required_keys_db = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]

        try:
            if os.path.exists(SECRETS_PATH):
                with open(SECRETS_PATH, "r", encoding="utf-8") as f:
                    secrets_data = json.load(f)

                    # 检查并加载智谱AI配置
                    for key in required_keys_zhipu:
                        if key not in secrets_data:
                            logging.error(f"关键配置 '{key}' 未在 {SECRETS_PATH} 中找到。")
                            raise ValueError(f"配置错误: {SECRETS_PATH} 中缺少 '{key}'")
                    
                    # 修正这里的键名以匹配你 secrets.json 中的实际键名
                    apikey = secrets_data["API_KEY"] 
                    BASE_URL = secrets_data["BASE_URL"]
                    current_model = secrets_data["MODEL"]
                   

                    # 检查并加载 MySQL 配置
                    for key in required_keys_db:
                        if key not in secrets_data:
                            logging.error(f"关键数据库配置 '{key}' 未在 {SECRETS_PATH} 中找到。")
                            raise ValueError(f"配置错误: {SECRETS_PATH} 中缺少 '{key}'")
                    
                    MYSQL_HOST = secrets_data["MYSQL_HOST"]
                    MYSQL_USER = secrets_data["MYSQL_USER"]
                    MYSQL_PASSWORD = secrets_data["MYSQL_PASSWORD"]
                    MYSQL_DATABASE = secrets_data["MYSQL_DATABASE"]
                    
                    logging.info(f"API 和数据库配置已从 {SECRETS_PATH} 成功加载。")
            else:
                logging.error(f"敏感信息配置文件 {SECRETS_PATH} 不存在。程序无法继续。")
                raise FileNotFoundError(f"必需的配置文件 {SECRETS_PATH} 未找到。")
        
        except json.JSONDecodeError:
            logging.error(f"解析敏感信息配置文件 {SECRETS_PATH} 失败。请检查文件格式。")
            raise
        except ValueError as ve: # 捕获我们自己抛出的 ValueError
            logging.error(str(ve))
            raise
        except Exception as e:
            logging.error(f"加载 API 或数据库配置时发生未知错误: {e}")
            raise

# --- 程序启动时加载 API 和数据库配置 ---
load_api_config()


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
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        logging.info(f"配置已保存到 {CONFIG_PATH}")
    except IOError as e:
        logging.error(f"保存配置文件 {CONFIG_PATH} 时出错: {e}")

# --- 程序启动时加载配置 ---
config = load_config()


# --- 修改：数据库辅助函数 ---
def get_db_connection():
    """创建并返回一个 MySQL 数据库连接。"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE # 连接到指定的数据库
        )
        # logging.debug(f"成功连接到 MySQL 数据库 '{MYSQL_DATABASE}'。")
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            logging.error(f"MySQL 连接错误: 用户 '{MYSQL_USER}' 或密码错误。")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            logging.error(f"MySQL 连接错误: 数据库 '{MYSQL_DATABASE}' 不存在。")
        else:
            logging.error(f"MySQL 连接错误: {err}")
        raise # 重新抛出异常，让调用者知道连接失败

def initialize_database():
    """初始化 MySQL 数据库和表。如果数据库不存在，则尝试创建它。"""
    try:
        # 1. 尝试连接到 MySQL 服务器（不指定数据库，以便创建数据库）
        logging.info(f"尝试连接到 MySQL 服务器: host={MYSQL_HOST}, user={MYSQL_USER}")
        conn_server = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        cursor_server = conn_server.cursor()
        logging.info(f"尝试创建数据库 '{MYSQL_DATABASE}' (如果不存在)...")
        # 使用反引号处理可能包含特殊字符的数据库名，并指定字符集
        cursor_server.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor_server.close()
        conn_server.close()
        logging.info(f"数据库 '{MYSQL_DATABASE}' 已确保存在。")

        # 2. 现在连接到特定的数据库并创建表
        with get_db_connection() as conn: # 这会连接到 MYSQL_DATABASE
            cursor = conn.cursor()
            logging.info(f"开始初始化数据库表于 '{MYSQL_DATABASE}'")
            # 创建用户表
            # AUTO_INCREMENT 是 MySQL 的自增关键字
            # VARCHAR(255) 通常用于存储哈希或短文本
            # ENGINE=InnoDB 和 CHARSET/COLLATE 是推荐的 MySQL 设置
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            logging.info("用户表 'users' 已检查/创建。")

            # 创建聊天记录表
            # TEXT 类型用于存储较长的消息
            # FOREIGN KEY 添加 ON DELETE CASCADE，当用户被删除时，其消息也会被删除
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    user_id INT NOT NULL,
                    user_msg TEXT,
                    ai_msg TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            logging.info("聊天记录表 'chat_messages' 已检查/创建。")

            # --- 修改：处理 chat_messages 表索引 (避免 IF EXISTS / IF NOT EXISTS) ---
            chat_messages_indices_to_drop = [
                "idx_chat_messages_session_userid_time",
                "idx_chat_messages_userid_session_time"
            ]
            for index_name in chat_messages_indices_to_drop:
                try:
                    cursor.execute(f"DROP INDEX {index_name} ON chat_messages")
                    logging.info(f"已删除 chat_messages 表上的旧索引 (如果存在): {index_name}")
                except mysql.connector.Error as err_drop_idx:
                    if err_drop_idx.errno == errorcode.ER_CANT_DROP_FIELD_OR_KEY: # 1091
                        logging.info(f"尝试删除 chat_messages 索引 '{index_name}' 时，索引不存在。")
                    else:
                        logging.warning(f"删除 chat_messages 索引 '{index_name}' 时出现非预期的错误: {err_drop_idx}")
            
            try:
                cursor.execute("""
                    CREATE INDEX idx_chat_messages_session_userid_time 
                    ON chat_messages (session_id, user_id, timestamp)
                """)
                logging.info("已创建索引 idx_chat_messages_session_userid_time")
            except mysql.connector.Error as err_create_idx:
                if err_create_idx.errno == errorcode.ER_DUP_KEYNAME: # 1061: Duplicate key name
                    logging.info("索引 idx_chat_messages_session_userid_time 已存在。")
                else:
                    logging.warning(f"创建索引 idx_chat_messages_session_userid_time 时发生错误: {err_create_idx}")
            
            try:
                cursor.execute("""
                    CREATE INDEX idx_chat_messages_userid_session_time 
                    ON chat_messages (user_id, session_id, timestamp DESC)
                """)
                logging.info("已创建索引 idx_chat_messages_userid_session_time")
            except mysql.connector.Error as err_create_idx:
                if err_create_idx.errno == errorcode.ER_DUP_KEYNAME: # 1061
                    logging.info("索引 idx_chat_messages_userid_session_time 已存在。")
                else:
                    logging.warning(f"创建索引 idx_chat_messages_userid_session_time 时发生错误: {err_create_idx}")
            logging.info("聊天记录表索引已检查/处理。")

            # 创建上传文件表
            # LONGBLOB 用于存储可能较大的文件内容
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    mime_type VARCHAR(100) NOT NULL,
                    file_content LONGBLOB NOT NULL,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            logging.info("上传文件表 'uploaded_files' 已检查/创建。")
            
            # --- 修改：处理 uploaded_files 表索引 (避免 IF EXISTS / IF NOT EXISTS) ---
            uploaded_files_index_to_drop = "idx_uploaded_files_user_id"
            try:
                cursor.execute(f"DROP INDEX {uploaded_files_index_to_drop} ON uploaded_files")
                logging.info(f"已删除 uploaded_files 表上的旧索引 (如果存在): {uploaded_files_index_to_drop}")
            except mysql.connector.Error as err_drop_idx_files:
                if err_drop_idx_files.errno == errorcode.ER_CANT_DROP_FIELD_OR_KEY: # 1091
                    logging.info(f"尝试删除 uploaded_files 索引 '{uploaded_files_index_to_drop}' 时，索引不存在。")
                else:
                    logging.warning(f"删除 uploaded_files 索引 '{uploaded_files_index_to_drop}' 时出现非预期的错误: {err_drop_idx_files}")

            try:
                cursor.execute("""
                    CREATE INDEX idx_uploaded_files_user_id 
                    ON uploaded_files (user_id)
                """)
                logging.info("已创建索引 idx_uploaded_files_user_id")
            except mysql.connector.Error as err_create_idx:
                if err_create_idx.errno == errorcode.ER_DUP_KEYNAME: # 1061
                    logging.info("索引 idx_uploaded_files_user_id 已存在。")
                else:
                    logging.warning(f"创建索引 idx_uploaded_files_user_id 时发生错误: {err_create_idx}")
            logging.info("上传文件表索引已检查/处理。")
            
            conn.commit()
            logging.info(f"数据库表在 '{MYSQL_DATABASE}' 中已成功初始化/验证。")

    except mysql.connector.Error as e:
        logging.error(f"MySQL 数据库初始化失败: {e}")
        if 'CREATE DATABASE' in str(e) and (e.errno == errorcode.ER_DBACCESS_DENIED_ERROR or e.errno == errorcode.ER_ACCESS_DENIED_ERROR):
            logging.error(f"创建数据库 '{MYSQL_DATABASE}' 失败。请检查 MySQL 用户 '{MYSQL_USER}' 是否有 CREATE DATABASE 权限，或者手动创建数据库。")
        elif 'CREATE TABLE' in str(e) and (e.errno == errorcode.ER_TABLEACCESS_DENIED_ERROR or e.errno == errorcode.ER_ACCESS_DENIED_ERROR):
             logging.error(f"在数据库 '{MYSQL_DATABASE}' 中创建表失败。请检查 MySQL 用户 '{MYSQL_USER}' 是否有对该数据库的 CREATE TABLE 权限。")
        raise 
    except Exception as e: # Python 的 AttributeError 等也会被这里捕获
        logging.error(f"数据库初始化过程中发生未知错误: {e}")
        raise

initialize_database() # 程序启动时检查并初始化数据库

# 全局状态

# --- 修改：用户认证相关函数 ---

# 查找用户
def find_user(username):
    try:
        with get_db_connection() as conn:
            # 使用 dictionary=True 使 cursor 返回字典而不是元组，方便按列名访问
            cursor = conn.cursor(dictionary=True)
            # MySQL 使用 %s 作为占位符
            # 这里的数据库语法是说：%s是占位符，username是变量，可以防止恶意注入
            cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
            user_row = cursor.fetchone()
            # cursor.close() # 'with' 语句会自动关闭游标和连接
            if user_row:
                # 返回一个字典，包含 id, username 和 password_hash
                return user_row # user_row 已经是字典了
            return None
    except mysql.connector.Error as e: # <-- 修改异常类型
        logging.error(f"查找用户 '{username}' 时数据库出错: {e}")
        return None
    except Exception as e:
        logging.error(f"查找用户 '{username}' 时发生未知错误: {e}")
        return None


# 注册用户
def register_user(username, hashed_password):
    if find_user(username): # 首先检查用户是否存在
        return False, "用户名已被注册。"

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # MySQL 使用 %s 作为占位符
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                           (username, hashed_password))
            conn.commit()
            # user_id = cursor.lastrowid # 如果需要获取新用户的ID
            # cursor.close()
        logging.info(f"新用户注册成功: {username}")
        return True, "注册成功！"
    except mysql.connector.Error as e: # <-- 修改异常类型
        # MySQL 的 IntegrityError 对于 UNIQUE 约束冲突通常是 ER_DUP_ENTRY (errno 1062)
        if e.errno == errorcode.ER_DUP_ENTRY:
            logging.warning(f"尝试注册已存在的用户名 (数据库约束): {username}")
            return False, "用户名已被注册。"
        logging.error(f"注册用户 '{username}' 时数据库出错: {e}")
        return False, "注册过程中发生服务器错误。"
    except Exception as e:
        logging.error(f"注册用户 '{username}' 时发生未知错误: {e}")
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



# --- 新增: 后台异步事件循环 ---
def start_event_loop(loop: asyncio.AbstractEventLoop):
    """在一个线程中启动并永远运行事件循环"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

# 创建一个全局的事件循环和后台线程
background_loop = asyncio.new_event_loop()
loop_thread = threading.Thread(target=start_event_loop, args=(background_loop,), daemon=True)
loop_thread.start()
logging.info("后台 asyncio 事件循环线程已启动。")
# --- 结束新增 ---

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
        # 修正: 将异步任务提交到后台循环，并等待结果
        future = asyncio.run_coroutine_threadsafe(ai_call(user_input), background_loop)
        response = future.result()  # 这会阻塞当前线程直到异步任务完成

        save_chat(user_id, user_input, response)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        logging.error(f"处理用户 {username} (ID: {user_id}) 消息时出错: {e}", exc_info=True)
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

# --- 核心修改：改造 ai_call 函数为一个独立的、无状态的函数 ---
async def ai_call(text):
    """
    与 LLM 交互，并根据需要通过 MCP 调用工具。
    这是一个异步函数，每次调用都会建立和断开与MCP服务器的连接。
    
     async 关键字告诉Python，这个函数内部可能会包含一些需要“等待”的操作（比如网络请求、文件读写），
    在等待期间，程序可以先去干点别的事，而不是傻等。
    这使得它非常适合处理I/O密集型任务。
    函数内部必须使用 await 关键字来执行这些“等待”操作。
    """
    # 获取CausalChat.py所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建到mcp_server.py的相对路径
    mcp_server_path = os.path.join(current_dir, "CausalChatMCP", "mcp_server.py")
    
    # 使用一个临时的退出堆栈来管理本次调用的资源
    #异步上下文管理器，所有进程都会装载到这个AsyncExitStack()中，方便我们后期清理
    async with AsyncExitStack() as exit_stack:
        mcp_tools = []
        session = None
        try:
            logging.info("ai_call: 准备连接到 MCP 服务器...")
            # 启动mcp_server.py子进程，用于将解释器的绝对路劲配置到该py文件中
            server_params = StdioServerParameters(
                command=sys.executable, #代表了当前正在运行的Python解释器的绝对路径
                args=[mcp_server_path] # 启动mcp_server.py子进程，用于将解释器的绝对路劲配置到该py文件中
            )
            # stdio_client(server_params) 启动 mcp_server.py 子进程，
            # 并建立起两者之间的 stdin/stdout 通信管道。
            # stdio_client成功后返回一个元组，包含了两个关键部分：一个用于从子进程读取数据流，一个用于向子进程写入数据流。我们通过元组解包的方式将它们分别赋给两个变量。
            read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(server_params))
            # 创建一个ClientSession对象，用于与MCP服务器进行交互。
            # 这个对象会处理与MCP服务器的通信细节，比如发送请求、接收响应等。
            # 它内部会使用read_stream和write_stream来与MCP服务器进行数据交换。
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            # 初始化MCP会话，确保MCP服务器已经准备好处理请求。
            await session.initialize()

            logging.info("ai_call: 成功连接 MCP 服务器，正在获取工具列表...")
             # mcp_server.py会返回一个列表，包含他所有被 @mcp.tool() 装饰的函数。
            tools_response = await session.list_tools()
            # 列表推导式子，将所有mcp语法转化为llm语法
            mcp_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            } for tool in tools_response.tools]
            logging.info(f"ai_call: 发现工具: {[tool['function']['name'] for tool in mcp_tools]}")

        except Exception as e:
            logging.error(f"连接或设置 MCP 服务器失败: {e}. 将作为普通聊天继续。")
            # 如果 MCP 失败，mcp_tools 列表将为空，模型将不会尝试使用工具。

        client = OpenAI(base_url=BASE_URL, api_key=apikey)
        messages = [
            {"role": "system", "content": "你是一个有用的工程师助手，可以使用工具来获取额外信息。"},
            {"role": "user", "content": text}
        ]

        # 第一次调用
        logging.info("ai_call: 首次调用 LLM...")
        response = client.chat.completions.create(
            model=current_model,
            messages=messages,
            tools=mcp_tools or None, # 如果列表为空，则不传递 tools 参数
            tool_choice="auto", # 自动选择是否使用工具
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        # 如果工具列表不为空，则调用工具

        if tool_calls and session:
            logging.info(f"LLM 决定调用工具: {[(call.function.name) for call in tool_calls]}")
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                #客户端通过会话，将工具名和解析好的参数字典发送给mcp_server.py子进程
                function_response_obj = await session.call_tool(function_name, function_args)
                function_response_text = function_response_obj.content[0].text
                # 返回所有内容，包括工具名和解析好的参数字典，这里的参数MCP定义的
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response_text,
                })
                # 这里返回的是tool角色
            
            # 第二次调用
            logging.info("ai_call: 携带工具结果，再次调用 LLM...")
            second_response = client.chat.completions.create(
                model=current_model,
                messages=messages,
            )
            return second_response.choices[0].message.content
        else:
            logging.info("ai_call: LLM 未调用工具，直接返回。")
            return response_message.content




## 保存历史文件 ( 添加 user_id 参数和列)
def save_chat(user_id, user_msg, ai_msg): # 修改：username -> user_id
    global current_session # 需要访问全局会话ID
    timestamp_dt = datetime.now() # 获取 datetime 对象
    # MySQL 的 TIMESTAMP 类型可以直接接受 datetime 对象，无需格式化为字符串

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # MySQL 使用 %s 作为占位符
            cursor.execute("""
                INSERT INTO chat_messages (session_id, user_id, user_msg, ai_msg, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """, (current_session, user_id, user_msg, ai_msg, timestamp_dt)) # 修改：使用 user_id 和 datetime 对象
            conn.commit()
            # cursor.close()
        # logging.info(f"聊天记录已保存 (用户 ID: {user_id}, 会话: {current_session})")
    except mysql.connector.Error as e: # <-- 修改异常类型
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
            cursor = conn.cursor(dictionary=True) #  <-- 使用字典游标
            # SQL 查询语句基本兼容，COALESCE 和子查询在 MySQL 中也支持
            # LENGTH(column) 在 MySQL 中也有效
            # %s 是 MySQL 的参数占位符
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
                FROM chat_messages cm_outer
                WHERE user_id = %s
                GROUP BY session_id
                ORDER BY last_time DESC
            """, (user_id,)) 
            session_rows = cursor.fetchall()
            # cursor.close()

        if not session_rows:
            logging.info(f"用户 {username} (ID: {user_id}) 没有会话记录")
            return jsonify([])

        for row in session_rows:
            session_id = row["session_id"]
            # MySQL TIMESTAMP 通常返回 datetime 对象，如果需要字符串，可以格式化
            last_time_obj = row["last_time"] # 已经是字符串格式 "YYYY-MM-DD HH:MM:SS"
            if isinstance(last_time_obj, datetime):
                last_time_str = last_time_obj.strftime("%Y-%m-%d %H:%M:%S")
            else: # 以防万一它已经是字符串了 (SQLite的MAX(timestamp)可能返回字符串)
                last_time_str = str(last_time_obj)
            
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

    except mysql.connector.Error as e: # <-- 修改异常类型
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
            cursor = conn.cursor(dictionary=True) # <-- 使用字典游标
            # %s 是 MySQL 的参数占位符
            cursor.execute("""
                SELECT user_msg, ai_msg FROM chat_messages
                WHERE session_id = %s AND user_id = %s 
                ORDER BY timestamp ASC
            """, (session_id, user_id)) # 修改：传递 user_id
            chat_rows = cursor.fetchall()
            # cursor.close()

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

    except mysql.connector.Error as e: # <-- 修改异常类型
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

@app.route('/api/upload_file', methods=['POST'])
def upload_file():
    username = request.form.get('username')
    if not username:
        logging.warning("CSV上传请求缺少用户名")
        return jsonify({'success': False, 'error': '用户未登录或请求无效'}), 401

    user_data = find_user(username)
    if not user_data:
        logging.warning(f"用户 {username} 尝试上传CSV，但用户不存在")
        return jsonify({'success': False, 'error': '用户认证失败'}), 401
    user_id = user_data['id']
    
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

    # 5. 读取文件内容
    try:
        file_content = file.read() # 将整个文件内容读取为 bytes
    except Exception as e:
        logging.error(f"用户 {username} 上传文件 {original_filename} 时读取内容失败: {e}")
        return jsonify({'success': False, 'error': '读取文件内容失败'}), 500
    
    # 6. 保存到数据库
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # file_content 是 bytes 类型，MySQL 的 BLOB/LONGBLOB 类型可以直接存储
            # %s 是 MySQL 的参数占位符
            cursor.execute("""
                INSERT INTO uploaded_files (user_id, filename, mime_type, file_content)
                VALUES (%s, %s, %s, %s)
            """, (user_id, original_filename, file.mimetype, file_content))
            conn.commit()
            # cursor.close()
        logging.info(f"用户 {username} (ID: {user_id}) 成功上传文件: {original_filename} (MIME: {file.mimetype})")
        return jsonify({'success': True, 'message': f'文件 "{original_filename}" 上传成功！'})
    except mysql.connector.Error as e: # <-- 修改异常类型
        logging.error(f"用户 {username} 保存文件 {original_filename} 到数据库时出错: {e}")
        return jsonify({'success': False, 'error': '保存文件到数据库失败'}), 500
    except Exception as e: # 捕获其他可能的未知错误
        logging.error(f"用户 {username} 上传文件 {original_filename} 时发生未知服务器错误: {e}")
        return jsonify({'success': False, 'error': '上传文件时发生服务器内部错误'}), 500

# 根路由 (不变)
@app.route('/')
def index():
    # 总是返回 chat.html，由前端 JS 决定显示登录还是主界面
    return send_from_directory('static', 'chat.html')

# 主程序入口
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
    