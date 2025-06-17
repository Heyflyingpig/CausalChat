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
import hashlib

# --- 新增：为 Windows 设置 asyncio 事件循环策略 ---
# 在 Windows 上，默认的 asyncio 事件循环 (SelectorEventLoop) 不支持子进程。
# MCP 客户端需要通过子进程启动服务器，因此我们必须切换到 ProactorEventLoop。
# 这行代码必须在任何 asyncio 操作（尤其是创建事件循环）之前执行。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# ---------------------------------------------

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__, static_folder='static')
# --- 新增：为 Flask Sessions 设置密钥 ---
# 会话管理（例如登录状态）需要一个密钥来对 cookie 进行加密签名。
# 这是支持多用户并发会话的基础。
# 我们将从 secrets.json 文件中加载它。
app.secret_key = None 
# ------------------------------------

BASE_DIR = os.path.dirname(__file__)
# --- 新增：确保图表目录存在 ---
os.makedirs(os.path.join(BASE_DIR, 'static', 'generated_graphs'), exist_ok=True)
# -----------------------------
SETTING_DIR = os.path.join(BASE_DIR, "setting")

DATABASE_PATH = None # <-- 修改：不再直接使用 SQLite 文件路径
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")

# --- 修改：全局状态管理 ---
# 将 MCP 和事件循环相关的状态集中管理
mcp_session: ClientSession | None = None
mcp_tools: list = []
mcp_process_stack = AsyncExitStack()
background_loop: asyncio.AbstractEventLoop | None = None
# -------------------------


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

        # --- 新增：添加 SECRET_KEY 到必需列表 ---
        required_keys_app = ["SECRET_KEY"]
        # ------------------------------------
        required_keys_zhipu = ["API_KEY", "BASE_URL", "MODEL"] # 假设你的 secrets.json 用的是这些键
        required_keys_db = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]

        try:
            if os.path.exists(SECRETS_PATH):
                with open(SECRETS_PATH, "r", encoding="utf-8") as f:
                    secrets_data = json.load(f)

                    # --- 新增：检查并加载应用密钥 ---
                    for key in required_keys_app:
                        if key not in secrets_data:
                            logging.error(f"关键应用配置 '{key}' 未在 {SECRETS_PATH} 中找到。")
                            raise ValueError(f"配置错误: {SECRETS_PATH} 中缺少 '{key}'")
                    app.secret_key = secrets_data["SECRET_KEY"]
                    # ----------------------------------

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
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                return config_data
        else:
            logging.info(f"配置文件 {CONFIG_PATH} 不存在，将创建。")
            # 文件不存在，返回默认空配置
            return {} # 返回一个空字典或包含其他非用户配置的字典
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"读取配置文件 {CONFIG_PATH} 时出错: {e}。将使用默认配置。")
        # 文件损坏或读取错误，同样返回默认
        return {}

def save_config(config_data):
    """将配置数据保存到 config.json"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            # --- 移除：确保不保存用户信息 ---
            # 我们从 config_data 中移除 logged_in_user (如果存在)，以防旧代码调用
            config_data.pop("logged_in_user", None)
            # --------------------------------
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

def check_database_readiness():
    """检查数据库是否已准备就绪，包括所需的表和基本连接性。
    这个函数不会创建或修改数据库结构，只是检查现有配置是否正确。
    如果数据库未初始化，请先运行 database_init.py 脚本。
    """
    try:
        logging.info(f"检查数据库连接和表结构就绪状态...")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 检查新的、优化的表是否存在
            required_tables = ['users', 'sessions', 'chat_messages', 'chat_attachments', 'uploaded_files', 'archived_sessions']
            cursor.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{MYSQL_DATABASE}'
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                error_msg = f"数据库表缺失: {list(missing_tables)}。请先运行 'python database_init.py' 初始化数据库。"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 简单的连接性测试
            cursor.execute("SELECT 1")
            test_result = cursor.fetchone()
            if not test_result or test_result[0] != 1:
                raise RuntimeError("数据库连接测试失败")
            
            logging.info(f"优化后的数据库 '{MYSQL_DATABASE}' 就绪检查通过。所有必需表已存在。")
            return True
            
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_BAD_DB_ERROR:
            error_msg = f"数据库 '{MYSQL_DATABASE}' 不存在。请先运行 'python database_init.py' 创建和初始化数据库。"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        elif e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            error_msg = f"无法访问数据库。请检查用户 '{MYSQL_USER}' 的权限配置。"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logging.error(f"数据库就绪性检查失败: {e}")
            raise
    except Exception as e:
        logging.error(f"数据库就绪性检查过程中发生未知错误: {e}")
        raise

# --- 修改：在应用启动时进行数据库就绪性检查而不是初始化 ---
try:
    check_database_readiness()
except RuntimeError as e:
    logging.critical(f"数据库未就绪，应用无法启动: {e}")
    print(f"\n❌ 数据库错误: {e}")
    print("请先运行以下命令初始化数据库:")
    print("python database_init.py")
    sys.exit(1)
except Exception as e:
    logging.critical(f"数据库检查失败，应用无法启动: {e}")
    print(f"\n❌ 数据库检查失败: {e}")
    sys.exit(1)

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

def get_chat_history(session_id: str, user_id: int, limit: int = 100) -> list:
    """从数据库获取指定会话的最近聊天记录。"""
    history = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            # 获取最近的 'limit' 条记录
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
    # --- 重构：使用 Flask Session 进行会话管理 ---
    from flask import session

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
        
        # --- 核心修改：在 Session 中存储用户信息 ---
        session.clear() # 先清除旧的会话数据
        session['user_id'] = user_data['id']
        session['username'] = user_data['username']
        # Session 会自动通过浏览器 cookie 维护状态，不再需要文件
        # -----------------------------------------
        
        return jsonify({'success': True, 'username': username})
    else:
        logging.warning(f"用户登录失败（密码错误）: {username}")
        return jsonify({'success': False, 'error': '密码错误'}), 401 # 401 Unauthorized

# 登出
@app.route('/api/logout', methods=['POST'])
def handle_logout():
    # --- 重构：使用 Flask Session ---
    from flask import session
    
    # 从会话中获取用户名用于日志记录
    username = session.get('username', '未知用户')
    logging.info(f"用户 {username} 请求退出登录")

    # --- 核心修改：清除会话 ---
    session.clear()
    # -------------------------

    return jsonify({'success': True})

# --- 新增：检查认证状态 API 端点 ---
@app.route('/api/check_auth', methods=['GET'])
def check_auth():
    """检查当前后端记录的登录状态"""
    # --- 重构：检查 Flask Session ---
    from flask import session
    if 'user_id' in session and 'username' in session:
        username = session['username']
        logging.debug(f"检查认证状态：用户 '{username}' (通过会话) 已登录")
        return jsonify({'isLoggedIn': True, 'username': username})
    else:
        logging.debug("检查认证状态：无有效会话")
        return jsonify({'isLoggedIn': False})



# --- 全面重构：MCP生命周期管理与后台事件循环 ---

async def initialize_mcp_connection(ready_event: threading.Event):
    """
    在应用启动时启动MCP服务器并建立一个持久的会话。
    完成后通过 event 通知主线程。
    """
    global mcp_session, mcp_tools
    logging.info("正在初始化持久 MCP 连接...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_server_path = os.path.join(current_dir, "CausalChatMCP", "mcp_server.py")
        
        server_params = StdioServerParameters(command=sys.executable, args=[mcp_server_path])
        
        read_stream, write_stream = await mcp_process_stack.enter_async_context(stdio_client(server_params))
        session = await mcp_process_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        
        tools_response = await session.list_tools()
        mcp_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        } for tool in tools_response.tools]

        mcp_session = session
        logging.info(f"MCP服务器连接成功，会话已激活。发现工具: {[tool['function']['name'] for tool in mcp_tools]}")
        
    except Exception as e:
        logging.error(f"严重错误：应用启动时初始化MCP连接失败: {e}", exc_info=True)
        mcp_session = None
    finally:
        logging.info("MCP 初始化过程结束，通知主线程。")
        ready_event.set()

def shutdown_mcp_connection():
    """在应用退出时，通过atexit钩子优雅地关闭MCP服务器子进程。"""
    if background_loop and background_loop.is_running():
        logging.info("请求关闭 MCP 服务器...")
        future = asyncio.run_coroutine_threadsafe(mcp_process_stack.aclose(), background_loop)
        try:
            future.result(timeout=5)
            logging.info("MCP 服务器已成功关闭。")
        except Exception as e:
            logging.error(f"关闭 MCP 服务器时出错: {e}")
    else:
        logging.warning("无法关闭 MCP 服务器：事件循环未运行。")

def start_event_loop(loop: asyncio.AbstractEventLoop, ready_event: threading.Event):
    """在一个线程中启动事件循环，并在启动时安排MCP初始化。"""
    global background_loop
    asyncio.set_event_loop(loop)
    background_loop = loop
    
    loop.create_task(initialize_mcp_connection(ready_event))
    
    logging.info("后台事件循环已启动，MCP 初始化任务已安排。")
    loop.run_forever()

# --- 重构结束 ---


# 利用flask的jsonify的框架，将后端处理转发到前端
@app.route('/api/send', methods=['POST'])
def handle_message():
    # --- 重构：从 Session 获取用户身份 ---
    from flask import session
    if 'user_id' not in session or 'username' not in session:
        return jsonify({'success': False, 'error': '用户未登录或会话已过期'}), 401
    
    user_id = session['user_id']
    username = session['username']
    # -----------------------------------

    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id') # <--- 新增：从前端获取会话ID

    if not session_id:
        logging.error(f"用户 {username} (ID: {user_id}) 发送消息时缺少 session_id")
        return jsonify({'success': False, 'error': '请求无效，缺少会话ID'}), 400

    logging.info(f"用户 {username} (ID: {user_id}) 在会话 {session_id} 中发送消息: {user_input[:50]}...")

    try:
        # --- 核心修改：将 session_id 传递给 ai_call ---
        future = asyncio.run_coroutine_threadsafe(ai_call(user_input, user_id, username, session_id), background_loop)
        response = future.result()  # 这会阻塞当前线程直到异步任务完成

        # --- 核心修改：显式传递 session_id，不再使用全局变量 ---
        save_chat(user_id, session_id, user_input, response)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        logging.error(f"处理用户 {username} (ID: {user_id}) 消息时出错: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'处理消息时出错: {e}'}), 500

@app.route('/api/new_chat',methods=['POST'])
def new_chat():
    from flask import session
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '用户未登录'}), 401
    
    user_id = session['user_id']
    username = session.get('username', '未知用户')
    new_session_id = str(uuid.uuid4())
    # 为新会话创建一个默认标题
    title = f"新对话 - {datetime.now().strftime('%m-%d %H:%M')}"

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 在 sessions 表中创建一条新记录
            cursor.execute(
                "INSERT INTO sessions (id, user_id, title) VALUES (%s, %s, %s)",
                (new_session_id, user_id, title)
            )
            conn.commit()
        
        logging.info(f"用户 {username} (ID: {user_id}) 创建新会话: {new_session_id}")
        return jsonify({'success': True, 'new_session_id': new_session_id})
    except mysql.connector.Error as e:
        logging.error(f"为用户 {username} 创建新会话时数据库出错: {e}")
        return jsonify({'success': False, 'error': '创建新会话失败'}), 500

# --- 核心修改：重构 ai_call 函数以使用持久连接 ---
async def ai_call(text, user_id, username, session_id):
    """
    使用全局持久化的 MCP 会话与 LLM 交互并调用工具。
    不再在每次调用时创建或销毁连接。
    """
    # --- 修改：上下文获取逻辑现在直接使用传入的 session_id ---
    history_messages = get_chat_history(session_id, user_id, limit=20)
    # ----------------------------------------------------

    client = OpenAI(base_url=BASE_URL, api_key=apikey)
    
    # 检查MCP会话是否在启动时成功建立
    if not mcp_session:
        logging.error("ai_call: MCP会话不可用。将作为普通聊天继续，不使用工具。")
        
        messages = [{"role": "system", "content": "你是一个有用的工程师助手，请根据上下文进行回复。"}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": text})
        
        response = client.chat.completions.create(
            model=current_model,
            messages=messages
        )
        # 保证返回格式一致
        return {"type": "text", "summary": response.choices[0].message.content}

    # ---- 使用已建立的会话 ----
    messages = [
        {"role": "system", "content": "你是一个有用的工程师助手，可以使用工具来获取额外信息。"
        "你的主要任务是分析工具返回的JSON数据，并以详细的自然语言（根据用户给你的语言风格）向用户总结关键发现。"
        "现在有一下几个要求：1.不要在你的回答中逐字重复整个JSON数据。请根据上下文进行回复。2. 请使用Markdown格式（例如，使用项目符号、加粗、表格等）来组织你的回答 3. 回答需要依照因果推断相关的知识和术语进行回答"},
    ]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": text})

    # 第一次调用
    # 这里可选择是否调用MCP，未来可添加功能
    logging.info(f"ai_call: 首次调用 LLM，包含 {len(history_messages)} 条历史消息...")
    response = client.chat.completions.create(
        model=current_model,
        messages=messages,
        tools=mcp_tools or None,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        logging.info(f"LLM 决定调用工具: {[call.function.name for call in tool_calls]}")
        messages.append(response_message)

        # 注意：当前设计只处理第一个工具调用
        tool_call = tool_calls[0]
        # 该工具代表调用mcp的意图
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # --- 核心修改：将从 session 中获得的安全 username 传递给工具function_args ---
        function_args['username'] = username
        logging.info(f"调用工具 '{function_name}'，增强后参数: {function_args}")
        
        function_response_obj = await mcp_session.call_tool(function_name, function_args)
        function_response_text = function_response_obj.content[0].text
        
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response_text,
        })
        
        # 第二次调用，让LLM总结工具结果
        logging.info("ai_call: 携带工具结果，再次调用 LLM...")
        second_response = client.chat.completions.create(
            model=current_model,
            messages=messages,
        )
        summary_text = second_response.choices[0].message.content

        # 如果是因果分析工具，则返回结构化数据
        if function_name == 'perform_causal_analysis':
            try:
                analysis_data = json.loads(function_response_text)
                if analysis_data.get("success"):
                    logging.info("因果分析成功，返回结构化数据和总结。")
                    return {
                        "type": "causal_graph",
                        "summary": summary_text,
                        "data": analysis_data.get("data")
                    }
            except json.JSONDecodeError:
                logging.error("无法解析来自因果分析工具的JSON响应。")
        
        # 对于其他工具或失败的分析，只返回文本总结
        logging.info("返回纯文本总结。")
        return {"type": "text", "summary": summary_text}

    else:
        logging.info("ai_call: LLM 未调用工具，直接返回。")
        # 保证返回格式一致
        return {"type": "text", "summary": response_message.content}


## 保存历史文件 ( 添加 user_id 参数和列)
def save_chat(user_id, session_id, user_msg, ai_response):
    """
    将用户和AI的交互保存到新的优化数据库结构中。
    - 在 chat_messages 中为用户和AI分别创建记录。
    - 如果AI响应包含附件，则在 chat_attachments 中创建记录。
    - 更新 sessions 表的元数据。
    """
    timestamp_dt = datetime.now()

    try:
        with get_db_connection() as conn:
            # --- 修改：使用字典游标并预先获取会话信息 ---
            cursor = conn.cursor(dictionary=True)

            # 0. 检查这是否是第一次保存消息，以决定是否更新标题
            cursor.execute("SELECT message_count FROM sessions WHERE id = %s", (session_id,))
            session_data = cursor.fetchone()
            is_first_message = session_data and session_data['message_count'] == 0
            
            # 1. 保存用户消息
            sql_user = """
                INSERT INTO chat_messages (session_id, user_id, message_type, content, created_at)
                VALUES (%s, %s, 'user', %s, %s)
            """
            cursor.execute(sql_user, (session_id, user_id, user_msg, timestamp_dt))
            
            # 2. 保存AI消息
            ai_content = ""
            has_attachment = False
            attachment_content = None
            attachment_type = 'other'

            if isinstance(ai_response, dict):
                ai_content = ai_response.get('summary', json.dumps(ai_response, ensure_ascii=False))
                if ai_response.get('type') == 'causal_graph' and 'data' in ai_response:
                    has_attachment = True
                    attachment_type = 'causal_graph'
                    attachment_content = json.dumps(ai_response, ensure_ascii=False) # 保存完整响应
            elif isinstance(ai_response, str):
                ai_content = ai_response
            else:
                ai_content = json.dumps(ai_response, ensure_ascii=False)

            sql_ai = """
                INSERT INTO chat_messages (session_id, user_id, message_type, content, has_attachment, created_at)
                VALUES (%s, %s, 'ai', %s, %s, %s)
            """
            cursor.execute(sql_ai, (session_id, user_id, ai_content, has_attachment, timestamp_dt))
            ai_message_id = cursor.lastrowid # 获取AI消息的ID，用于关联附件

            # 3. 如果有附件，保存到 chat_attachments
            if has_attachment and attachment_content:
                sql_attachment = """
                    INSERT INTO chat_attachments (message_id, attachment_type, content, created_at)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql_attachment, (ai_message_id, attachment_type, attachment_content, timestamp_dt))

            # --- 修改：根据是否为第一条消息，决定是否更新标题 ---
            if is_first_message:
                # 4a. 更新会话，包括新标题
                new_title = user_msg[:8] # 截取前20个字符作为标题
                new_title = new_title + "..." if len(user_msg) > 8 else new_title
                sql_update_session = """
                    UPDATE sessions 
                    SET title = %s, last_activity_at = %s, message_count = message_count + 2
                    WHERE id = %s AND user_id = %s
                """
                cursor.execute(sql_update_session, (new_title, timestamp_dt, session_id, user_id))
            else:
                # 4b. 只更新活动时间和消息数
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


# 会话管理接口 (**修改：** 过滤用户)
@app.route('/api/sessions')
def get_sessions():
    from flask import session
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

@app.route('/api/files')
# 获取文件列表
def get_file_list():
    from flask import session
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
        
            

# 加载特定会话内容 (**修改：** 增加用户验证)
@app.route('/api/load_session')
def load_session_content():
    from flask import session
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "用户未登录或会话已过期"}), 401
    
    user_id = session['user_id']
    username = session['username']

    session_id = request.args.get('session')

    if not session_id:
        return jsonify({"success": False, "error": "缺少 session ID"}), 400

    logging.info(f"用户 {username} (ID: {user_id}) 请求加载会话: {session_id} (新版逻辑)")

    messages = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            # 首先验证用户是否有权访问此会话
            cursor.execute("SELECT id FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            if not cursor.fetchone():
                logging.warning(f"用户 {username} 尝试访问无权或不存在的会话 {session_id}")
                return jsonify({"success": False, "error": "无法加载该会话或会话不存在"}), 404

            # 获取所有消息，并左连接附件表
            cursor.execute("""
                SELECT 
                    cm.id, cm.message_type, cm.content, cm.has_attachment,
                    ca.content as attachment_content
                FROM chat_messages cm
                LEFT JOIN chat_attachments ca ON cm.id = ca.message_id AND cm.has_attachment = TRUE
                WHERE cm.session_id = %s
                ORDER BY cm.created_at ASC
            """, (session_id,))
            chat_rows = cursor.fetchall()

        for row in chat_rows:
            sender = "user" if row["message_type"] == 'user' else "ai"
            
            # 如果是AI消息，且有附件，则优先使用附件内容
            if sender == "ai" and row["has_attachment"] and row["attachment_content"]:
                try:
                    # 附件内容本身就是完整的结构化JSON
                    structured_content = json.loads(row["attachment_content"])
                    messages.append({"sender": "ai", "text": structured_content})
                except (json.JSONDecodeError, TypeError):
                    logging.warning(f"无法解析附件内容，回退到文本。Message ID: {row['id']}")
                    messages.append({"sender": "ai", "text": row["content"]})
            else:
                # 对于用户消息或没有附件的AI消息，直接使用content
                messages.append({"sender": sender, "text": row["content"]})
        
        logging.info(f"用户 {username} 成功加载会话 {session_id}")
        return jsonify({"success": True, "messages": messages})

    except mysql.connector.Error as e:
        logging.error(f"加载会话 {session_id} (用户 {username}) 时数据库出错: {e}")
        return jsonify({"success": False, "error": f"加载会话时出错: {e}"}), 500
    except Exception as e:
        logging.error(f"加载会话 {session_id} (用户 {username}) 时发生未知错误: {e}")
        return jsonify({"success": False, "error": f"加载会话时出错: {e}"}), 500
 
@app.route('/api/change_session', methods=['POST'])
def change_session():
    # --- 新增：用户认证检查 ---
    from flask import session
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "用户未登录或会话已过期"}), 401
    
    user_id = session['user_id']
    
    # --- 修改：从 POST 请求的 JSON body 中获取数据 ---
    data = request.json
    title = data.get('title')
    session_id = data.get('session_id')

    if not title or not session_id:
        return jsonify({"success": False, "error": "缺少标题或会话ID"}), 400

    try:
        with get_db_connection() as conn:
            # --- 修改：增加 user_id 条件以确保安全 ---
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET title = %s WHERE id = %s AND user_id = %s",
                (title, session_id, user_id)
            )
            conn.commit()
            
            # --- 新增：检查是否有行被更新，处理权限不足或会话不存在的情况 ---
            if cursor.rowcount == 0:
                logging.warning(f"用户 {user_id} 尝试更新不属于自己的会话 {session_id} 或会话不存在。")
                return jsonify({"success": False, "error": "无法更新会话，权限不足或会话不存在"}), 404
            
        logging.info(f"用户 {user_id} 成功将会话 {session_id} 的标题更新为 '{title}'")
        return jsonify({"success": True, "message": "会话标题已更新"})
    except mysql.connector.Error as e:
        logging.error(f"更新会话标题时数据库出错 (用户ID: {user_id}, 会话ID: {session_id}): {e}")
        return jsonify({"success": False, "error": "更新会话标题时数据库出错"}), 500

@app.route('/api/delete_session', methods=['POST'])
def delete_session():
    # --- 核心修改：安全和完整的删除逻辑 ---
    from flask import session
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
            
            # 0. 验证用户是否有权限删除此会话
            cursor.execute("SELECT id FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            if not cursor.fetchone():
                conn.rollback() # 回滚事务
                logging.warning(f"用户 {user_id} 尝试删除无权或不存在的会话 {session_id}")
                return jsonify({"success": False, "error": "无法删除该会话，权限不足或会话不存在"}), 404

            # 1. 删除与该会话相关的附件 (通过连接 chat_messages)
            # 这是为了处理 chat_attachments 和 chat_messages 之间没有直接外键的情况
            sql_delete_attachments = """
                DELETE ca FROM chat_attachments ca
                JOIN chat_messages cm ON ca.message_id = cm.id
                WHERE cm.session_id = %s
            """
            cursor.execute(sql_delete_attachments, (session_id,))
            logging.info(f"为会话 {session_id} 删除了 {cursor.rowcount} 个附件")

            # 2. 删除该会话的所有聊天记录
            cursor.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
            logging.info(f"为会话 {session_id} 删除了 {cursor.rowcount} 条聊天记录")

            # 3. 删除会话本身
            cursor.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            
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

@app.route('/api/delete_file', methods=['POST'])
def delete_file():
    from flask import session
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

## 上传文件
@app.route('/api/upload_file', methods=['POST'])
def upload_file():
    # --- 重构：从 Session 获取用户身份 ---
    from flask import session
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '用户未登录或会话已过期'}), 401
    
    user_id = session['user_id']
    username = session.get('username', '未知用户') # 用于日志
    
    session_id = request.form.get('session_id')
    if not session_id:
        logging.warning(f"用户 {username} 上传文件请求缺少 session_id")
        return jsonify({'success': False, 'error': '请求无效，缺少会话ID'}), 400
    # ------------------------------------

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

# 根路由 (不变)
@app.route('/')
def index():
    # 总是返回 chat.html，由前端 JS 决定显示登录还是主界面
    return send_from_directory('static', 'chat.html')

# 主程序入口
if __name__ == '__main__':
    # 注册应用退出时的清理函数
    atexit.register(shutdown_mcp_connection)

    # --- 启动后台事件循环并等待 MCP 就绪 ---
    mcp_ready_event = threading.Event()
    
    background_event_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(
        target=start_event_loop, 
        args=(background_event_loop, mcp_ready_event),
        daemon=True
    )
    loop_thread.start()
    
    logging.info("主线程正在等待 MCP 初始化...")
    is_ready = mcp_ready_event.wait(timeout=30.0)

    if not is_ready:
        logging.critical("MCP 服务在30秒内未能完成初始化。应用即将退出。")
        if background_loop and background_loop.is_running():
            asyncio.run_coroutine_threadsafe(mcp_process_stack.aclose(), background_loop)
        sys.exit(1)

    if not mcp_session:
        logging.critical("MCP 初始化完成但会话无效。应用即将退出。")
        sys.exit(1)
        
    logging.info("MCP 服务已就绪，启动 Flask Web 服务器...")
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)
    