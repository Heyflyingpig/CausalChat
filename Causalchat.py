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
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import threading
import hashlib
import bcrypt

try:
    from config.settings import settings
except (ValueError, FileNotFoundError) as e:
    logging.critical(f"无法加载应用配置，程序终止。错误: {e}")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from typing import Any, Type, List
from pydantic import BaseModel, create_model, Field
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# 在 Windows 上，默认的 asyncio 事件循环 (SelectorEventLoop) 不支持子进程。
# MCP 客户端需要通过子进程启动服务器，因此我们必须切换到 ProactorEventLoop。
# 这行代码必须在任何 asyncio 操作（尤其是创建事件循环）之前执行。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 再次获取根logger并显式设置级别，以防被其他库覆盖
logging.getLogger().setLevel(logging.INFO)


app = Flask(__name__, static_folder='static')
#  为 Flask Sessions 设置密钥 
# 会话管理（例如登录状态）需要一个密钥来对 cookie 进行加密签名。
# 这是支持多用户并发会话的基础。
# 我们将从新的配置模块中加载它。
app.secret_key = settings.SECRET_KEY

BASE_DIR = os.path.dirname(__file__)
#这里是创建文件makedirs，后面的参数意思是确定文件目录存在
os.makedirs(os.path.join(BASE_DIR, 'static', 'generated_graphs'), exist_ok=True)
SETTING_DIR = os.path.join(BASE_DIR, "setting")

# 将 MCP 和事件循环,llm和rag链的相关的状态集中管理

mcp_session: ClientSession | None = None
mcp_tools: list = []
mcp_process_stack = AsyncExitStack()
background_loop: asyncio.AbstractEventLoop | None = None
rag_chain = None
llm = None
agent_graph = None


def initialize_llm():
    """在应用启动时初始化全局LLM实例。"""
    global llm, agent_graph
    # 使用新的配置对象
    if not all([settings.MODEL, settings.BASE_URL, settings.API_KEY]):
        logging.error("LLM 配置不完整，无法初始化。")
        return False
    
    logging.info(f"正在初始化 LLM 模型: {settings.MODEL}")
    llm = ChatOpenAI(
        model=settings.MODEL,
        base_url=settings.BASE_URL,
        api_key=settings.API_KEY,
        temperature=0,
        streaming=False,
    )
    logging.info("LLM 实例初始化成功。")
    #  Agent Graph 的创建将推迟到 MCP 连接就绪后 
    return True


def get_db_connection():
    """创建并返回一个 MySQL 数据库连接。"""
    try:
        # 使用新的配置对象
        conn = mysql.connector.connect(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE # 连接到指定的数据库
        )
        # logging.debug(f"成功连接到 MySQL 数据库 '{settings.MYSQL_DATABASE}'。")
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            logging.error(f"MySQL 连接错误: 用户 '{settings.MYSQL_USER}' 或密码错误。")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            logging.error(f"MySQL 连接错误: 数据库 '{settings.MYSQL_DATABASE}' 不存在。")
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
                WHERE table_schema = '{settings.MYSQL_DATABASE}'
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
            
            logging.info(f"数据库 '{settings.MYSQL_DATABASE}' 就绪检查通过。所有必需表已存在。")
            return True
            
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_BAD_DB_ERROR:
            error_msg = f"数据库 '{settings.MYSQL_DATABASE}' 不存在。请先运行 'python database_init.py' 创建和初始化数据库。"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        elif e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            error_msg = f"无法访问数据库。请检查用户 '{settings.MYSQL_USER}' 的权限配置。"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logging.error(f"数据库就绪性检查失败: {e}")
            raise
    except Exception as e:
        logging.error(f"数据库就绪性检查过程中发生未知错误: {e}")
        raise

# 在应用启动时进行数据库就绪性检查而不是初始化 
try:
    check_database_readiness()
except RuntimeError as e:
    logging.critical(f"数据库未就绪，应用无法启动: {e}")
    print(f"\n 数据库错误: {e}")
    print("请先运行以下命令初始化数据库:")
    print("python database_init.py")
    sys.exit(1)
except Exception as e:
    logging.critical(f"数据库检查失败，应用无法启动: {e}")
    print(f"\n 数据库检查失败: {e}")
    sys.exit(1)

# 全局状态


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



# 获取注册值,检查注册值
@app.route('/api/register', methods=['POST'])
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
@app.route('/api/login', methods=['POST'])
def handle_login():
    """
    处理用户登录请求。
    接收前端通过HTTPS发送的明文密码，使用bcrypt进行验证。
    """
    #  重构：使用 Flask Session 进行会话管理 
    from flask import session

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
@app.route('/api/logout', methods=['POST'])
def handle_logout():
    #  重构：使用 Flask Session 
    from flask import session
    
    # 从会话中获取用户名用于日志记录
    username = session.get('username', '未知用户')
    logging.info(f"用户 {username} 请求退出登录")

    #  核心修改：清除会话 
    session.clear()
    # -

    return jsonify({'success': True})

#  检查认证状态 API 端点 
@app.route('/api/check_auth', methods=['GET'])
def check_auth():
    """检查当前后端记录的登录状态"""
    #  重构：检查 Flask Session 
    from flask import session
    if 'user_id' in session and 'username' in session:
        username = session['username']
        logging.debug(f"检查认证状态：用户 '{username}' (通过会话) 已登录")
        return jsonify({'isLoggedIn': True, 'username': username})
    else:
        logging.debug("检查认证状态：无有效会话")
        return jsonify({'isLoggedIn': False})



# MCP生命周期管理与后台事件循环

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


# 获取发送的各种值
# 暂时闲置，未来可能使用
@app.route('/api/send', methods=['POST'])
def handle_message():
    from flask import session
    if 'user_id' not in session or 'username' not in session:
        return jsonify({'success': False, 'error': '用户未登录或会话已过期'}), 401
    
    user_id = session['user_id']
    username = session['username']
    # --

    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id') # < 从前端获取会话ID

    if not session_id:
        logging.error(f"用户 {username} (ID: {user_id}) 发送消息时缺少 session_id")
        return jsonify({'success': False, 'error': '请求无效，缺少会话ID'}), 400

    logging.info(f"用户 {username} (ID: {user_id}) 在会话 {session_id} 中发送消息: {user_input[:50]}...")

    try:
        # 等待异步操作完成
        future = asyncio.run_coroutine_threadsafe(ai_call(user_input, user_id, username, session_id), background_loop)
        response = future.result()  # 这会阻塞当前线程直到异步任务完成

        #  核心修改：显式传递 session_id，不再使用全局变量 
        save_chat(user_id, session_id, user_input, response)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        logging.error(f"处理用户 {username} (ID: {user_id}) 消息时出错: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'处理消息时出错: {e}'}), 500

@app.route('/api/send_stream', methods=['POST'])
def handle_message_stream():
    """
    SSE流式端点：实时推送agent节点执行事件
    """
    from flask import session, stream_with_context, Response
    
    # 认证检查
    if 'user_id' not in session or 'username' not in session:
        return jsonify({'success': False, 'error': '用户未登录或会话已过期'}), 401
    
    user_id = session['user_id']
    username = session['username']
    
    # 获取请求参数
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id')
    
    if not session_id:
        logging.error(f"用户 {username} (ID: {user_id}) 发送流式消息时缺少 session_id")
        return jsonify({'success': False, 'error': '请求无效，缺少会话ID'}), 400
    
    logging.info(f"[流式] 用户 {username} (ID: {user_id}) 在会话 {session_id} 中发送消息: {user_input[:50]}...")
    
    def generate():
        """
        生成器函数，用于SSE流式传输
        """
        try:
            # 在后台事件循环中运行异步生成器
            loop = background_loop
            queue = asyncio.Queue()
            
            async def producer():
                try:
                    async for event_data in ai_call_stream(user_input, user_id, username, session_id):
                        # 放入队列
                        await queue.put(event_data)
                except Exception as e:
                    logging.error(f"[SSE] 生成器错误: {e}", exc_info=True)
                    error_event = f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
                    await queue.put(error_event)
                finally:
                    await queue.put(None)  # 结束标记
            
            # 启动生产者
            future = asyncio.run_coroutine_threadsafe(producer(), loop)
            
            # 从队列中读取并yield
            while True:
                # 使用 run_coroutine_threadsafe 获取队列中的数据
                get_future = asyncio.run_coroutine_threadsafe(queue.get(), loop)
                event_data = get_future.result(timeout=300)  # 5分钟超时
                
                if event_data is None:  # 结束标记
                    break
                    
                yield event_data
                
                # 如果是final_result或interrupt，保存聊天记录
                try:
                    if event_data.startswith("data: "):
                        event_json = json.loads(event_data[6:])  # 去掉 "data: " 前缀
                        
                        if event_json.get("type") == "final_result":
                            # 保存聊天记录
                            response_data = event_json.get("data", {})
                            save_chat(user_id, session_id, user_input, response_data)
                            logging.info(f"[SSE] 已保存聊天记录到数据库")
                        
                        elif event_json.get("type") == "interrupt":
                            # interrupt场景也保存
                            response_data = {
                                "type": "human_input_required",
                                "summary": event_json.get("message", "")
                            }
                            save_chat(user_id, session_id, user_input, response_data)
                            logging.info(f"[SSE] 已保存interrupt聊天记录")
                except Exception as e:
                    logging.warning(f"[SSE] 保存聊天记录时出错: {e}")
            
            # 等待生产者完成
            future.result(timeout=5)
            
        except Exception as e:
            logging.error(f"[SSE] 流式传输错误: {e}", exc_info=True)
            error_event = f"data: {json.dumps({'type': 'error', 'message': f'流式传输错误: {str(e)}'}, ensure_ascii=False)}\n\n"
            yield error_event
    
    # 返回SSE响应
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@app.route('/api/new_chat',methods=['POST'])

def new_chat():
    from flask import session
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '用户未登录'}), 401
    
    user_id = session['user_id']
    username = session.get('username', '未知用户')
    new_session_id = str(uuid.uuid4())

    # 核心修改：不再立即创建数据库记录，只生成session_id
    # 会话记录将在用户发送第一条消息时通过 save_chat() 函数创建
    logging.info(f"用户 {username} (ID: {user_id}) 生成新会话ID: {new_session_id} (延迟创建)")
    return jsonify({'success': True, 'new_session_id': new_session_id})

# 初始化rag链
def initialize_rag_system():
    """在应用启动时加载向量数据库并构建RAG链。"""
    global rag_chain, llm
    logging.info("正在初始化 RAG 知识库系统...")
    try:
        knowledge_base_dir = os.path.join(BASE_DIR, "knowledge_base")
        model_path = os.path.join(knowledge_base_dir, "models", "bge-small-zh-v1.5")
        persist_directory = os.path.join(knowledge_base_dir, "db")

        if not os.path.exists(persist_directory):
            error_msg = f"知识库持久化目录不存在: {persist_directory}。请先运行 'python knowledge_base/build_knowledge.py' 来构建知识库。"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        #  初始化组件 
        logging.info("正在加载 RAG 的 Embedding 模型...")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embedding_function = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        logging.info("正在加载向量数据库...")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})

        #  构建RAG链 
        template = (
            "请只根据以下提供的上下文信息来回答问题。\n"
            "如果根据上下文信息无法回答问题，请直接说\"根据提供的知识库，我无法回答该问题\"，不要自行编造答案。\n\n"
            "上下文:\n{context}\n\n"
            "问题:\n{question}"
        )
        prompt = ChatPromptTemplate.from_template(template)

        # RAG链将问题传递给检索器获取上下文，然后与问题一起传递给提示模板，再由LLM处理，最后输出字符串
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logging.info("RAG 知识库系统初始化成功。")
        return True

    except Exception as e:
        logging.error(f"初始化 RAG 系统时发生严重错误: {e}", exc_info=True)
        rag_chain = None
        return False



#  LangChain Agent 的 MCP 工具封装 
# 这里是对mcp格式的langchain翻译，翻译成一个类
# 目前的逻辑下，并没有使用该方法
def create_pydantic(schema: dict, model_name: str) -> Type[BaseModel]:
    """
    根据 MCP 工具提供的 JSON Schema 动态创建 Pydantic 模型。
    这是让 LangChain Agent 理解工具参数的关键。
    """
    fields = {}
    properties = schema.get('properties', {})
    required_fields = schema.get('required', [])
    ## 转换为键值对应的列表
    for prop_name, prop_schema in properties.items():
        # 这里对类型做了简化映射，可以根据未来工具的复杂性进行扩展
        field_type: Type[Any] = str  # 默认为字符串类型
        if prop_schema.get('type') == 'integer':
            field_type = int
        elif prop_schema.get('type') == 'number':
            field_type = float
        elif prop_schema.get('type') == 'boolean':
            field_type = bool
        
        # Pydantic 的 create_model 需要一个元组: (类型, 默认值)
        # 对于必需字段，默认值是 ... (Ellipsis)
        if prop_name in required_fields:
            fields[prop_name] = (field_type, ...)
        else:
            fields[prop_name] = (field_type, None)
    
    # 使用 Pydantic 的 create_model 动态创建模型类
    return create_model(model_name, **fields)

class McpTool(BaseTool):
    """
    一个自定义的 LangChain 工具 (BaseTool)，用于封装 MCP 会话的工具调用功能。
    它充当了 LangChain Agent 和我们现有 MCP 服务之间的桥梁。
    """
    name: str
    description: str
    args_schema: Type[BaseModel]  # 强制工具必须有参数结构
    session: "ClientSession"      # 类型前向引用（类型前向引用指的是在类型注解中使用尚未在当前作用域定义的类型名）
    username: str

 # mcp定于的异步执行方法 _arun表示这个方法可以接收任意数量的关键字参数，并将它们收集到一个名为 kwargs 的字典中
    async def _arun(self, **kwargs: Any) -> Any:
        """
        通过 MCP 会话异步执行工具。
        Agent Executor 会将从 LLM 获取的参数作为关键字参数传递到这里。
        """
        # 将 MCP server 工具所需的 'username' 参数补充进去
        kwargs['username'] = self.username
        logging.info(f"LangChain Agent 正在调用工具 '{self.name}'，参数: {kwargs}")
        
        # 通过已建立的 mcp_session 调用真实的工具
        # 相当于调用mcp
        response_obj = await self.session.call_tool(self.name, kwargs)
        
        # 提取文本内容并返回给 Agent
        function_response_text = response_obj.content[0].text
        logging.debug(f"工具 '{self.name}' 返回了原始数据 (前200字符): {function_response_text[:200]}...")
        
        return function_response_text

class KnowledgeBaseToolInput(BaseModel):
    """知识库查询工具的输入模型。"""
    # Field(description=...) 的作用是给这个字段附加一个描述。
    query: str = Field(description="需要从知识库中查询的具体问题或关键词。")

class KnowledgeBaseTool(BaseTool):
    """
    一个用于查询本地知识库以获取因果推断相关知识的工具。
    """
    name: str = "knowledge_base_query"
    description: str = (
        "用于回答关于因果推断、统计学和机器学习的通用知识性问题。"
        "当你需要查找一个概念的定义、解释一个术语或获取背景知识时，必须使用此工具。"
        "例如，当分析结果中出现 '混杂变量' 时，你可以用它来查询 '混杂变量是什么'。"
    )
    args_schema: Type[BaseModel] = KnowledgeBaseToolInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("KnowledgeBaseTool 不支持同步执行。")

    async def _arun(self, query: str) -> Any:
        """通过 RAG 链异步执行知识库查询。"""
        global rag_chain
        if not rag_chain:
            return "知识库系统当前不可用。"
            
        logging.info(f"知识库工具正在查询: '{query}'")
        try:
            response = await rag_chain.ainvoke(query)
            logging.debug(f"知识库工具返回了原始数据 (前200字符): {response[:200]}...")
            return response
        except Exception as e:
            logging.error(f"知识库查询时发生错误: {e}", exc_info=True)
            return f"查询知识库时出错: {e}"


from causal_agent.graph import create_graph
from causal_agent.state import CausalChatState
from langgraph.types import Command  # 用于恢复 interrupt() 暂停的图



NODE_DESCRIPTIONS = {
    "agent": "路由决策 - 分析用户意图",
    "fold": "加载文件 - 读取并验证数据",
    "preprocess": "数据预处理 - 生成统计摘要和可视化",
    "execute_tools": "执行分析 - 运行因果分析和知识库查询",
    "postprocess": "后处理 - 环路检测和边评估",
    "report": "生成报告 - 整合分析结果",
    "normal_chat": "普通对话",
    "inquiry_answer": "基于报告回答问题"
}

async def ai_call(text, user_id, username, session_id):
    """
    使用我们模块化的 LangGraph agent 来处理用户请求。
    支持中断和恢复机制（基于checkpointer，无需全局状态）。
    """
    logging.info(f"处理用户 {username} 的消息，会话ID: {session_id}")
    
    # 配置：使用 session_id 作为 thread_id，让 Checkpointer 自动管理状态
    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id
        }
    }
    
    # 检查当前状态，判断是否是恢复中断的会话
    # LangGraph会通过checkpointer自动检测interrupt状态
    try:
        state = await agent_graph.aget_state(config)
        # 如果next为空且tasks不为空，说明有pending interrupt
        is_interrupted = state.next == () and state.tasks
        
        if is_interrupted:
            logging.info(f"检测到会话 {session_id} 处于中断状态，使用Command(resume=...)恢复")
            # 使用 Command(resume=...) 恢复执行，传入用户输入
            input_data = Command(resume=text)
        else:
            logging.info(f"正常对话或第一次对话")
            # 构建输入（只需要新消息，历史由 Checkpointer 管理）
            input_data = {
                "messages": [HumanMessage(content=text)],
                "user_id": user_id,
                "username": username,
                "session_id": session_id
            }
    except Exception as e:
        logging.warning(f"无法获取状态，假设为新对话: {e}")
        input_data = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id,
            "username": username,
            "session_id": session_id
        }
    
    #  执行图 
    try:
        final_state_data = None
        interrupt_info = None
        
        # 使用 astream 流式执行，可以捕获 interrupt 事件
        async for event in agent_graph.astream(input_data, config, stream_mode="values"):
            logging.info(f"收到事件: {list(event.keys())}")
            
            # 检查是否遇到 interrupt
            if "__interrupt__" in event:
                interrupt_info = event["__interrupt__"]
                logging.info(f"检测到 interrupt: {interrupt_info}")
                final_state_data = event  # 保存包含interrupt的状态
                break
            
            # 收集正常的状态更新
            if isinstance(event, dict) and event:
                final_state_data = event
        
        if interrupt_info:
            # 提取 interrupt 传递的问题
            # interrupt_info 是一个列表，包含Interrupt对象
            interrupt_obj = interrupt_info[0] if isinstance(interrupt_info, (list, tuple)) else interrupt_info
            
            # Interrupt 对象有 value 属性，包含实际的问题文本
            question = interrupt_obj.value if hasattr(interrupt_obj, 'value') else str(interrupt_obj)
            
            # 注意：不需要全局状态，checkpointer会自动管理
            logging.info(f"图已暂停，等待用户输入。问题: {question}")
            return {
                "type": "human_input_required",
                "summary": question
            }
        
        if final_state_data:
            return process_final_result(final_state_data)
        else:
            # 如果没有收集到状态，直接获取最终状态
            final_state_data = await agent_graph.aget_state(config)
            return process_final_result(final_state_data.values)
            
    except Exception as e:
        logging.error(f"执行 LangGraph Agent 时发生错误: {e}", exc_info=True)
        return {"type": "text", "summary": f"处理请求时出现错误: {e}"}

async def ai_call_stream(text, user_id, username, session_id):
    """
    流式版本的 ai_call，使用 astream() 捕获节点执行更新。
    这是一个生成器函数，会yield SSE格式的事件数据。
    """
    logging.info(f"[流式] 处理用户 {username} 的消息，会话ID: {session_id}")
    
    # 配置：使用 session_id 作为 thread_id
    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id
        }
    }
    
    # 检查当前状态，判断是否是恢复中断的会话
    try:
        state = await agent_graph.aget_state(config)
        ## 检查是否中断
        is_interrupted = state.next == () and state.tasks
        
        if is_interrupted:
            logging.info(f"[流式] 检测到会话 {session_id} 处于中断状态，使用Command(resume=...)恢复")
            input_data = Command(resume=text)
        else:
            logging.info(f"[流式] 正常对话或第一次对话")
            input_data = {
                "messages": [HumanMessage(content=text)],
                "user_id": user_id,
                "username": username,
                "session_id": session_id
            }
    except Exception as e:
        logging.warning(f"[流式] 无法获取状态，假设为新对话: {e}")
        input_data = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id,
            "username": username,
            "session_id": session_id
        }
    
    import time
    node_start_times = {}
    final_state_data = None
    interrupt_info = None
    last_node = None
    
    try:
        # 使用 astream 流式执行，捕获节点更新
        # stream_mode="updates" 会在每个节点执行后返回更新
        async for chunk in agent_graph.astream(input_data, config, stream_mode="updates"):
            logging.info(f"[SSE] 收到更新: {list(chunk.keys())}")
            
            # chunk的格式: {node_name: node_output}
            for node_name, node_output in chunk.items():
                if node_name in NODE_DESCRIPTIONS:

                    # 如果有上一个节点，且当前节点与上一个不同，先发送上一个节点的结束事件
                    if last_node and last_node != node_name and last_node in node_start_times:
                        start_time = node_start_times[last_node]
                        duration = round(time.time() - start_time, 2)
                        
                        event_data = {
                            "type": "node_end",
                            "node_name": last_node,
                            "duration": duration,
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        logging.info(f"[SSE] 节点完成: {last_node} (耗时: {duration}s)")
                    
                    # 发送当前节点的开始事件
                    if last_node != node_name:
                        node_start_times[node_name] = time.time()
                        
                        event_data = {
                            "type": "node_start",
                            "node_name": node_name,
                            "node_desc": NODE_DESCRIPTIONS[node_name],
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        logging.info(f"[SSE] 节点开始: {node_name}")
                        
                        last_node = node_name
                
                # 保存节点输出
                if isinstance(node_output, dict):
                    final_state_data = node_output
        
        # === 流结束后，发送最后一个节点的结束事件 ===
        if last_node and last_node in node_start_times:
            start_time = node_start_times[last_node]
            duration = round(time.time() - start_time, 2)
            
            event_data = {
                "type": "node_end",
                "node_name": last_node,
                "duration": duration,
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            logging.info(f"[SSE] 节点完成: {last_node} (耗时: {duration}s)")
        
        # 获取最终状态以检查interrupt
        state = await agent_graph.aget_state(config)
        final_state_data = state.values
        
        # 检查是否有interrupt
        if "__interrupt__" in final_state_data:
            interrupt_info = final_state_data["__interrupt__"]
            
            # 提取问题文本
            interrupt_obj = interrupt_info[0] if isinstance(interrupt_info, (list, tuple)) else interrupt_info
            question = interrupt_obj.value if hasattr(interrupt_obj, 'value') else str(interrupt_obj)
            
            event_data = {
                "type": "interrupt",
                "message": question
            }
            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            logging.info(f"[SSE] 图已暂停，等待用户输入")
        else:
            # 发送最终结果
            result = process_final_result(final_state_data)
            event_data = {
                "type": "final_result",
                "data": result
            }
            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            logging.info(f"[SSE] 发送最终结果")
            
    except Exception as e:
        logging.error(f"[流式] 执行 LangGraph Agent 时发生错误: {e}", exc_info=True)
        error_data = {
            "type": "error",
            "message": f"处理请求时出现错误: {str(e)}"
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

def process_final_result(final_state_data):
    """
    处理图正常完成后的最终结果
    
    优先级策略：
    1. 优先返回最新的 AI 消息（messages[-1]）
    2. 如果最新消息是决策消息，则检查是否有 final_report
    3. 如果有因果图数据，返回结构化响应
    """
    
    # === 优先级 1：检查最后一条消息 ===
    messages = final_state_data.get("messages", [])
    if messages:
        last_message = messages[-1]
        
        # 如果最后一条是 AI 消息
        if isinstance(last_message, AIMessage):
            message_name = getattr(last_message, 'name', None)
            
            # 检查是否是有实际内容的回复节点
            # （不是决策消息，而是真正的回复）
            if message_name in ['normal_chat', 'inquiry_answer']:
                logging.info(f"返回 {message_name} 节点的回复")
                return {
                    "type": "text",
                    "summary": last_message.content
                }
            
            # 如果是 report 节点生成的决策消息
            # 检查是否同时有 final_report
            if message_name == 'report' and final_state_data.get("final_report"):
                logging.info("返回完整的因果分析报告")
                
                # 检查是否有因果图数据（结构化返回）
                if final_state_data.get("causal_analysis_result"):
                    analysis_data = final_state_data["causal_analysis_result"]
                    if analysis_data.get("success"):
                        return {
                            "type": "causal_graph",
                            "summary": final_state_data["final_report"],
                            "data": analysis_data.get("data")
                        }
                
                # 只有报告，没有图数据
                return {
                    "type": "text",
                    "summary": final_state_data["final_report"]
                }
    
    # === 优先级 2：降级方案，返回 final_report（如果有）===
    final_report = final_state_data.get("final_report")
    if final_report:
        logging.info("未找到最新消息，降级返回 final_report")
        return {"type": "text", "summary": final_report}
    
    # === 优先级 3：最后的降级方案 ===
    logging.warning("未找到任何可返回的内容，返回默认消息")
    return {"type": "text", "summary": "抱歉，我在处理时遇到了问题。"}


## 保存历史文件 ( 添加 user_id 参数和列)
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
            # 
            
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
                # 确保 summary 键存在且不为 None
                ai_content = ai_response.get('summary')
                if ai_content is None:
                    # 如果 summary 为空，将整个响应序列化为字符串作为备用
                    ai_content = json.dumps(ai_response, ensure_ascii=False)
                
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

            # 修改：根据是否为第一条消息，决定是否更新标题
            if is_first_message:
                # 4a. 更新会话，包括新标题（或确认创建时的标题）
                new_title = user_msg[:8] # 截取前8个字符作为标题
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


# 会话管理接口,获取会话
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

    logging.info(f"用户 {username} (ID: {user_id}) 请求加载会话: {session_id} (延迟创建模式)")

    messages = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            #  核心修改：处理延迟创建的session 
            # 首先检查session是否存在
            cursor.execute("SELECT id FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            session_exists = cursor.fetchone()
            
            if not session_exists:
                # Session还不存在（用户还没发送第一条消息），返回空消息列表
                logging.info(f"会话 {session_id} 尚未创建（延迟创建模式），返回空消息列表")
                return jsonify({"success": True, "messages": []})
            # --

            # 获取所有消息，并左连接附件表
            # 这里的 cm 是 chat_messages 表的别名,ca 是 chat_attachments 表的别名
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
        
        logging.info(f"用户 {username} 成功加载会话 {session_id} ({len(messages)} 条消息)")
        return jsonify({"success": True, "messages": messages})

    except mysql.connector.Error as e:
        logging.error(f"加载会话 {session_id} (用户 {username}) 时数据库出错: {e}")
        return jsonify({"success": False, "error": f"加载会话时出错: {e}"}), 500
    except Exception as e:
        logging.error(f"加载会话 {session_id} (用户 {username}) 时发生未知错误: {e}")
        return jsonify({"success": False, "error": f"加载会话时出错: {e}"}), 500
 
@app.route('/api/change_session', methods=['POST'])
def change_session():
    #  用户认证检查 
    from flask import session
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
            #  修改：增加 user_id 条件以确保安全，并处理延迟创建的session 
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET title = %s WHERE id = %s AND user_id = %s",
                (title, session_id, user_id)
            )
            conn.commit()
            
            #  修改：更详细的错误处理，区分延迟创建的session 
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

@app.route('/api/delete_session', methods=['POST'])
def delete_session():
    #  核心修改：安全和完整的删除逻辑，支持延迟创建 
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
            "userAgreement": "Userprivacy.md",
            "userManual": "manual.md"
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
    #  重构：从 Session 获取用户身份 
    from flask import session
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

# 根路由 (不变)
@app.route('/')
def index():
    # 总是返回 chat.html，由前端 JS 决定显示登录还是主界面
    return send_from_directory('static', 'chat.html')

# 主程序入口
if __name__ == '__main__':
    # 注册应用退出时的清理函数
    atexit.register(shutdown_mcp_connection)

    # 1. 初始化 LLM
    if not initialize_llm():
        logging.critical("LLM 初始化失败，应用无法启动。")
        sys.exit(1)

    # 2. 初始化 RAG 系统
    if not initialize_rag_system():
        # RAG 不是致命错误，允许在没有知识库的情况下继续运行
        logging.warning("RAG 系统初始化失败。应用将以无知识库模式运行。")
    
    # 3. 启动后台事件循环并等待 MCP 就绪
    mcp_ready_event = threading.Event()
    
    ## 后台线程
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

    logging.info("正在根据 LLM 和 MCP 会话创建 Agent Graph...")
    
    agent_graph = create_graph(llm, mcp_session)
    logging.info("Agent Graph 创建成功。")
    # -- 
    logging.info("MCP 连接就绪，启动 Flask 应用服务器...")
    # 启动 Flask 应用
    # use_reloader=False 是必须的，因为重载器会启动一个子进程，
    # 这会干扰我们已经手动管理的 MCP 子进程和事件循环。
    # 
    # Docker环境注意：
    # - host='0.0.0.0' 监听所有网络接口，允许容器外部访问
    # - host='127.0.0.1' 只允许容器内部访问，Docker端口映射会失效
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

    