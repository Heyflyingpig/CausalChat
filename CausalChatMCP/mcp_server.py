import os
import logging
import sys

# --- Path fix: Add project root to sys.path ---
# This ensures that the 'causal' module can be found when this script is run as a subprocess.
MCP_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MCP_SERVER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of path fix ---

import json
import mysql.connector
from mysql.connector import errorcode
from mcp.server.fastmcp import FastMCP
from causal.causalachieve import run_pc_analysis

# --- 配置日志记录到文件 ---
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mcp_server.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
    ]
)
logging.info("--- MCP Server Script Started, Logging Initialized ---")

# --- 数据库配置和加载 ---
# 修改：构建到项目根目录的绝对路径来定位 secrets.json，增强健壮性
MCP_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MCP_SERVER_DIR)
SECRETS_PATH = os.path.join(PROJECT_ROOT, "secrets.json")
MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE = None, None, None, None

def load_db_config():
    """从 secrets.json 加载数据库配置。"""
    global MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    try:
        if not os.path.exists(SECRETS_PATH):
            raise FileNotFoundError(f"MCP Server: secrets.json not found at expected path '{SECRETS_PATH}'.")
        with open(SECRETS_PATH, "r", encoding="utf-8") as f:
            secrets = json.load(f)
        MYSQL_HOST = secrets["MYSQL_HOST"]
        MYSQL_USER = secrets["MYSQL_USER"]
        MYSQL_PASSWORD = secrets["MYSQL_PASSWORD"]
        MYSQL_DATABASE = secrets["MYSQL_DATABASE"]
        logging.info("MCP Server: 数据库配置已成功加载。")
    except Exception as e:
        logging.error(f"MCP Server: 加载数据库配置失败: {e}", exc_info=True)
        raise

load_db_config()

# --- 数据库辅助函数 ---
def get_db_connection():
    """创建并返回一个MySQL数据库连接。"""
    try:
        return mysql.connector.connect(
            host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DATABASE
        )
    except mysql.connector.Error as err:
        logging.error(f"MCP Server: MySQL 连接错误: {err}")
        raise

def find_user(username: str) -> dict | None:
    """按用户名查找用户并返回用户信息字典。"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, username FROM users WHERE username = %s", (username,))
            return cursor.fetchone()
    except mysql.connector.Error as e:
        logging.error(f"MCP Server: 查找用户 '{username}' 时出错: {e}")
        return None

def get_file_content_from_db(user_id: int, filename: str) -> bytes | None:
    """从数据库为指定用户获取文件内容。"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT file_content FROM uploaded_files WHERE user_id = %s AND filename = %s ORDER BY upload_timestamp DESC LIMIT 1",
                (user_id, filename)
            )
            result = cursor.fetchone()
            return result['file_content'] if result else None
    except mysql.connector.Error as e:
        logging.error(f"MCP Server: 从数据库获取文件 '{filename}' (用户ID: {user_id}) 时出错: {e}")
        return None

# --- MCP 服务器和工具定义 ---
mcp = FastMCP("causal-analyzer")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@mcp.tool()
async def read_file(filename: str) -> str:
    """
    读取指定文件的内容。

    Args:
        filename: 要读取的文件的名称 (例如: data_to_read.txt).
    
    Returns:
        文件的内容，如果文件不存在或无法访问则返回错误信息。
    """
    try:
        # 构建安全的文件路径 (相对于 mcp_server.py 所在的 CausalChatMCP 目录)
        # 注意：此工具的上下文与主应用的文件系统不同。
        secure_path = os.path.join(BASE_DIR, filename)
        
        # 验证路径是否在预期目录下
        if os.path.commonpath([BASE_DIR]) != os.path.commonpath([BASE_DIR, secure_path]):
            return f"错误：禁止访问路径 '{filename}'。"

        if not os.path.exists(secure_path):
            return f"错误：文件 '{filename}' 未找到。"

        with open(secure_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件时发生错误: {e}"

@mcp.tool()
async def perform_causal_analysis(filename: str, username: str) -> str:
    """
    对用户上传的CSV文件执行因果分析。

    Args:
        filename: 要分析的CSV文件的名称 (例如, 'my_data.csv')。
        username: 请求分析的用户名 (由系统自动添加)。
    
    Returns:
        一个包含分析结果的JSON字符串。
    """
    logging.info(f"工具 'perform_causal_analysis' 已为用户 '{username}' 的文件 '{filename}' 调用。")
    try:
        user_data = find_user(username)
        if not user_data:
            return json.dumps({"success": False, "message": f"错误：用户 '{username}' 不存在。"}, ensure_ascii=False)
        user_id = user_data['id']

        file_content_bytes = get_file_content_from_db(user_id, filename)
        if file_content_bytes is None:
            return json.dumps({"success": False, "message": f"错误：未找到文件 '{filename}'。请确认文件已上传。"}, ensure_ascii=False)
        
        try:
            csv_data_string = file_content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return json.dumps({"success": False, "message": f"错误：文件 '{filename}' 不是有效的 UTF-8 编码。"}, ensure_ascii=False)
        
        analysis_result = run_pc_analysis(csv_data_string)
        return json.dumps(analysis_result, ensure_ascii=False)

    except Exception as e:
        logging.error(f"'perform_causal_analysis' 工具执行出错: {e}", exc_info=True)
        return json.dumps({"success": False, "message": f"执行工具时发生内部错误: {e}"}, ensure_ascii=False)

if __name__ == "__main__":
    logging.info("--- MCP 因果分析服务器启动 ---")

    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logging.error(f"MCP 服务器运行时出现致命错误: {e}", exc_info=True)
    finally:
        logging.info("--- MCP 因果分析服务器关闭 ---") 