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
    """从数据库为指定用户获取文件内容，并更新访问记录。"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            # 优先按原始文件名查找，并按最新上传/访问排序
            cursor.execute(
                "SELECT id, file_content FROM uploaded_files WHERE user_id = %s AND original_filename = %s ORDER BY last_accessed_at DESC LIMIT 1",
                (user_id, filename)
            )
            result = cursor.fetchone()
            
            if result:
                file_id = result['id']
                file_content = result['file_content']
                
                # 更新访问时间和计数
                cursor.execute(
                    "UPDATE uploaded_files SET last_accessed_at = NOW(), access_count = access_count + 1 WHERE id = %s",
                    (file_id,)
                )
                conn.commit()
                logging.info(f"MCP Server: 成功获取文件 '{filename}' (ID: {file_id}) 并更新访问记录。")
                return file_content
            else:
                logging.warning(f"MCP Server: 未找到文件 '{filename}' (用户ID: {user_id})。")
                return None

    except mysql.connector.Error as e:
        logging.error(f"MCP Server: 从数据库获取文件 '{filename}' (用户ID: {user_id}) 时出错: {e}")
        return None

def get_recentfile(user_id: int) -> dict | None:
    """获取用户最近上传或访问的文件的记录。"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            # 按 last_accessed_at 降序排序，找到最近操作过的文件
            cursor.execute(
                """
                SELECT id, file_content, original_filename 
                FROM uploaded_files 
                WHERE user_id = %s 
                ORDER BY last_accessed_at DESC 
                LIMIT 1
                """,
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                logging.info(f"MCP Server: 找到用户 {user_id} 的最近文件: '{result['original_filename']}' (ID: {result['id']})")
                return result
            else:
                logging.warning(f"MCP Server: 未找到用户 {user_id} 的任何文件。")
                return None
    except mysql.connector.Error as e:
        logging.error(f"MCP Server: 为用户 {user_id} 获取最近文件时出错: {e}")
        return None

# --- MCP 服务器和工具定义 ---
mcp = FastMCP("causal-analyzer")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@mcp.tool()
async def perform_causal_analysis(username: str, filename: str = None) -> str:
    """
    对用户上传的CSV文件执行因果分析。当用户请求进行分析、重新分析、或再次执行分析时，都应调用此工具。如果用户没有指定文件名，会自动分析最近上传的文件。

    Args:
        username: 请求分析的用户名 (由系统自动添加)。
        filename (可选): 要分析的CSV文件的名称 (例如, 'my_data.csv')。如果未提供，将自动分析最近上传的文件。
    
    Returns:
        一个包含分析结果的JSON字符串，成功时会包含被分析的文件名。
    """
    logging.info(f"工具 'perform_causal_analysis' 已为用户 '{username}' 的文件 '{filename or '最近文件'}' 调用。")
    try:
        user_data = find_user(username)
        if not user_data:
            return json.dumps({"success": False, "message": f"错误：用户 '{username}' 不存在。"}, ensure_ascii=False)
        user_id = user_data['id']

        file_content_bytes = None
        target_filename = None

        # 1. 如果提供了文件名，则优先尝试获取它
        if filename:
            logging.info(f"正在为用户 {user_id} 获取指定文件: '{filename}'")
            file_content_bytes = get_file_content_from_db(user_id, filename)
            if file_content_bytes:
                target_filename = filename
            else:
                logging.warning(f"指定文件 '{filename}' 未找到，将尝试查找最近的文件作为后备。")

        # 2. 如果没有提供文件名，或者指定的文件未找到，则查找最近的文件
        if not file_content_bytes:
            logging.info(f"未提供文件名或指定文件未找到，正在为用户 {user_id} 查找最近文件。")
            most_recent_file = get_recentfile(user_id)
            if most_recent_file:
                file_content_bytes = most_recent_file['file_content']
                target_filename = most_recent_file['original_filename']
                logging.info(f"已自动选择最近文件: '{target_filename}'")

        # 3. 如果两种方式都找不到文件，则返回错误
        if not file_content_bytes:
            error_message = f"错误：未找到文件 '{filename}'。" if filename else "错误：找不到任何上传过的文件。"
            error_message += " 请先上传一个文件或检查文件名。"
            return json.dumps({"success": False, "message": error_message}, ensure_ascii=False)
        
        # 4. 执行分析
        try:
            csv_data_string = file_content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return json.dumps({"success": False, "message": f"错误：文件 '{target_filename}' 不是有效的 UTF-8 编码。"}, ensure_ascii=False)
        
        analysis_result = run_pc_analysis(csv_data_string)
        
        # 在成功的结果中添加被分析的文件名，以便LLM可以告知用户
        if analysis_result.get("success"):
            analysis_result["analyzed_filename"] = target_filename

        return json.dumps(analysis_result, ensure_ascii=False)

    except Exception as e:
        logging.error(f"'perform_causal_analysis' 工具执行出错: {e}", exc_info=True)
        return json.dumps({"success": False, "message": f"执行工具时发生内部错误: {e}"}, ensure_ascii=False)

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

if __name__ == "__main__":
    logging.info("--- MCP 因果分析服务器启动 ---")

    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logging.error(f"MCP 服务器运行时出现致命错误: {e}", exc_info=True)
    finally:
        logging.info("--- MCP 因果分析服务器关闭 ---") 