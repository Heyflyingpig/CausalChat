import os
import logging
from mcp.server.fastmcp import FastMCP

# --- 新增: 配置日志记录到文件 ---
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mcp_server.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        # 如果你还想在控制台看到 mcp_server.py 的直接输出（如果有的话），可以保留 StreamHandler
        # logging.StreamHandler() 
    ]
)
logging.info("--- MCP Server Script Started, Logging Initialized ---")

# 初始化 FastMCP 服务器，命名为 "file-reader"
mcp = FastMCP("file-reader")

# 获取当前脚本所在的目录
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
        # 构建安全的文件路径
        secure_path = os.path.join(BASE_DIR, filename)
        
        # 再次检查，确保最终路径仍在预期目录下，防止路径遍历攻击
        if os.path.commonpath([BASE_DIR]) != os.path.commonpath([BASE_DIR, secure_path]):
            return f"错误：禁止访问路径 '{filename}'。"

        if not os.path.exists(secure_path):
            return f"错误：文件 '{filename}' 未找到。"

        with open(secure_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content
    except Exception as e:
        return f"读取文件时发生错误: {e}"

if __name__ == "__main__":
    logging.info("--- MCP 文件读取服务器启动 ---")
    logging.info(f"提供的工具: read_file(filename)")
    # 启动服务器，使用 stdio 作为传输方式
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logging.error(f"MCP 服务器运行时出现致命错误: {e}", exc_info=True)
    finally:
        logging.info("--- MCP 文件读取服务器关闭 ---") 