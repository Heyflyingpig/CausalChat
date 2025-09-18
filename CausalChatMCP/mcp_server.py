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


# --- MCP 服务器和工具定义 ---
mcp = FastMCP("causal-analyzer")

@mcp.tool()
async def perform_causal_analysis(csv_data: str) -> str:
    """
    接收一个包含CSV数据的字符串，并对其执行因果分析。
    这是一个纯计算工具，不执行任何数据库或文件系统操作。

    Args:
        csv_data: 一个包含完整CSV文件内容的字符串。
    
    Returns:
        一个包含分析结果的JSON字符串。
    """
    logging.info(f"工具 'perform_causal_analysis' 已被调用，输入数据长度: {len(csv_data)}。")
    try:
        # 工具的核心职责：执行分析
        analysis_result = run_pc_analysis(csv_data)
        
        return json.dumps(analysis_result, ensure_ascii=False)

    except Exception as e:
        logging.error(f"'perform_causal_analysis' 工具执行出错: {e}", exc_info=True)
        return json.dumps({"success": False, "message": f"执行分析时发生内部错误: {e}"}, ensure_ascii=False)

if __name__ == "__main__":
    logging.info("--- MCP 因果分析服务器启动 ---")

    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logging.error(f"MCP 服务器运行时出现致命错误: {e}", exc_info=True)
    finally:
        logging.info("--- MCP 因果分析服务器关闭 ---") 