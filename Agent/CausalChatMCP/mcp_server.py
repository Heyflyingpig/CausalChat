import os
import logging
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Agent/CausalChatMCP
AGENT_DIR = os.path.dirname(CURRENT_DIR)                      # .../Agent
PROJECT_ROOT = os.path.dirname(AGENT_DIR)                     # 项目根目录

for p in (PROJECT_ROOT, AGENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


import json
from mcp.server.fastmcp import FastMCP
from Agent.causal.causalachieve import run_pc_analysis


log_file_path = os.path.join(CURRENT_DIR, 'mcp_server.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
    ]
)
logging.info("--- MCP Server Script Started, Logging Initialized ---")


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