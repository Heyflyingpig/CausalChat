"""
@task封装因果分析任务
"""
from langgraph.func import task
import logging
from mcp import ClientSession
from typing import List, Dict
import json

@task
async def causal_analysis_task(file_content: str, mcp_session: ClientSession) -> Dict:
    """
    Task: 执行因果分析（通过 MCP）
    
    Args:
        file_content: CSV 文件内容字符串
        mcp_session: MCP 客户端会话
    
    Returns:
        dic: 因果分析字典
    """
    logging.info("正在启动因果分析任务...")
    try:
            tool_response = await mcp_session.call_tool(
                "perform_causal_analysis",
                {"csv_data": file_content}
            )
            
            result = json.loads(tool_response.content[0].text)
            logging.info("Task: MCP 因果分析完成")
            return result
            
    except Exception as e:
        logging.error(f"Task: MCP 调用失败: {e}")
        return {"success": False, "message": str(e)}