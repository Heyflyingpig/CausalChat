"""
@task封装的rag任务
"""
from langgraph.func import task
import logging
from Agent.knowledge_base.query_rag import get_rag_response
from typing import List, Dict

@task
def rag_query_task(questions: List[str]) -> Dict:
    """
    Task: 查询知识库（RAG）
    a
    Args:
        questions: 要查询的问题列表
    
    Returns:
        dic: 知识库查询结果
    """
    logging.info("正在启动RAG查询任务...")
    try:
        rag_response = get_rag_response(questions)
        logging.info("Task: 知识库查询完成")
        return {"success": True, "response": rag_response}
        
    except Exception as e:
        logging.error(f"Task: 知识库查询失败: {e}")
        return {"success": False, "response": str(e)}
