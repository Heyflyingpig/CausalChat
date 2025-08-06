from .state import CausalChatState
from langchain_core.messages import AIMessage
import logging

def decision_router(state: CausalChatState) -> str:
    """
    这是图中的主要“决策”边。
    它检查来自代理的最新消息以决定下一步行动。
    """
    logging.info("--- 路由: 主决策 ---")
    agent_decision = state["messages"][-1].content
    
    if "信息完备" in agent_decision:
        logging.info("路由决策 -> 前往[后处理]")
        return "postprocess"
    elif "信息不全" in agent_decision:
        logging.info("路由决策 -> 前往[预处理]")
        return "preprocess"
    else: # "普通问答" or any other default
        logging.info("路由决策 -> 前往[普通问答]")
        return "normal_chat"

def preprocess_router(state: CausalChatState) -> str:
    """
    This router checks if the parameters collected by the preprocess_node are sufficient
    to proceed with tool execution.
    """
    logging.info("--- 路由: 预处理后决策 ---")
    preprocess_decision = state["messages"][-1].content
    if "信息完备" in preprocess_decision:
        logging.info("路由决策 -> 参数充足, 前往[执行工具]")
        return "execute_tools"
    else:
        logging.info("路由决策 -> 参数不足, 前往[询问用户]")
        return "ask_human"

def execute_tool_router(state:CausalChatState) -> str:
    '''
    通向decision_router
    '''
    logging.info("--- 路由: 执行工具后决策 ---")
    logging.info(f"前往decision_router")
    return "agent"

def postprocess_router(state:CausalChatState) -> str:
    '''
    通向report_node
    '''
    logging.info("--- 路由: 后处理后决策 ---")
    logging.info(f"前往report_node")
    return "report"


