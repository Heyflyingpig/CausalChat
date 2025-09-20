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
        logging.info("路由决策 -> 前往[文件加载]")
        return "fold_router"
    else: # "普通问答" or any other default
        logging.info("路由决策 -> 前往[普通问答]")
        return "normal_chat"

def fold_router(state: CausalChatState) -> str:
    """
    这是图中的文件加载“决策”边。
    它检查来自代理的最新消息以决定下一步行动。
    """
    logging.info("--- 路由: 文件加载决策 ---")
    fold_process_decision = state["messages"][-1].content
    if "信息完备" in fold_process_decision:
        logging.info("路由决策 -> 前往[执行预处理]")
        return "preprocess"
    else:
        logging.info("路由决策 -> 前往[询问用户]")
        return "ask_human"

def preprocess_router(state: CausalChatState) -> str:
    """
    参数验证节点后的路由器。
    如果验证成功，则执行工具
    """
    logging.info("--- 路由: 预处理后决策 ---")
    preprocess_decision = state["messages"][-1].content
    logging.info("路由决策 -> 参数充足, 前往[执行工具]")
    return "execute_tools"


def execute_tool_router(state:CausalChatState) -> str:
    """
    工具执行节点后的路由器。
    返回agent节点，等待agent节点做出决策
    """
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

def ask_human_router(state:CausalChatState) -> str:
    '''
    人工干预之后，通向agent_node
    '''
    logging.info("--- 路由: 询问用户后决策 ---")
    logging.info(f"前往agent_node")
    return "agent"

