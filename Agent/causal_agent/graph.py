from functools import partial
from langgraph.graph import StateGraph, END
from .state import CausalChatState
from . import nodes, edges
import logging

# === 导入 Checkpoint 相关 ===
from Database.mysql_checkpointer import MySQLSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

def create_graph(llm: "ChatOpenAI", mcp_session: "ClientSession"):
    """
    组件node和edge成为边
    """
    workflow = StateGraph(CausalChatState)

    # 使用 functools.partial 将 llm 实例绑定到节点函数上
    # 这使得节点在被 LangGraph 调用时，除了 state 之外，还能接收到 llm 对象
    agent_node_with_llm = partial(nodes.agent_node, llm=llm)
    fold_node_with_llm = partial(nodes.fold_node, llm=llm)
    preprocess_node_with_llm = partial(nodes.preprocess_node, llm=llm)
    execute_tools_node_with_session = partial(nodes.execute_tools_node, mcp_session=mcp_session, llm=llm)
    postprocess_node_with_llm = partial(nodes.postprocess_node, llm=llm)
    inquiry_answer_node_with_llm = partial(nodes.inquiry_answer_node, llm=llm)
    report_node_with_llm = partial(nodes.report_node, llm=llm)
    normal_chat_node_with_llm = partial(nodes.normal_chat_node, llm=llm)

    # Add all the nodes to the graph
    workflow.add_node("agent", agent_node_with_llm)
    workflow.add_node("fold", fold_node_with_llm)
    workflow.add_node("preprocess", preprocess_node_with_llm)
    workflow.add_node("execute_tools", execute_tools_node_with_session)
    workflow.add_node("postprocess", postprocess_node_with_llm)
    workflow.add_node("report", report_node_with_llm)
    workflow.add_node("normal_chat", normal_chat_node_with_llm)
    workflow.add_node("inquiry_answer", inquiry_answer_node_with_llm)

    
    # Set the entry point of the graph
    workflow.set_entry_point("agent")

    # Add conditional edges that determine the flow based on router functions
    workflow.add_conditional_edges(
        "agent",
        edges.decision_router,
        {
            "fold": "fold",
            "normal_chat": "normal_chat",
            "postprocess": "postprocess",
            "inquiry_answer": "inquiry_answer"
        }
    )
    workflow.add_conditional_edges(
        "fold",
        edges.fold_router,
        {
            "preprocess": "preprocess",
            "agent": "agent"  
        }
    )
    
    workflow.add_conditional_edges(
        "preprocess",
        edges.preprocess_router,
        {
            "execute_tools": "execute_tools",
        }
    )
    workflow.add_conditional_edges(
        "execute_tools",
        edges.execute_tool_router,
        {
            "agent": "agent"
        }
    )

    workflow.add_conditional_edges(
        "postprocess",
        edges.postprocess_router,
        {
            "report": "report"
        }
    )

 
    # Define the end points of the graph. A graph can have multiple finishing points.
    workflow.add_edge("report", END)
    workflow.add_edge("normal_chat", END)
    

    checkpointer = None
    try:
        # 从配置文件加载数据库连接信息
        from config.settings import settings
        
        connection_config = {
            'host': settings.MYSQL_HOST,
            'port': 3306,  # MySQL 默认端口
            'user': settings.MYSQL_USER,
            'password': settings.MYSQL_PASSWORD,
            'database': settings.MYSQL_DATABASE
        }
        
        # 创建 MySQLSaver 实例
        # serde 使用 JsonPlusSerializer(pickle_fallback=True) 处理 DataFrame 等复杂对象
        # 虽然现在没有
        checkpointer = MySQLSaver(
            connection_config=connection_config,
            serde=JsonPlusSerializer(pickle_fallback=True)
        )
        
        
        logging.info("MySQL Checkpointer 已启用")
        
    except Exception as e:
        logging.warning(f" Checkpointer 初始化失败，将不启用持久化: {e}")
        checkpointer = None


    app = workflow.compile(
        checkpointer=checkpointer  #
    )
    
    return app


agent_graph = None
