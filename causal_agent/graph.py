from functools import partial
from langgraph.graph import StateGraph, END

from .state import CausalChatState
from . import nodes, edges

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
    execute_tools_node_with_session = partial(nodes.execute_tools_node, mcp_session=mcp_session,llm=llm)
    postprocess_node_with_llm = partial(nodes.postprocess_node, llm=llm)
    report_node_with_llm = partial(nodes.report_node, llm=llm)

    # Add all the nodes to the graph
    workflow.add_node("agent", agent_node_with_llm)
    workflow.add_node("fold", fold_node_with_llm)
    workflow.add_node("preprocess", preprocess_node_with_llm)
    workflow.add_node("execute_tools", execute_tools_node_with_session)
    workflow.add_node("postprocess", postprocess_node_with_llm)
    workflow.add_node("report", report_node_with_llm)
    workflow.add_node("normal_chat", nodes.normal_chat_node)
    workflow.add_node("ask_human", nodes.ask_human_node)

    # Set the entry point of the graph
    workflow.set_entry_point("agent")

    # Add conditional edges that determine the flow based on router functions
    workflow.add_conditional_edges(
        "agent",
        edges.decision_router,
        {
            "preprocess": "preprocess",
            "fold": "fold",
            "normal_chat": "normal_chat"
        }
    )
    workflow.add_conditional_edges(
        "fold",
        edges.fold_router,
        {
            "preprocess": "preprocess",
            "ask_human": "ask_human"
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
    workflow.add_conditional_edges(
        "ask_human",
        edges.ask_human_router,
        {
            "agent": "agent"
        }
    )
 
    # Define the end points of the graph. A graph can have multiple finishing points.
    workflow.add_edge("report", END)
    workflow.add_edge("normal_chat", END)
    

    # Compile the graph into a runnable application
    # 通过设置 `interrupt_before`, 我们告诉图在执行 `ask_human` 节点之前暂停。
    app = workflow.compile(interrupt_before=["ask_human"])
    
    return app


agent_graph = None
