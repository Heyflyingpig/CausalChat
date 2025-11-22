from typing import TypedDict, List, Optional, Any, Annotated
from langchain_core.messages import BaseMessage
from operator import add

class CausalChatState(TypedDict):
    """
    Represents the state of our graph. This TypedDict acts as the "memory"
    or "state" that is passed between all the nodes in the graph.

    Attributes:
        messages: The history of messages in the conversation.
        user_id: The ID of the current user.
        username: The name of the current user.
        session_id: The ID of the current chat session.
        
        # 是否调用工具
        tool_call_request: Optional[dict]

        # 数据摘要
        analysis_parameters: Optional[dict]
        # 数据源文件 (字符串格式，用于工具调用)
        file_content: Optional[str] 
        # 解析后的数据框 (Pandas DataFrame，用于内部处理)
        dataframe: Optional[Any]

        # Fields for storing results from tools
        causal_analysis_result: Optional[dict]
        knowledge_base_result: Optional[str]
        
        # 后处理补充结果
        postprocess_result: Optional[dict]

        # Fields for final output and flow control
        final_report: Optional[str]
    """
    messages: Annotated[List[BaseMessage], add]
    
    username: str
    user_id: int
    session_id: str
    fold_name:str

    tool_call_request: Optional[bool]
    
    ## 从Pandas DataFrame中提取一个结构化的摘要，包含因果推断所需的详细统计信息。
    analysis_parameters: Optional[dict]
    file_content: Optional[str]
    ## dataframe: Optional[Any]
    
    causal_analysis_result: Optional[dict]
    knowledge_base_result: Optional[str]
    
    preprocess_summary: Optional[str]
    postprocess_result: Optional[dict]

    # 报告
    final_report: Optional[str]
    visualization_mapping: Optional[dict]
    

    
    # 可视化结果
    visualizations: Optional[dict]
