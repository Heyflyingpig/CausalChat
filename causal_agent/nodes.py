from .state import CausalChatState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional
import logging

# --- 1. 定义LLM决策的结构化输出模型 ---
class RouteQuery(BaseModel):
    """定义Agent决策的选项。"""
    route: Literal["preprocess", "postprocess", "normal_chat"] = Field(
        ...,
        description="根据用户的对话历史和意图，选择下一步应该走的路径。"
    )

# agentnode节点用于做初步的decision
def agent_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    Agent节点，是图的起点，用于判断是否需要进入causal循环，
    根据当前状态强制LLM做出三选一的决策，然后将该决策转化为消息。
    """
    logging.info("--- 步骤: Agent 节点 (LLM 决策) ---")

    # 检查生成报告所需的结果是否已存在。
    has_tool_results = state.get('causal_analysis_result') is not None
    
    # --- 3. 构建引导LLM决策的Prompt ---
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """你是一个专业的AI助手路由中枢。你的任务是根据用户的对话历史和当前状态，决定下一步的最佳路径。

# 当前状态摘要:
- 是否已获得分析工具的结果: {has_tool_results}

# 你的决策选项:
1. `postprocess`: 如果已经获得了因果分析结果 ({has_tool_results} is True)，可以选择此路径以进入后处理模块。
2. `preprocess`: 如果用户想要进行因果分析 (例如，对话中提到“分析”、“处理数据”后者是与“因果推断”相关的用语时候)，但我们还没有分析结果 ({has_tool_results} is False)，选择此路径以启动预处理模块。
3. `normal_chat`: 如果用户的提问只是一个与因果领域不相关的消息，不需要调用任何复杂的因果分析工具，选择此路径，进入normal_chat_node(正常问答)模块。

请根据下面的对话历史，做出你的选择。"""),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    # --- 4. 构建并调用LLM链 ---
    # .with_structured_output(RouteQuery) 是关键，它会强制LLM的输出符合RouteQuery的格式
    runnable = prompt | llm.with_structured_output(RouteQuery)
    
    logging.info("正在调用LLM进行路由决策...")
    structured_response = runnable.invoke({
        "messages": state["messages"],
        "has_tool_results": has_tool_results
    })
    logging.info(f"LLM决策结果: {structured_response.route}")

    # --- 5. 根据LLM的结构化决策，生成用于路由的消息 ---
    if structured_response.route == 'postprocess':
        response_message = AIMessage(content="决策：信息完备，进入后处理模块。", name="agent")
    elif structured_response.route == 'preprocess':
        response_message = AIMessage(content="决策：信息不全，启动数据预处理。", name="agent")
    else: # 'normal_chat'
        response_message = AIMessage(content="决策：普通问答。", name="agent")

    # 将Agent的决策追加到消息历史中
    state["messages"].append(response_message)
    return {"messages": state["messages"]}

# --- 2. 定义预处理节点 ---
class PreprocessQuery(BaseModel):
    """定义Agent决策的选项。"""
    route: Literal["execute_tools", "ask_human"] = Field(
        ...,
        description="根据用户的对话历史和意图，选择下一步应该走的路径。"
    )

def preprocess_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    项目预处理模块
    加载用户上传的文件
    处理用户输入的参数是否有缺漏，也就是需要分析的参数是否有误
    """
    logging.info("--- 步骤: 预处理节点 ---")
    ## 测试案例
    file_parameters= {"file_id": "123", "target": "Y", "treatment": "X","test":"TURE"} # Placeholder
    
    ## 传入数据之后，需要做决策，看是否需要补充参数
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         你是一个专业的AI助手路由中枢。你的任务是根据用户的对话历史和当前状态，决定下一步的最佳路径。
         根据目前用户上传的文件参数和历史记录中的用户问题描述，分析是否需要补充参数
         
         # 当前状态摘要
         1. 用户上传的文件参数：{file_parameters}

         # 你的决策选项
         1. `execute_tools`:如果{file_parameters}参数齐全，能回答目前的用户问题，则选择此路径
         2. `ask_human`:如果{file_parameters}参数不齐全，则选择此路径，则选择此路径
         
        请根据下面的对话历史，做出你的选择。"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    runnable = prompt | llm.with_structured_output(PreprocessQuery)
    
    logging.info("正在调用LLM进行路由决策...")
    structured_response = runnable.invoke({
        "messages": state["messages"],
        "file_parameters": file_parameters
    })
    logging.info(f"LLM决策结果: {structured_response.route}")

    if structured_response.route == 'execute_tools':
        response_message = AIMessage(content="决策：信息完备，进入执行工具模块。", name="preprocess")
        
    elif structured_response.route == 'ask_human':
        response_message = AIMessage(content="决策：信息不全，启动人机交互模块。", name="preprocess")

    state["messages"].append(response_message)
    # 无论是否完备都可以存在状态中
    state["analysis_parameters"] = file_parameters
    return {"messages": state["messages"], "analysis_parameters": state["analysis_parameters"]}


def execute_tools_node(state: CausalChatState) -> dict:
    """
    核心构建
    执行工具模块：
    主要是为了调用各种工具，根据当前状态和参数，选择单个或者多个工具进行分析
    直到认为分析完成
    """
    logging.info("--- 步骤: 执行工具节点 ---")
    ## 工具箱
    tools = []
    ## 模拟工具调用成功
    state["causal_analysis_result"] = {"success": True, "data": {"nodes": ["A", "B"], "edges": [["A", "B"]]}} # Placeholder
    
    
    tool_response = AIMessage(content="信息完备：工具执行完成，获得了因果分析结果。", name="execute_tools")
    state["messages"].append(tool_response)
    
    return {
        "causal_analysis_result": state["causal_analysis_result"],
        "messages": state["messages"]
    }

class PostprocessOutput(BaseModel):
    """后处理步骤的结构化输出，包含补充的参数和说明。"""
    supplemented_parameters: Optional[dict] = Field(
        default_factory=dict,
        description="模拟补充的参数，以键值对形式存在。如果没有补充，则为空字典。"
    )
    explanation: str = Field(
        ...,
        description="向用户解释为什么补充了这些参数（如果补充了），以及这些参数对生成最终报告的影响。如果未补充，则说明结果已很完整。"
    )

def postprocess_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    后处理模块：
    主要是对最后输出的参数，图进行最后的检查的处理
    如果需要补充什么，模拟添加参数
    """
    logging.info("--- 步骤: 后处理节点 ---")
    analysis_result = state["causal_analysis_result"]
    knowledge_base_result = state["knowledge_base_result"]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个专业的AI因果分析助手。
你的任务是审查现有的分析结果，并判断是否需要补充额外的信息或参数，以生成一份更全面、更准确的因果分析报告。

# 当前状态摘要
1. 因果分析结果: {analysis_result}
2. 知识库查询结果: {knowledge_base_result}

# 你的任务
1.  检查以上信息是否足以生成一份高质量的报告。
2.  如果信息不完整，请在`supplemented_parameters`中**模拟**添加必要的参数。
3.  在`explanation`字段中，清晰地解释你的决策。如果补充了参数，请说明补充了什么以及为什么；如果未补充，请确认信息完整。

请严格按照`PostprocessOutput`的格式输出。"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    runnable = prompt | llm.with_structured_output(PostprocessOutput)

    logging.info("正在调用LLM进行后处理决策...")
    structured_response = runnable.invoke({
        "messages": state["messages"],
        "analysis_result": analysis_result,
        "knowledge_base_result": knowledge_base_result
    })
    logging.info(f"LLM后处理决策结果: {structured_response}")
    
    # 创建一条新的AI消息，向用户解释后处理步骤的决策
    response_message = AIMessage(
        content=structured_response.explanation,
        name="postprocess"
    )
    
    state["messages"].append(response_message)
    state["postprocess_result"] = structured_response.dict()

    return {
        "messages": state["messages"],
        "postprocess_result": state["postprocess_result"]
    }

def report_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    报告模块：
    主要是对所有的参数生成一份报告
    
    """
    logging.info("--- 步骤: 报告模块 ---")
    causal_analysis_result = state["causal_analysis_result"]
    knowledge_base_result = state["knowledge_base_result"]
    postprocess_result = state["postprocess_result"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                ("system",
                 """
                 你是一个专业的AI助手的路由中枢。
                 你的任务是根据用户的对话历史和当前状态，按照要求的报告格式生成一份综合的，完整的因果领域报告
                 # 当前状态摘要
                 1. 因果分析结果：{causal_analysis_result}
                 2. 知识库结果：{knowledge_base_result}
                 3. 后处理结果：{postprocess_result}
                 
                 # 报告格式要求
                 1. 报告格式为markdown格式
                 2. 报告内容包括：
                    - 分析总结：需要概括性的总结因果分析结果
                    - 分析过程：需要详细描述分析的过程
                    - 分析结果：总结分析                          
                 """
                 
                 ),
                MessagesPlaceholder(variable_name="messages"),
            )
        ]
    )
    # 格式化字符串输出
    runnable = prompt | llm | StrOutputParser()
    response = runnable.invoke({
        "messages": state["messages"],
        "causal_analysis_result": causal_analysis_result,
        "knowledge_base_result": knowledge_base_result,
        "postprocess_result": postprocess_result
    })

    logging.info(f"LLM报告结果: {response}")
    
    state["final_report"] = response
    return {"final_report": state["final_report"]}

def normal_chat_node(state: CausalChatState) -> dict:
    """
    Represents "正常问答".
    This is for when the agent determines it's a simple chat conversation.
    """
    logging.info("--- 步骤: 普通问答节点 ---")

    last_message = state["messages"][-1].content
    state["final_report"] = f"回复: {last_message}" # Placeholder for simple reply
    return {"final_report": state["final_report"]}


def ask_human_node(state: CausalChatState) -> dict:
    """
    人机交互模块
    主要是为了向用户询问更多信息，以便继续分析
    """
    logging.info("--- 步骤: 人机交互节点 ---")
    question = "我需要更多信息才能继续。例如，您想分析哪个文件？" # Placeholder question
    state["ask_human"] = question
    return {"ask_human": state["ask_human"]}
