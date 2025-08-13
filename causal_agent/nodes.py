from .state import CausalChatState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
import logging
import asyncio
import json
import io
import pandas as pd
import mysql.connector
from mcp import ClientSession

from config.settings import settings

# --- 数据库辅助函数 ---
# 这些函数帮助节点与应用程序的数据库进行交互，以获取文件等资源。
def get_db_connection():
    """创建并返回一个MySQL数据库连接。"""
    try:
        return mysql.connector.connect(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE
        )
    except mysql.connector.Error as err:
        logging.error(f"Agent Node: MySQL 连接错误: {err}")
        raise

def get_file_content(user_id: int, filename: str) -> bytes | None:
    """从数据库为指定用户获取文件内容。"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT file_content FROM uploaded_files WHERE user_id = %s AND original_filename = %s ORDER BY last_accessed_at DESC LIMIT 1",
                (user_id, filename)
            )
            result = cursor.fetchone()
            return result['file_content'] if result else None
    except mysql.connector.Error as e:
        logging.error(f"Agent Node: 从数据库获取文件 '{filename}' (用户ID: {user_id}) 时出错: {e}")
        return None

def get_recent_file(user_id: int) -> tuple[bytes | None, str | None]:
    """获取用户最近上传或访问的文件的内容和名称。"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT file_content, original_filename FROM uploaded_files WHERE user_id = %s ORDER BY last_accessed_at DESC LIMIT 1",
                (user_id,)
            )
            result = cursor.fetchone()
            if result:
                return result['file_content'], result['original_filename']
            return None, None
    except mysql.connector.Error as e:
        logging.error(f"Agent Node: 为用户 {user_id} 获取最近文件时出错: {e}")
        return None, None

# --- 1. 定义LLM决策的结构化输出模型 ---
class RouteQuery(BaseModel):
    """定义Agent决策的选项。"""
    route: Literal["postprocess", "fold", "normal_chat"] = Field(
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
2. `fold`: 如果用户想要进行因果分析 (例如，对话中提到“分析”、“处理数据”或与“因果推断”相关的用语)，但我们还没有分析结果 ({has_tool_results} is False)，选择此路径以启动文件加载模块。
3. `normal_chat`: 如果用户的提问只是一个与因果领域不相关的消息，不需要调用任何复杂的因果分析工具，选择此路径。

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
    elif structured_response.route == 'fold':
        response_message = AIMessage(content="决策：信息不全，启动文件加载模块。", name="agent")
    else: # 'normal_chat'
        response_message = AIMessage(content="决策：普通问答。", name="agent")

    state["messages"].append(response_message)
    return {"messages": state["messages"]}

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    从Pandas DataFrame中提取一个结构化的摘要。
    """
    summary = {}
    summary['n_rows'] = len(df)
    summary['n_cols'] = len(df.columns)
    # 统计列名
    summary['columns'] = df.columns.tolist()
    
    # 数据结构
    data_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 20:
                 data_types[col] = 'Categorical (from Numeric)'
            else:
                 data_types[col] = 'Numeric'
        else:
            data_types[col] = 'Categorical'
    summary['data_types'] = data_types
    
    return summary

class foldQuery(BaseModel):
    """定义了预处理步骤中用于从用户对话里提取文件名的模型。"""
    filename: Optional[str] = Field(
        None, 
        description="从用户对话中识别出的要分析的数据文件名。如果用户没有明确提及，请留空。"
    )

class fold_processQuery(BaseModel):
    """定义fold节点决策的选项。"""
    route: Literal["preprocess", "ask_human"] = Field(
        ...,
        description="根据参数验证的结果，决定是执行工具还是询问用户。"
    )
def fold_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    文件加载与解析节点。
    1.  使用LLM从对话中提取文件名。
    2.  从数据库加载文件内容。
    3.  使用Pandas解析数据并生成摘要。
    4.  将所有结果存入状态。
    5.  进行判断，如果参数充足，进入preprocessing节点，如果参数不充足，则进入人机循环节点
    """
    logging.info("--- 步骤: 文件加载与解析节点 ---")
    user_id = state.get("user_id")

    # 1. 使用LLM提取文件名
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """你是一个智能助手，你的任务是从用户的最新消息中识别出他们想要分析的文件名。
如果用户明确提到了一个文件名（通常以 `.csv` 结尾），请提取它。
如果用户只是说“分析数据”或“用最新的文件”，没有指定具体名称，请将 `filename` 字段留空。

示例:
- 用户: "用 `marketing_campaign.csv` 帮我分析一下..." -> 提取: `filename='marketing_campaign.csv'`
- 用户: "分析一下我的数据" -> 提取: `filename=None`
"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    runnable = prompt | llm.with_structured_output(foldQuery)
    extracted = runnable.invoke({"messages": state["messages"]})
    
    filename = extracted.filename
    
    try:
        if filename:
            file_content_bytes = get_file_content(user_id, filename)
            loaded_filename = filename
        else:
            file_content_bytes, loaded_filename = get_recent_file(user_id)
    
    
        if not file_content_bytes:
            raise FileNotFoundError("找不到任何可供分析的文件。请先上传一个CSV文件。")

        file_content_str = file_content_bytes.decode('utf-8')
        df = pd.read_csv(io.StringIO(file_content_str))
        data_summary = get_data_summary(df)
    
    except Exception as e:
        error_msg = f"在文件加载或解析阶段发生错误: {e}"
        logging.error(error_msg, exc_info=True)
        return {"ask_human": error_msg}
        
## 也许会造成上下文超格
    state['file_content'] = file_content_str
    state['dataframe'] = df
    state['analysis_parameters'] = data_summary
    
    
    prompt = ChatPromptTemplate.from_messages([
    ("system",
        """你是一位严谨的AI数据分析师。你的任务是基于用户需求和已加载的数据信息，判断是否可以立即开始因果分析。

# 数据摘要
你正在处理的数据包含以下信息：
- **所有可用列名**: {columns}
- **每列的数据类型**: {data_types}

# 你的任务
仔细阅读下面的对话历史，并结合以上数据摘要，严格判断 `target` (目标/结果变量) 和 `treatment` (处理/干预变量) 是否都已明确指定，并且它们都**真实存在于`所有可用列名`列表中**。

# 你的决策选项
1. `execute_tools`: **当** `target` 和 `treatment` 两个变量都已在对话中被明确提及，并且它们的名字都精确地出现在 `{columns}` 列表中时，选择此路径。
**或者** 用户说明“请进行分析”而不描述任何参数,则证明分析全部参数，选择此路径
2. `ask_human`: **在任何其他情况下**，比如缺少 `target`、缺少 `treatment`，或者指定的变量名不在 `{columns}` 列表中，都必须选择此路径。

请做出你的决策。"""),
    MessagesPlaceholder(variable_name="messages"),
])
    
    runnable = prompt | llm.with_structured_output(PreprocessQuery)
    
    logging.info("正在调用LLM进行严格的参数验证...")
    structured_response = runnable.invoke({
        "messages": state["messages"],
        "columns": str(data_summary.get('columns', [])),
        "data_types": str(data_summary.get('data_types', {}))
    })
    logging.info(f"LLM参数验证决策结果: {structured_response.route}")

    if structured_response.route == 'execute_tools':
        response_message = AIMessage(content="决策：参数验证通过，信息完备，进入执行工具模块。", name="preprocess")
        return {"messages": state["messages"] + [response_message]}
    else:
        question = (
            "我需要您帮助我明确一下分析的变量。根据您上传的数据，我看到的可用的列有：\n"
            f"`{', '.join(data_summary.get('columns', []))}`\n\n"
            "请问您想分析哪个是处理变量（Treatment），哪个是结果变量（Target）？"
        )
        response_message = AIMessage(content=f"决策：参数不全或无效，需要向用户确认。", name="preprocess")

        return {
            "messages": state["messages"] + [response_message], "ask_human": question ,
            "file_content": state['file_content'],
            "dataframe": state['dataframe'],
            "analysis_parameters": state['analysis_parameters']
        }


class PreprocessQuery(BaseModel):
    """定义Agent决策的选项。"""
    route: Literal["execute_tools", "ask_human"] = Field(
        ...,
        description="根据参数验证的结果，决定是执行工具还是询问用户。"
    )

def preprocess_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    项目预处理模块
    加载用户上传的文件
    对输入数据进行数据预处理
    """
    logging.info("--- 步骤: 预处理与参数验证节点 ---")
    
    data_summary = state.get("analysis_parameters", {})
    if not data_summary:
        return {"ask_human": "抱歉，数据摘要信息丢失，无法进行参数验证。"}

   
    return {"messages": state["messages"] + [response_message], "ask_human": question}

def execute_tools_node(state: CausalChatState, mcp_session: ClientSession) -> dict:
    """
    执行工具模块（计算调用者）。
    1.  从状态中获取由 `fold_node` 准备好的文件内容（原始字符串）。
    2.  调用纯计算的 `perform_causal_analysis` MCP工具。
    3.  处理返回结果并更新状态。
    """
    logging.info("--- 步骤: 执行工具节点 ---")
    ## 工具箱
    tools = []
    ## 模拟工具调用成功
    state["causal_analysis_result"] = {"success": True, "data": {"nodes": ["A", "B"], "edges": [["A", "B"]]}} # Placeholder
    
    file_content = state.get("file_content")

    if not file_content:
        # 如果预处理没有成功加载数据，则无法继续
        logging.warning("execute_tools_node：状态中缺少 'file_content'，无法执行工具。")
        state["ask_human"] = "数据加载步骤似乎失败了，我无法执行分析。请您重试或检查文件。"
        return state

    tool_call_kwargs = {"csv_data": file_content}
    logging.info(f"正在调用 MCP 工具 'perform_causal_analysis'...")

    try:
        async def call_tool_async():
            return await mcp_session.call_tool("perform_causal_analysis", tool_call_kwargs)
        
        tool_response_obj = asyncio.run(call_tool_async())
        tool_response_text = tool_response_obj.content[0].text
        tool_result = json.loads(tool_response_text)

        if tool_result.get("success"):
            state["causal_analysis_result"] = tool_result
            response_message = AIMessage(
                content="信息完备：工具执行完成，获得了因果分析结果。", 
                name="execute_tools"
            )
            state["messages"].append(response_message)
            return {
                "causal_analysis_result": state["causal_analysis_result"],
                "messages": state["messages"]
            }
        else:
            error_message = tool_result.get("message", "未知工具错误")
            logging.warning(f"工具执行失败: {error_message}")
            state["ask_human"] = f"分析失败：{error_message}"
            return {"ask_human": state["ask_human"]}

    except Exception as e:
        logging.error(f"调用 MCP 工具时发生严重错误: {e}", exc_info=True)
        state["ask_human"] = f"执行分析工具时发生意外的系统错误: {e}"
        return {"ask_human": state["ask_human"]}


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
    return {"final_report": f"回复: {last_message}"}


def ask_human_node(state: CausalChatState) -> dict:
    """
    人机交互模块
    主要是为了向用户询问更多信息，以便继续分析
    """
    logging.info("--- 步骤: 人机交互节点 ---")
    question = "我需要更多信息才能继续。例如，您想分析哪个文件？" # Placeholder question
    state["ask_human"] = question
    return {"ask_human": state["ask_human"]}
