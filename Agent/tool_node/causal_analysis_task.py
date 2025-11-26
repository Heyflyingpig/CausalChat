"""
@task封装因果分析任务
实现 LLM 动态选择 MCP 工具的机制
"""
from langgraph.func import task
import logging
from mcp import ClientSession
from typing import List, Dict, Optional
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from Agent.causal_agent.state import CausalChatState
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

class ToolSelection(BaseModel):
    """LLM 工具选择的结果模型"""
    selected_tool: str = Field(..., description="选择的工具名称")
    reason: str = Field(..., description="选择该工具的理由")


def _format_tools_for_prompt(tools) -> str:
    """
    将 MCP 工具列表格式化为 Prompt 可读的字符串

    Args:
        tools: mcp_session.list_tools() 返回的工具列表

    Returns:
        格式化后的工具描述字符串
    """
    tool_descriptions = []
    for i, tool in enumerate(tools, 1):
        # MCP 工具对象通常有 name 和 description 属性
        name = tool.name
        description = tool.description or "无描述"
        tool_descriptions.append(f"{i}. **{name}**\n   描述: {description}")

    return "\n\n".join(tool_descriptions)


async def select_causal_tool(
        mcp_session: ClientSession,
        llm: ChatOpenAI,
        state: CausalChatState
    ) -> Optional[str]:
    """
    让 LLM 根据可用工具描述和用户上下文，选择最合适的因果分析工具

    Args:
        mcp_session: MCP 客户端会话
        llm: LangChain LLM 实例
        user_context: 用户的分析意图/上下文（可选）

    Returns:
        选中的工具名称，如果选择失败返回 None
    """
    logging.info("正在获取 MCP 可用工具列表...")
    tools_response = await mcp_session.list_tools()
    causal_tools = tools_response.tools

    if not causal_tools:
        logging.warning("MCP 服务器没有可用工具")
        return None

    tools_description = _format_tools_for_prompt(causal_tools)
    logging.info(f"找到 {len(causal_tools)} 个因果分析工具，让 LLM 选择...")


    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个因果分析工具选择助手。根据用户的分析需求和可用工具的描述，选择最合适的工具。
        
        ## 已有分析结果：
        {data_summary}
        {preprocess_summary}
        ## 可用的因果分析工具：
        {tools_description}

        ## 选择原则：
        1. 如果用户提到"隐变量"、"潜变量"、"未观测变量"、"混杂因子"，优先选择能处理隐变量的工具（如 OLC）
        2. 如果用户没有特殊要求，选择通用性最好的工具（如 PC）
        3. 如果用户提到特定算法名称，直接选择对应工具

        ## 输出要求：
        返回一个 JSON 对象，包含：
        - selected_tool: 选择的工具名称（必须是上面列出的工具名之一）
        - reason: 简要说明选择理由

        不要输出任何 Markdown 格式或额外文字。"""),
                ("human", "用户的分析需求/上下文：{messages}\n\n请选择最合适的工具。")
    ])

    try:
        runnable = prompt | llm | JsonOutputParser()
        response = runnable.invoke({
            "tools_description": tools_description,
            "messages": state["messages"],
            "data_summary": json.dumps(state.get("analysis_parameters", {}), indent=2, ensure_ascii=False),
            "preprocess_summary": state.get("preprocess_summary", ""),
        })

        selection = ToolSelection.model_validate(response)
        logging.info(f"LLM 选择工具: {selection.selected_tool}，理由: {selection.reason}")

        # 验证选择的工具确实存在
        valid_names = [t.name for t in causal_tools]
        if selection.selected_tool in valid_names:
            return selection.selected_tool
        else:
            logging.warning(f"LLM 选择的工具 '{selection.selected_tool}' 不在可用列表中，使用第一个工具")
            return causal_tools[0].name

    except Exception as e:
        logging.error(f"LLM 工具选择失败: {e}，使用默认工具")
        return causal_tools[0].name


@task
async def causal_analysis_task(
    file_content: str,
    mcp_session: ClientSession,
    llm: ChatOpenAI,
    state: CausalChatState
) -> Dict:
    """
    Task: 执行因果分析（通过 MCP，动态选择工具）

    流程：
        1. 调用 list_tools() 获取所有可用的 MCP 工具
        2. LLM 根据工具描述和用户上下文选择最合适的工具
        3. 调用选中的工具执行因果分析

    Args:
        file_content: CSV 文件内容字符串
        mcp_session: MCP 客户端会话
        llm: LangChain LLM 实例
        user_context: 用户的分析意图

    Returns:
        dict: 因果分析结果字典
    """
    logging.info("正在启动因果分析任务...")

    try:
        selected_tool = await select_causal_tool(mcp_session, llm, state)

        if not selected_tool:
            return {
                "success": False,
                "message": "没有找到可用的因果分析工具，请检查 MCP 服务器配置"
            }

        logging.info(f"使用工具: {selected_tool}")

        tool_response = await mcp_session.call_tool(
            selected_tool,
            {"csv_data": file_content}
        )

        result = json.loads(tool_response.content[0].text)
        result["selected_tool"] = selected_tool  # 记录使用的工具

        logging.info(f"Task: MCP 因果分析完成，使用工具: {selected_tool}")
        return result

    except Exception as e:
        logging.error(f"Task: MCP 调用失败: {e}")
        return {"success": False, "message": str(e)}