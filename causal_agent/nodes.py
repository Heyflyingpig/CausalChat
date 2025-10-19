from .state import CausalChatState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any, List, Tuple, Dict
import logging
import asyncio
import json
import io
import pandas as pd
import numpy as np
import networkx as nx
import mysql.connector
from mcp import ClientSession



from config.settings import settings

## 导入人设
from causal_agent.back_prompt import data_prompt
## 知识库查询人设
from causal_agent.back_prompt import causal_rag_prompt
## 报告人设
from causal_agent.back_prompt import causal_report_prompt
# 数据库
from Database.agent_connect import get_file_content, get_recent_file
# 定义LLM决策的结构化输出模型 ---
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
    
    # 构建引导LLM决策的Prompt ---
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
    
    # 构建并调用LLM链 

    runnable = prompt | llm.with_structured_output(RouteQuery)
    
    logging.info("正在调用LLM进行路由决策...")
    structured_response = runnable.invoke({
        "messages": state["messages"],
        "has_tool_results": has_tool_results
    })
    logging.info(f"LLM决策结果: {structured_response.route}")

    # 根据LLM的结构化决策，生成用于路由的消息 
    if structured_response.route == 'postprocess':
        response_message = AIMessage(content="决策：信息完备，进入后处理模块。", name="agent")
    elif structured_response.route == 'fold':
        response_message = AIMessage(content="决策：信息不全，启动文件加载模块。", name="agent")
    else: # 'normal_chat'
        response_message = AIMessage(content="决策：普通问答。", name="agent")

    state["messages"].append(response_message)
    return {"messages": state["messages"]}

class foldQuery(BaseModel):
    """从用户对话中提取文件名及因果分析所需的关键参数。"""
    filename: Optional[str] = Field(
        None,
        description="从用户对话中识别出的要分析的数据文件名 (e.g., 'data.csv')。如果未明确提及，则留空。"
    )
    target: Optional[str] = Field(
        None,
        description="从用户对话中识别出的目标变量(target)或结果变量(outcome)。如果未提及，则留空。"
    )
    treatment: Optional[str] = Field(
        None,
        description="从用户对话中识别出的处理变量(treatment)或干预变量(intervention)。如果未提及，则留空。"
    )

## fold节点用到的函数
from Processing.fold_processing import get_data_summary
from Processing.fold_verify import validate_analysis
from Processing.data_visualize import generate_visualizations
from knowledge_base.query_rag import get_rag_response


def fold_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    文件加载、解析与验证节点。
    1.  使用LLM从对话中一次性提取文件名、目标和处理变量。
    2.  从数据库加载文件内容。
    3.  运行 get_data_summary 进行全面的数据分析。
    4.  调用 validate_analysis 进行严格的条件验证。
    5.  根据验证结果，决策进入 'preprocess' 节点或 'ask_human' 节点。
    """
    logging.info("--- 步骤: 文件加载、解析与验证节点 ---")
    user_id = state.get("user_id")

    # 1. 使用LLM一次性提取文件名和分析意图
    prompt = ChatPromptTemplate.from_messages([
            ("system",
            """你是一个智能助手，你的任务是从用户的最新消息中识别出以下信息：
            1.  用户想要分析的文件名 (通常以 `.csv` 结尾)。
            2.  用户关心的目标变量 (target/outcome)。
            3.  用户想要评估效果的处理变量 (treatment/intervention)。

            - 如果用户明确提到了文件名，请提取它。
            - 如果用户只是说“分析数据”或“用最新的文件”，没有指定具体名称，请将 `filename` 字段留空。
            - 如果用户提到了目标或处理变量，请提取它们。如果没提，就留空。

            示例:
            - 用户: "用 `marketing_campaign.csv` 帮我分析一下'销售额'和'促销活动'的关系..."
            -> 提取: `filename='marketing_campaign.csv'`, `target='销售额'`, `treatment='促销活动'`
            - 用户: "分析一下我的数据，看看是什么影响了客户流失"
            -> 提取: `filename=None`, `target='客户流失'`, `treatment=None`
            - 用户: "帮我跑一下最新的数据"
            -> 提取: `filename=None`, `target=None`, `treatment=None`
            """),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    runnable = prompt | llm.with_structured_output(foldQuery)
    extracted = runnable.invoke({"messages": state["messages"]})
    filename = extracted.filename
    target = extracted.target
    treatment = extracted.treatment
    
    try:
        if filename:
            file_content_bytes = get_file_content(user_id, filename)
            
            # 注意这里的文件名后续并没有用到
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
        state["ask_human"] = error_msg
        
        recommend_message = AIMessage(content=f"决策：文件加载或解析阶段发生错误: {e}", name="fold")
        state["messages"].append(recommend_message)
        return state

    ## 也许会造成上下文超格
    state['file_content'] = file_content_str
    state['dataframe'] = df
    state['analysis_parameters'] = data_summary
    
    # 4. 运行确定性验证
    is_ready, issues, recommends = validate_analysis(
        data_summary, 
        target=target, 
        treatment=treatment
    )

    # 5. 根据验证结果决策
    if is_ready == 0:
        logging.info("验证通过，进入预处理节点。")
        ## 这里是完全替换还是补充呢
        recommend_message = AIMessage(content = "决策：信息完备，进入预处理节点。", name="fold")
        state["messages"].append(recommend_message)
        state['analysis_parameters'].update({"target": target, "treatment": treatment})


        # 针对建议，生成提示
        if recommends:
            recommend_message = AIMessage(content=f"决策：信息完备，进入预处理节点。温馨提示：\n- {'\n- '.join(recommends)}", )
            state["messages"].append(recommend_message)
        
        return state
    
    else:
        logging.warning(f"验证失败，需要人工干预。原因: {', '.join(issues)}")
        
        # 对于issue中有存在变量缺失的情况的，进行修正询问，对于数据有问题，进行数据补充询问
        has_param_issue = any("目标变量" in issue or "处理变量" in issue for issue in issues)
        has_data_quality_issue = any("缺失" in issue or "样本量" in issue or "常数列" in issue or "高基数" in issue or "ID列" in issue for issue in issues)

        call_to_action = "请您根据上述问题进行调整。" # 通用备用方案
        if has_param_issue:
            call_to_action = "请您根据上述问题，明确或修正'目标变量'和'处理变量'的指定。"
        elif has_data_quality_issue:
            call_to_action = "您的数据似乎存在一些质量问题。请您考虑对数据进行清洗，或上传一份新的文件。"
        
        columns_list = data_summary.get('columns', [])
        question = (
            "为了开始因果分析，我需要您的帮助来解决以下问题：\n"
            f"- {'\n- '.join(issues)}\n\n"
            f"作为参考，您的数据中包含以下可用列：\n`{', '.join(columns_list)}`\n\n"
            f"**{call_to_action}**"
        )
        state["ask_human"] = question
        recommend_message = AIMessage(content=f"决策：信息不全，需要人工干预。{question}", name="fold")
        state["messages"].append(recommend_message)
        return state


def preprocess_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    项目预处理模块:
    1.  从状态(state)中加载 DataFrame 和数据摘要。
    2.  调用 `generate_visualizations` 生成数据图表。
        - 如果缺少可视化库 (seaborn, matplotlib)，会跳过此步并向用户发出警告。
    3.  调用 LLM 对数据摘要进行自然语言总结。
    4.  将图表和总结存入状态，然后直接进入下一步。
    """
    logging.info("--- 步骤: 数据预处理与分析节点 ---")

    df = state.get("dataframe")
    analysis_parameters = state.get("analysis_parameters", {})

    if df is None or not analysis_parameters:
        error_msg = "无法执行预处理，因为数据或其摘要信息在状态中丢失。"
        logging.error(error_msg)
        state["ask_human"] = error_msg
        recommend_message = AIMessage(content=f"决策：无法执行预处理，因为数据或其摘要信息在状态中丢失。", name="preprocess")
        state["messages"].append(recommend_message)
        return state

    # 生成可视化图表 
    visualizations = {}
    try:
        visualizations = generate_visualizations(df, analysis_parameters)
        state["visualizations"] = visualizations
        logging.info("数据可视化图表已成功生成。")
    
    except Exception as e:
        logging.error(f"生成数据可视化时发生未知错误: {e}", exc_info=True)
        # 准备一条消息，通知用户未知错误
        error_message = AIMessage(
            content=f"生成数据可视化图表时遇到一个未知错误: {e}",
            name="preprocess"
        )
        state["messages"].append(error_message)

    # 3. 调用LLM进行自然语言总结
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """
             system role: {system_role}
             mission:
            你的任务是根据提供的数据摘要信息，为即将进行的因果分析撰写一段简洁明了的自然语言总结。

            # 数据摘要信息:
            {data_summary}

            # 你的任务:
            1.  **开篇总结**: 简要说明数据集的规模（行数和列数）。
            2.  **目标变量和处理变量的摘录**: 对输入数据中的“target”和“treatment”进行摘取，并告知用户目前处理的变量是这两个变量。
            3.  **风险提示**: 提及数据中存在的潜在问题，例如高缺失值列、常数列、高基数分类变量或疑似ID列。
            4.  **结论**: 给出一个总体评价，说明数据是否已准备好进行下一步的因果分析。

            请使用清晰、专业的语言，让非技术人员也能理解数据的基本状况。
            """),
        ]
    )
    
    
    runnable = prompt | llm | StrOutputParser()
    
    logging.info("正在调用LLM生成数据分析总结...")
    
    
    preprocess_summary = runnable.invoke({
        "data_summary": json.dumps(analysis_parameters, indent=2, ensure_ascii=False),
        "system_role": data_prompt()
    })
    logging.info(f"LLM数据总结结果: {preprocess_summary}")

    # 4. 更新状态
    state["preprocess_summary"] = preprocess_summary
    
    summary_message = AIMessage(
        content= "决策：数据预处理完成，进入工具处理路由",
        name="preprocess"
    )
    state["messages"].append(summary_message)

    return state



class RagQuestion(BaseModel):
    """用于生成知识库查询问题的模型。"""
    questions: List[str] = Field(
        default_factory=list,
        description="根据对话历史和数据摘要，为知识库生成一个或多个精确、具体的问题列表。"
    )

def execute_tools_node(state: CausalChatState, mcp_session: ClientSession, llm: ChatOpenAI) -> dict:
    """
    执行工具模块 (重构版)。
    1.  **并行执行**: 同时启动因果分析 (通过MCP) 和知识库查询 (RAG)。
    2.  **准备输入**: 
        -   对于因果分析，从状态中获取文件内容的原始字符串。
        -   对于知识库，首先调用LLM根据对话历史和数据摘要动态生成一个查询问题。
    3.  **调用工具**:
        -   异步调用 MCP 的 `perform_causal_analysis` 工具。
        -   同步调用 RAG 的 `get_rag_response` 函数 (因其内部是同步的)。
    4.  **结果处理**: 收集两个工具的返回结果，处理可能的异常，并更新状态。
    """
    
    logging.info("--- 步骤: 执行工具节点 ---")
    
    ## 获取编码的文件信息
    file_content = state.get("file_content")

    ## 获取分析参数
    analysis_parameters = state.get("analysis_parameters", {})

    # -MCP
    async def run_mcp_task():
        logging.info("正在启动 MCP 工具 'perform_causal_analysis'...")
        try:
            tool_call_kwargs = {"csv_data": file_content}
            tool_response_obj = await mcp_session.call_tool("perform_causal_analysis", tool_call_kwargs)
            tool_response_text = tool_response_obj.content[0].text
            tool_result = json.loads(tool_response_text)
            logging.info("MCP 工具调用成功。")
            return tool_result
        except Exception as e:
            logging.error(f"调用 MCP 工具时发生严重错误: {e}", exc_info=True)
            return {"success": False, "message": f"执行分析工具时发生意外的系统错误: {e}"}

    # RAG 

    def run_rag_query_task():
        logging.info("正在启动 RAG 知识库查询...")
        try:
            # 1. 调用LLM动态生成问题
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 """
                 system role: {system_role}
                
                你是一个因果数据分析领域的专家，你的任务是根据用户的对话历史和当前的数据摘要，识别出其中需要通过知识库进行澄清的关键概念或潜在问题。

                # 数据摘要:
                {data_summary}

                # 你的任务:
                综合以上信息，生成一个包含多个个问题的JSON列表，并赋值给 'questions' 字段。这些问题应该简洁、明确，旨在从知识库中检索信息，以帮助用户更好地理解当前分析的背景、方法论或潜在风险。

                例如:
                - 如果用户提到了'混杂因子'，你可以生成一个问题："什么是混杂因子，以及如何在因果分析中控制它？"
                - 如果数据显示有大量缺失值，你可以生成一个问题："数据缺失在因果分析中会引入哪些类型的偏倚？"
                - 如果分析涉及时间序列数据，你可以生成一个问题："在处理时间序列数据时，PC算法有哪些局限性？"

                请严格按照RagQuestion的格式输出一个问题列表。"""),
                MessagesPlaceholder(variable_name="messages"),
            ])
            
            question_generator_runnable = rag_prompt | llm.with_structured_output(RagQuestion)
            
            logging.info("正在调用LLM生成RAG查询问题...")
            
            generated_question_obj = question_generator_runnable.invoke({
                "messages": state["messages"],
                "data_summary": json.dumps(analysis_parameters, indent=2, ensure_ascii=False),
                "system_role": causal_rag_prompt()
            })
            questions = generated_question_obj.questions
            logging.info(f"LLM生成的RAG问题列表: {questions}")
            
            # 生成的回答是一个列表
            rag_response = get_rag_response(questions)
            logging.info("RAG 知识库查询成功。")
            return {"success": True, "response": rag_response}
        except Exception as e:
            logging.error(f"执行 RAG 查询时发生错误: {e}", exc_info=True)
            return {"success": False, "response": f"查询知识库时发生错误: {e}"}

    # asyncio.run 会启动事件循环来运行异步任务
    # 我们在一个异步函数中运行RAG的同步代码，以更好地管理它
    async def main_task():
        # 首先对异步函数进行包装，包装成一个task对象
        mcp_task = asyncio.create_task(run_mcp_task())
        
        # 在事件循环中运行同步的RAG任务
        loop = asyncio.get_running_loop()
        # 让函数在这里暂停，处理两个任务，直到两个任务都完成
        # 知识库查询是同步任务，同步任务如果和异步任务通过进行，会导致异步任务阻塞
        # 所以在这一步需要抽出一个空闲线程来运行知识库查询任务
        rag_result = await loop.run_in_executor(None, run_rag_query_task)
        
        causal_result = await mcp_task
        # 返回两个任务的结果
        return causal_result, rag_result

    causal_analysis_result, knowledge_base_result = asyncio.run(main_task())

    # --- 更新状态 ---
    state["causal_analysis_result"] = causal_analysis_result
    state["knowledge_base_result"] = knowledge_base_result

    # 根据主要工具（因果分析）的结果来决定下一步
    if causal_analysis_result.get("success"):
        response_message = AIMessage(
            content="信息完备：工具执行完成，获得了因果分析和知识库查询结果。", 
            name="execute_tools"
        )
        state["messages"].append(response_message)
        state["tool_call_request"] = True
    
    else:
        error_message = causal_analysis_result.get("message", "未知工具错误")
        logging.warning(f"工具执行失败: {error_message}")
        response_message = AIMessage(
            content=f"决策：工具执行失败：{error_message}",
            name="execute_tools"
        )
        state["messages"].append(response_message)
        state["tool_call_request"] = False

    return state



# 环路检测模块
from Postprocessing.cycles_check.detect_cycles import detect_cycles
from Postprocessing.cycles_check.extract_causal_return import extract_adjacency_matrix
from Postprocessing.cycles_check.fix_cycles import fix_cycles_with_llm


# 边评估模块
from Postprocessing.evaluate_edge.evaluate_edge_llm import evaluate_edges_with_llm

def postprocess_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    后处理模块：
    1. 提取并验证因果图结构
    2. 环路检测和修正
    3. LLM辅助评估关键边的合理性
    4. 准备修正记录和格式化数据供报告使用
    
    技术说明：
        - 使用networkx进行图结构分析
        - 使用LLM进行环路修正和边评估决策
        - 所有修正操作都会被详细记录
    """
    logging.info("--- 步骤: 后处理节点 ---")
    
    try:
        # 提取原始因果图
        analysis_result = state["causal_analysis_result"]
        adjacency_matrix, node_names = extract_adjacency_matrix(analysis_result)
        
        # 如果提取失败，返回错误
        if adjacency_matrix.size == 0:
            error_msg = "无法从分析结果中提取有效的因果图数据。"
            logging.error(error_msg)
            state["messages"].append(AIMessage(content=f"决策：{error_msg}", name="postprocess"))
            state["postprocess_result"] = {"error": error_msg}
            return state
        
        logging.info(f"提取到 {len(node_names)} 个节点的因果图")
        
        # 创建原始图的副本用于修正
        working_matrix = adjacency_matrix.copy()
        
        # 环路检测和修正
        has_cycle, cycles = detect_cycles(working_matrix, node_names)
        if has_cycle:
            logging.info(f"检测到 {len(cycles)} 个环路，开始LLM辅助修正...")
            working_matrix = fix_cycles_with_llm(
                working_matrix, 
                cycles, 
                node_names,
                llm, 
                state
            )
            # 再次检测以确认环路已被消除
            has_cycle_after, _ = detect_cycles(working_matrix, node_names)
            if has_cycle_after:
                logging.warning("警告：部分环路仍然存在，可能需要人工干预。")

            else:
                logging.info("所有环路已成功修正！")
        
        # LLM评估关键边
        analysis_parameters = state.get("analysis_parameters", {})
        ananlysis_result = state.get(("causal_analysis_result"),{})
        if isinstance(ananlysis_result, str):
            critical_edges = ananlysis_result.get("raw_results", {}).get("edges", [])
        else:
            critical_edges = []
        
        edge_evaluations = {}
        if critical_edges:
            logging.info(f"识别到 {len(critical_edges)} 条关键边，开始LLM评估...")
            edge_evaluations = evaluate_edges_with_llm(critical_edges, state, llm)
        else:
            logging.info("未识别到需要评估的关键边")
        
        
        # 准备结构化输出
        postprocess_result = {
            "original_graph": state["causal_analysis_result"].get("data", {}),
            "revised_graph": edge_evaluations.get("decision", []),
            "revision_summary": edge_evaluations.get("reason", ""),
            "had_cycles": has_cycle,
            "num_cycles_fixed": len(cycles) if has_cycle else 0
        }
        
        state["postprocess_result"] = postprocess_result
        
        
        # 如果有环路被修正，添加额外说明
        if has_cycle:
            explanation = f"\n\n**注意**：原始图中检测到 {len(cycles)} 个环路，部分已通过LLM辅助决策进行修正。理由如下：{edge_evaluations.get('reason', '')}"      
            response_message = AIMessage(
                content=explanation,
                name="postprocess"
            )
            state["messages"].append(response_message)
        
        response_message = AIMessage(
            content="决策：后处理完成，准备进入报告生成阶段",
            name="postprocess"
        )
        state["messages"].append(response_message)

        logging.info("后处理完成，准备进入报告生成阶段")
        
        return state
        
    except Exception as e:
        state["postprocess_result"] = {"error": str(e) + f"\n\n将使用原始分析结果继续生成报告。"}
        # 异常处理：记录错误但不中断流程
        error_message = AIMessage(
            content=f"后处理遇到问题: {str(e)}\n\n将使用原始分析结果继续生成报告。",
            name="postprocess"
        )
        state["messages"].append(error_message)
        
        return state


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
                 system role: {system_role}
                 
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
        "postprocess_result": postprocess_result,
        "system_role": causal_report_prompt()
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
    此节点是图暂停以等待用户输入的地方。图的编译设置为在此节点运行之前中断。
    主应用逻辑将从状态的 'ask_human' 字段中获取问题，并在获得用户响应后，
    用更新过的消息历史恢复执行。
    """
    logging.info("--- 步骤: 人机交互节点 (已从中断中恢复，继续执行) ---")
    if "ask_human" in state:
        state.pop("ask_human")

    return state
