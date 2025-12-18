from .state import CausalChatState
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any, List, Tuple, Dict
import logging
import json
import io
import pandas as pd
import numpy as np
import networkx as nx
from mcp import ClientSession
from langgraph.func import task
from langgraph.types import interrupt

## 基本配置
from config.settings import settings

## 导入人设
## 因果分析人设
from .back_prompt import causal_prompt
## 数据分析人设
from .back_prompt import data_prompt
## 知识库查询人设
from .back_prompt import causal_rag_prompt
## 报告人设
from .back_prompt import causal_report_prompt

# 数据库
from Database.agent_connect import get_file_content, get_recent_file
# 处理llm的输出，提取json对象
from Agent.tool_node.excute_output import excute_output

class RouteQuery(BaseModel):
    """定义Agent决策的选项。"""
    route: Literal["postprocess", "fold", "normal_chat","inquiry_answer"] = Field(
        ...,
        description="根据用户的对话历史和意图，选择下一步应该走的路径。"
    )

# agentnode节点用于做初步的decision
def agent_node(state: CausalChatState, llm: ChatOpenAI) -> dict:

    """
    Agent节点，是图的起点，用于判断是否需要进入causal循环，
    根据当前状态强制LLM做出四选一的决策，然后将该决策转化为消息。
    """
    logging.info(" 步骤: Agent 节点 (LLM 决策) ")

    # 检查生成报告所需的结果是否已存在。
    has_tool_results = state.get('causal_analysis_result') is not None
    
    agent_prompt = """
            你是一个专业的AI助手路由中枢。你的任务是根据用户的对话历史和当前状态，决定下一步的最佳路径。
            
            # 用户需求或者对话历史:{messages}
            # 当前状态摘要:
            - 是否已获得分析工具的结果: {has_tool_results}
            - 是否已获取到了最终的报告：{final_report}

            # 你的决策选项:
            1. `postprocess`: 如果已经获得了因果分析结果 ({has_tool_results} is True)，可以选择此路径以进入后处理模块。
            2. `fold`: 如果用户想要进行因果分析 (例如，对话中提到“分析”、“处理数据”或与“因果推断”相关的用语)，但我们还没有分析结果 ({has_tool_results} is False)，选择此路径以启动文件加载模块。
            3. `normal_chat`: 如果用户的提问只是一个与因果领域不相关的消息，不需要调用任何复杂的因果分析工具，选择此路径。
            4. `inquiry_answer`: 如果已经获取到了最终的报告（{final_report} is not None），选择此路径以直接根据报告回答用户的问题。
            
            请根据下面的对话历史，做出你的选择。
            你必须按照RouteQuery返回一个只包含 "route" 键的 JSON 对象格式来返回你的决策。
            **绝对不要**在你的回复中包含任何Markdown格式（例如 ```json ... ```）。
            例如:
            {{
                "route": "postprocess"
            }}
            """
    # 构建引导LLM决策的Prompt 
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", agent_prompt),
            ("human", "请根据上述指示生成路径。"),
        ]
    )
    
    # 强制模型使用JSON模式，这是更可靠的方法
    structured_llm = llm.bind(
        response_format={"type": "json_object"}
    )
    
    # 构建并调用LLM链
    runnable = prompt | structured_llm | JsonOutputParser()
    
    logging.info("正在调用LLM进行路由决策...")
    try:
        decision_dict = runnable.invoke(
            {
                "messages": state["messages"][-1],
                "has_tool_results": has_tool_results,
                "final_report": state.get("final_report", None)
            }
        )
        # 用Pydantic模型解析和验证这个字典
        structured_response = RouteQuery.model_validate(decision_dict)
        route_decision = structured_response.route

    except Exception as e:
        # 这里的异常可能来自JSON解析，也可能来自Pydantic验证
        logging.warning(f"无法从LLM响应中解析或验证路由决策: {e}。将回退到 normal_chat。")
        route_decision = "normal_chat"  # 设置默认值

    logging.info(f"LLM决策结果: {route_decision}")

    # 根据LLM的结构化决策，生成用于路由的消息 
    if route_decision == 'postprocess':
        response_message = AIMessage(content="决策：信息完备，进入后处理模块。", name="agent")
    elif route_decision == 'fold':
        response_message = AIMessage(content="决策：信息不全，启动文件加载模块。", name="agent")
    elif route_decision == 'inquiry_answer':
        response_message = AIMessage(content="决策：报告已获取，进入根据报告追问模块。", name="agent")
    else: # 'normal_chat'
        response_message = AIMessage(content="决策：普通问答。", name="agent")

    # 只返回新消息，不修改 state["messages"]
    return {"messages": [response_message]}

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
from Agent.Processing.fold_processing import get_data_summary
from Agent.Processing.fold_verify import validate_analysis
from Agent.Processing.data_visualize import generate_visualizations
from Agent.knowledge_base.query_rag import get_rag_response


def fold_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    文件加载、解析与验证节点。
    1.  使用LLM从对话中一次性提取文件名、目标和处理变量。
    2.  从数据库加载文件内容。
    3.  运行 get_data_summary 进行全面的数据分析。
    4.  调用 validate_analysis 进行严格的条件验证。
    5.  根据验证结果，决策进入 'preprocess' 节点或 'ask_human' 节点。
    """
    logging.info(" 步骤: 文件加载、解析与验证节点 ")
    user_id = state.get("user_id")

    # 1. 使用LLM一次性提取文件名和分析意图
    prompt = ChatPromptTemplate.from_messages([
            ("system",
            """你是一个智能助手，你的任务是从用户的最新消息中识别出以下信息，并以JSON格式返回：
            1.  用户想要分析的文件名 (通常以 `.csv` 结尾)。
            2.  用户关心的目标变量 (target/outcome)。
            3.  用户想要评估效果的处理变量 (treatment/intervention)。

            - 如果用户明确提到了文件名，请提取它。
            - 如果用户只是说"分析数据"或"用最新的文件"，没有指定具体名称，请将 `filename` 字段设为 null（不要使用字符串 "None"）。
            - 如果用户提到了目标或处理变量，请提取它们。如果没提，请设为 null（不要使用字符串 "None"）。

            示例:
            - 用户: "用 `marketing_campaign.csv` 帮我分析一下'销售额'和'促销活动'的关系..."
            -> 提取: `filename='marketing_campaign.csv'`, `target='销售额'`, `treatment='促销活动'`
            - 用户: "分析一下我的数据，看看是什么影响了客户流失"
            -> 提取: `filename=null`, `target='客户流失'`, `treatment=null`
            - 用户: "帮我跑一下最新的数据"
            -> 提取: `filename=null`, `target=null`, `treatment=null`
            
            你必须严格按照 `foldQuery` 的 schema 返回一个 JSON 对象。
            **绝对不要**在你的回复中包含任何Markdown格式或解释性文字。
            **重要：如果某个字段为空，请使用 JSON 的 null 值，而不是字符串 "None"。**

            示例输出（所有字段都有值）:
            {{
                "filename": "marketing_campaign.csv",
                "target": "销售额",
                "treatment": "促销活动"
            }}
            
            示例输出（部分字段为空）:
            {{
                "filename": null,
                "target": "客户流失",
                "treatment": null
            }}
            
            """),
            MessagesPlaceholder(variable_name="messages"),
        ])
    try:
        runnable = prompt | llm | JsonOutputParser()
        response = runnable.invoke({"messages": state["messages"]})

        structured_response = foldQuery.model_validate(response)

        filename = structured_response.filename
        target = structured_response.target
        treatment = structured_response.treatment
        
        # 转换结构
        if filename == "None":
            filename = None
        if target == "None":
            target = None
        if treatment == "None":
            treatment = None
            
    except Exception as e:
        logging.error(f"无法从LLM响应中解析或验证提取信息: {e}。将返回错误值")
        filename = None
        target = None
        treatment = None
    
    loaded_filename = None
    
    logging.info(f"filename: {filename}, state.get('fold_name'): {state.get('fold_name')}")
    try:
        if filename :
            file_content_bytes = get_file_content(user_id, filename)
            state['fold_name'] = filename
            
            # 注意这里的文件名后续并没有用到
            loaded_filename = filename
        elif state.get('fold_name'):
            loaded_filename = state.get('fold_name')
            file_content_bytes = get_file_content(user_id, loaded_filename)

        else:
            file_content_bytes , loaded_filename = get_recent_file(user_id)

        if not file_content_bytes or not loaded_filename:
            raise FileNotFoundError("找不到任何可供分析的文件。请先上传一个CSV文件。")
        
        state['fold_name'] = loaded_filename
        file_content_str = file_content_bytes.decode('utf-8')
        df = pd.read_csv(io.StringIO(file_content_str))
        data_summary = get_data_summary(df)
    
    except Exception as e:
        error_msg = f"在文件加载或解析阶段发生错误: {e}"
        logging.error(error_msg, exc_info=True)
        
        # 使用 interrupt() 暂停并等待用户输入
        user_response = interrupt(error_msg)
        new_message = HumanMessage(content=user_response)
        
        return {"messages": [new_message]}

    ## 优化：只保存 file_content 和摘要，不保存 DataFrame
    # 原因：DataFrame 序列化体积大，file_content 可随时重新生成 DataFrame
    state['file_content'] = file_content_str
    # state['dataframe'] = df  # 避免序列化开销
    state['analysis_parameters'] = data_summary
    
    # 运行确定性验证
    is_ready, issues, recommends = validate_analysis(
        data_summary, 
        target=target, 
        treatment=treatment
    )

    # 根据验证结果决策
    if is_ready == 0 or is_ready == 1:
        logging.info("验证通过，进入预处理节点。")
        state['analysis_parameters'].update({"target": target, "treatment": treatment})

        # 收集需要返回的新消息
        new_messages = []
        recommend_message = AIMessage(content = "决策：信息完备，进入预处理节点。", name="fold")
        new_messages.append(recommend_message)

        # 针对建议，生成提示
        if recommends:
            recommend_message = AIMessage(content=f"决策：信息完备，进入预处理节点。温馨提示：\n- {recommends}")
            new_messages.append(recommend_message)
        
        return {"messages": new_messages, 
                "analysis_parameters": state['analysis_parameters'], 
                "fold_name": state['fold_name'], 
                "file_content": state['file_content'],
                "tool_call_request": False
                }
    
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
            f"- {issues}\n\n"
            f"作为参考，您的数据中包含以下可用列：\n`{', '.join(columns_list)}`\n\n"
            f"**{call_to_action}**"
        )
        
        # interrupt() 会立即暂停节点执行，返回 question 给调用者
        # 当用户提供输入后，interrupt() 会返回用户的输入值
        user_response = interrupt(question)
        
        logging.info(f"用户提供的响应: {user_response}")
        
        # 将用户的响应添加到消息历史，并返回一个路由消息
        # 让 fold_router 知道需要回到 agent 重新判断
        return {
            "messages": [
                HumanMessage(content=user_response),
                AIMessage(content="决策：已收到用户输入，返回 agent 重新判断", name="fold")
            ],
            "fold_name": state['fold_name'],
            "file_content": state['file_content'],
            "analysis_parameters": state['analysis_parameters']
        }


def preprocess_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    项目预处理模块:
    1.  从状态(state)中加载 DataFrame 和数据摘要。
    2.  调用 `generate_visualizations` 生成数据图表。
        - 如果缺少可视化库 (seaborn, matplotlib)，会跳过此步并向用户发出警告。
    3.  调用 LLM 对数据摘要进行自然语言总结。
    4.  将图表和总结存入状态，然后直接进入下一步。
    """
    logging.info(" 步骤: 数据预处理与分析节点 ")

    # 从 file_content 动态生成 DataFrame
    file_content = state.get("file_content")
    analysis_parameters = state.get("analysis_parameters", {})

    if file_content is None or not analysis_parameters:
        error_msg = "无法执行预处理，因为数据或其摘要信息在状态中丢失。"
        logging.error(error_msg)
        
        user_response = interrupt(error_msg)
        new_message = HumanMessage(content=user_response)
        
        return {"messages": [new_message]}

    # 动态生成 DataFrame（从 file_content）
    df = pd.read_csv(io.StringIO(file_content))
    logging.info(f"从 file_content 重新生成 DataFrame，shape: {df.shape}")

    # 生成可视化图表 
    visualizations = {}
    try:
        visualizations = generate_visualizations(df, analysis_parameters)
        state["visualizations"] = visualizations
        logging.info("数据可视化图表已成功生成。")
    
    except Exception as e:
        logging.error(f"生成数据可视化时发生未知错误: {e}", exc_info=True)
        # 可视化失败不阻断流程，记录日志即可
        # 不需要添加消息到state，继续执行后续步骤
    
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
            ("human", "请根据上述指示和提供的数据摘要，生成总结报告。")
        ]
    )
    
    
    runnable = prompt | llm | StrOutputParser()
    
    logging.info("正在调用LLM生成数据分析总结...")
    
    
    preprocess_summary = runnable.invoke({
        "data_summary": json.dumps(analysis_parameters, indent=2, ensure_ascii=False),
        "system_role": data_prompt()
    })
    logging.info(f"LLM数据总结结果: {preprocess_summary}")

    # 4. 更新状态并返回新消息
    state["preprocess_summary"] = preprocess_summary
    
    summary_message = AIMessage(
        content= "决策：数据预处理完成，进入工具处理路由",
        name="preprocess"
    )

    # 只返回新消息和需要更新的状态
    return {
        "messages": [summary_message],
        "preprocess_summary": state["preprocess_summary"],
        "visualizations": state.get("visualizations", {})
    }


from Agent.tool_node.causal_analysis_task import causal_analysis_task
from Agent.tool_node.rag_query_task import rag_query_task
from Agent.tool_node.rag_questions import get_rag_questions


def execute_tools_node(state: CausalChatState, mcp_session: ClientSession, llm: ChatOpenAI) -> dict:
    """
    执行工具模块。
    输入：mcp模块
    输出：因果分析结果和知识库查询结果
    """
    
    logging.info(" 步骤: 执行工具节点 ")
    
    ## 获取编码的文件信息
    file_content = state.get("file_content")

    # 并行执行任务并等待结果 
    logging.info(" 开始并行执行因果分析和RAG查询 ")

    rag_question_task = get_rag_questions(state, llm, num_questions=2)

    causal_task = causal_analysis_task(file_content, mcp_session, llm, state)

    rag_questions = rag_question_task.result()

    rag_task = rag_query_task(rag_questions)    
    
    causal_task_result = causal_task.result()

    rag_task_result = rag_task.result()
    
    logging.info(" 因果分析和RAG查询均已完成 ")
    
    state["causal_analysis_result"] = causal_task_result
    state["knowledge_base_result"] = rag_task_result
    
    # 根据主要工具（因果分析）的结果来决定下一步
    if causal_task_result.get("success"):
        response_message = AIMessage(
            content="信息完备：工具执行完成，获得了因果分析和知识库查询结果。", 
            name="execute_tools"
        )
        # 只返回新消息和需要更新的状态
        return {
            "messages": [response_message],
            "causal_analysis_result": state["causal_analysis_result"],
            "knowledge_base_result": state["knowledge_base_result"],
            "tool_call_request": True
        }
    
    else:
        error_message = causal_task_result.get("message", "未知工具错误")
        logging.warning(f"工具执行失败: {error_message}")
        response_message = AIMessage(
            content=f"决策：工具执行失败：{error_message}",
            name="execute_tools"
        )
        # 只返回新消息和需要更新的状态
        return {
            "messages": [response_message],
            "causal_analysis_result": state["causal_analysis_result"],
            "knowledge_base_result": state["knowledge_base_result"],
            "tool_call_request": False
        }



# 环路检测模块
from Agent.Postprocessing.cycles_check.detect_cycles import detect_cycles
from Agent.Postprocessing.cycles_check.extract_causal_return import extract_adjacency_matrix
from Agent.Postprocessing.cycles_check.fix_cycles import fix_cycles_with_llm

# 边评估模块
from Agent.Postprocessing.evaluate_edge.evaluate_edge_llm import evaluate_edges_with_llm

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
    logging.info(" 步骤: 后处理节点 ")
    
    try:
        # 提取原始因果图
        analysis_result = state["causal_analysis_result"]
        adjacency_matrix, node_names = extract_adjacency_matrix(analysis_result)
        
        # 如果提取失败，返回错误
        if adjacency_matrix.size == 0:
            error_msg = "无法从分析结果中提取有效的因果图数据。"
            logging.error(error_msg)
            # 只返回新消息和错误状态
            return {
                "messages": [AIMessage(content=f"决策：{error_msg}", name="postprocess")],
                "postprocess_result": {"error": error_msg}
            }
        
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
        
        
        # 收集需要返回的新消息
        new_messages = []
        
        # 如果有环路被修正，添加额外说明
        if has_cycle:
            explanation = f"\n\n**注意**：原始图中检测到 {len(cycles)} 个环路，部分已通过LLM辅助决策进行修正。理由如下：{edge_evaluations.get('reason', '')}"      
            new_messages.append(AIMessage(content=explanation, name="postprocess"))
        
        new_messages.append(AIMessage(
            content="决策：后处理完成，准备进入报告生成阶段",
            name="postprocess"
        ))

        logging.info("后处理完成，准备进入报告生成阶段")
        
        # 只返回新消息和后处理结果
        return {
            "messages": new_messages,
            "postprocess_result": state["postprocess_result"]
        }
        
    except Exception as e:
        postprocess_result = {"error": str(e) + f"\n\n将使用原始分析结果继续生成报告。"}
        # 异常处理：记录错误但不中断流程
        error_message = AIMessage(
            content=f"后处理遇到问题: {str(e)}\n\n将使用原始分析结果继续生成报告。",
            name="postprocess"
        )
        
        # 只返回新消息和错误状态
        return {
            "messages": [error_message],
            "postprocess_result": postprocess_result
        }

## 调用元数据
from Agent.Report.Metadata_sum import metadata_summary, metadata_mapping
def report_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    报告模块：
    主要是对所有的参数生成一份报告
    
    """
    logging.info(" 步骤: 报告模块 ")
    # 分离 system prompt 和 messages placeholder
    system_prompt_template = (
        """
         system role: {system_role}
         #输出语言：**请用英文回复**
         
         你的任务是根据用户的对话历史和当前状态，按照要求的报告格式生成一份综合的，完整的因果领域报告
         # 当前状态摘要
         1. 预处理结果：{preprocess_summary}
         2. 预处理元数据：{preprocess_meta_data}
         2. 因果分析结果：{causal_analysis_result}
         3. 知识库结果：{knowledge_base_result}
         4. 后处理结果：{postprocess_result}
         
        ## 报告结构要求
        1. **数据概览**：基于上述数据概览进行总结

        2. **数据可视化**：在合适的位置插入图表，帮助读者理解数据分布
            - 如果用户没有提到具体的变量类型，必须插入所有变量的图表，变量需要从预处理元数据中获取
            - 如果用户提到了具体的变量类型，则只插入该变量的图表，变量类型需要从预处理元数据中获取
        3. **分析过程**：详细描述因果分析的步骤和方法
        4. **分析结果**：总结主要发现和因果关系

        ## 图表插入规则
        - 当你想要插入某个图表时，直接在文本中使用对应的占位符(占位符见预处理元数据)
        - 例如：要展示年龄分布，就写 [[CHART:histogram_age]]
        - 占位符需要单独成行，前后空一行
        - 在占位符前后添加必要的文字说明，解释这个图表展示了什么
        ### 示例格式:
        #### 年龄分布特征
        从收集的数据来看，用户年龄主要集中在...

        [[CHART:histogram_age]]

        上图展示了年龄的分布情况，我们可以观察到...、
        """
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # 格式化字符串输出
    runnable = prompt | llm | StrOutputParser()
    
    meta_data = metadata_summary(state.get("analysis_parameters", {}), state.get("visualizations", {}))
    mapping_data = metadata_mapping(meta_data, state.get("visualizations", {}))
    
    # 在invoke时，将模板变量和消息历史分开传入
    response = runnable.invoke({
        "messages": state["messages"],
        "preprocess_meta_data": meta_data,
        "preprocess_summary": state.get("preprocess_summary", {}),
        "causal_analysis_result": state.get("causal_analysis_result", {}),
        "knowledge_base_result": state.get("knowledge_base_result", {}),
        "postprocess_result": state.get("postprocess_result", {}),
        "system_role": causal_report_prompt()
    })

    logging.info(f"LLM报告结果: {response}")

    report_complete_message = AIMessage(
        content="决策：因果分析报告已生成完成。",
        name="report"
    )
    # 占位符替换：将报告中的占位符替换为实际的 HTML 图片标签

    ## 注释:避免数据库中存入最终报告的html图片标签，导致数据库爆炸
    # final_report = response
    # try:
    #     for placeholder, base64_str in mapping_data.items():

    #         html_img = f'<img src="data:image/png;base64,{base64_str}" alt="{placeholder}" style="max-width:100%; height:auto; display:block; margin:20px 0;" />'
    #         final_report = final_report.replace(placeholder, html_img)
        
    #     logging.info(f"成功替换了 {len(mapping_data)} 个图表占位符")
    # except Exception as e:
    #     logging.error(f"替换图表占位符时发生错误: {e}", exc_info=True)
    #     # 如果替换失败，仍然返回原始报告（不含图片）
    #     final_report = response
    # 只返回新消息和最终报告
    return {
        "final_report": response,  
        "visualization_mapping": mapping_data,
        "messages": [report_complete_message]
    }

def normal_chat_node(state: CausalChatState,llm: ChatOpenAI) -> dict:
    """
    Represents "正常问答".
    This is for when the agent determines it's a simple chat conversation.
    """
    logging.info(" 步骤: 普通问答节点 ")
    
    prompt_template = (
        """
        system role: 你是日常聊天助手，你的任务是根据用户的对话历史，回答用户的问题。
        
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    response = runnable.invoke({
        "messages": state["messages"],
    })
    # 只返回新消息
    return {"messages": [AIMessage(content=response, name="normal_chat")]}

def inquiry_answer_node(state: CausalChatState, llm: ChatOpenAI) -> dict:
    """
    根据报告追问用户的问题
    """
    logging.info(" 步骤: 根据报告回答用户的问题节点 ")
    prompt_template = (
        """
        system role: {system_role}
        # 当前状态摘要
        1. 因果分析结果：{causal_analysis_result}
        2. 知识库结果：{knowledge_base_result}
        3. 后处理结果：{postprocess_result}
        4. 报告：{final_report}
        
        # 你的任务：根据历史摘要和所有分析结果，回答用户问题
        - 用户的问题：{messages}
        
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    response = runnable.invoke({
        "messages": state["messages"],
        "causal_analysis_result": state.get("causal_analysis_result", {}),
        "knowledge_base_result": state.get("knowledge_base_result", {}),
        "postprocess_result": state.get("postprocess_result", {}),
        "final_report": state.get("final_report", {}),
        "system_role": causal_prompt()
    })
    # 只返回新消息
    return {"messages": [AIMessage(content=response, name="inquiry_answer")]}

