from pydantic import BaseModel, Field
from typing import List
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from Agent.causal_agent.state import CausalChatState
import json
import logging

class CycleFixDecision(BaseModel):
    """LLM决策删除哪条边来打破环路的结构化输出。"""
    remove_edge: List[str] = Field(
        ...,
        description="要删除的边，格式为[起点变量名, 终点变量名]，表示删除'起点->终点'这条边。"
    )
    reason: str = Field(
        ...,
        description="删除这条边的理由，基于领域知识和数据特征的分析。"
    )

def fix_cycles_with_llm(
        adjacency_matrix: np.ndarray, 
        cycles: List[List[str]], 
        node_names: List[str],
        llm: ChatOpenAI, 
        state: CausalChatState
    ) -> np.ndarray:
    """
    使用LLM辅助决策，通过删除某些边来打破环路。
    
    Args:
        adjacency_matrix: 原始邻接矩阵
        cycles: 检测到的所有环路
        node_names: 节点名称列表
        llm: LangChain的ChatOpenAI实例
        state: 当前状态，用于获取数据摘要和知识库结果
        
    Returns:
        修正后的邻接矩阵
        
    策略：
        - 对每个环路，调用LLM决策删除哪条边
        - LLM基于数据特征和领域知识做出判断
        - 逐个修正所有环路
    """
    revised_matrix = adjacency_matrix.copy()
    
    # 获取上下文信息
    analysis_parameters = state.get("analysis_parameters", {})
    knowledge_base_result = state.get("knowledge_base_result", "无可用领域知识")
    
    # 简化知识库结果（截取前500字符）
    knowledge_excerpt = knowledge_base_result[:500] if isinstance(knowledge_base_result, str) else str(knowledge_base_result)[:500]
    
    for idx, cycle in enumerate(cycles):
        try:
            # 构建环路描述
            cycle_description = " -> ".join(cycle) + f" -> {cycle[0]}"
            
            logging.info(f"正在请求LLM修正环路 {idx+1}/{len(cycles)}: {cycle_description}")
            
            # 构建prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是因果推断领域的专家。检测到因果图中存在环路，这违反了因果关系的基本假设（因果关系必须是有向无环的）。

        你的任务是决定应该删除环路中的哪一条边，以打破这个环路。

        # 环路信息
        {cycle_description}

        # 参考信息
        数据特征摘要：
        {data_summary}

        相关领域知识：
        {knowledge_excerpt}

        # 决策原则
        1. 优先保留在数据中有强统计关联的边
        2. 考虑变量之间的时序关系（原因在前，结果在后）
        3. 参考领域知识中的常识判断
        4. 如果有target和treatment变量，优先保留与它们相关的边

        请仔细分析后，决定删除环路中的哪一条边。输出格式必须严格遵循CycleFixDecision模型。"""),
            ])
            
            runnable = prompt | llm.with_structured_output(CycleFixDecision)
            
            decision = runnable.invoke({
                "cycle_description": cycle_description,
                "data_summary": json.dumps(analysis_parameters, ensure_ascii=False, indent=2),
                "knowledge_excerpt": knowledge_excerpt
            })
            
            # 解析LLM的决策
            edge_to_remove = decision.remove_edge
            reason = decision.reason
            
            if len(edge_to_remove) != 2:
                logging.error(f"LLM返回的边格式错误: {edge_to_remove}")
                continue
            
            from_node, to_node = edge_to_remove
            
            # 找到节点索引
            if from_node not in node_names or to_node not in node_names:
                logging.error(f"LLM返回的节点名称不在节点列表中: {from_node} -> {to_node}")
                continue
            
            from_idx = node_names.index(from_node)
            to_idx = node_names.index(to_node)
            
            # 删除边：记住邻接矩阵的约定 matrix[i][j]=1 表示 j->i
            # 所以删除 from_node -> to_node 意味着设置 matrix[to_idx][from_idx] = 0
            if revised_matrix[to_idx][from_idx] != 0:
                revised_matrix[to_idx][from_idx] = 0
                logging.info(f"已删除边: {from_node} -> {to_node}")
                logging.info(f"删除理由: {reason}")
            else:
                logging.warning(f"边 {from_node} -> {to_node} 在矩阵中不存在，可能已被删除。")
                
        except Exception as e:
            logging.error(f"修正环路 {idx+1} 时发生错误: {e}", exc_info=True)
            continue
    
    return revised_matrix