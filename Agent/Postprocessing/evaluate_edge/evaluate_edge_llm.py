from pydantic import BaseModel, Field
from typing import List, Tuple, Literal, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from Agent.causal_agent.state import CausalChatState
import json
import logging

from Agent.causal_agent.back_prompt import evaluate_edge_prompt

class EdgeEvaluation(BaseModel):
    """LLM对边的评估结果。"""

    decision: List[str] = Field(
        ...,
        description="一个合理的边列表，格式为[“起点变量名 --> 终点变量名”, “起点变量名 --> 终点变量名”, ...]"
    )
    reason: str = Field(
        ...,
        description="决策理由，基于数据特征和领域知识"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="对该决策的信心程度"
    )

def evaluate_edges_with_llm(
        critical_edges: List[Tuple[str, str]],
        state: CausalChatState,
        llm: ChatOpenAI
    ) -> Dict[Tuple[str, str], EdgeEvaluation]:
    f"""
    使用LLM评估关键边的合理性。
    
    Args:
        critical_edges: 需要评估的边列表
        state: 当前状态
        llm: LangChain的ChatOpenAI实例
        
    Returns:
        边修改/保留字典，包括数据，评估置信度，和理由
    """
    err_evaluation = {"decision": critical_edges, "reason": "", "confidence": "low"}
    
    if not critical_edges:
        logging.info("没有关键边需要评估")
        return err_evaluation
    
    analysis_parameters = state.get("analysis_parameters", "无可用数据摘要")
    knowledge_base_result = state.get("knowledge_base_result", "无可用领域知识")
    
    # 简化知识库结果
    knowledge_excerpt = knowledge_base_result[:500] if isinstance(knowledge_base_result, str) else str(knowledge_base_result)[:500]
    
    
    logging.info(f"开始LLM评估 {len(critical_edges)} 条关键边...")
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            system role: {system_role}

            # 边信息
            边：{critical_edges}

            # 参考信息
            数据特征摘要：
            {data_summary}

            相关领域知识：
            {knowledge_knowledge}

            # 修改决策
            一个合理的边列表，格式如下：
            [“起点变量名 --> 终点变量名”, “起点变量名 --> 终点变量名”, ...]

            # 修改原则
            1. 除非是极其不合理，否则倾向于保留原边
            2. 考虑时序关系和逻辑依赖,量化修正对最终因果图的影响
            3. 对每个修正决策提供充分的因果学理由
            4. 明确指出修正的理论依据（如违反时间顺序、生物学不可能等），参考领域常识和专业知识

            请给出你的修改决策。"""),
            ])
            
        runnable = prompt | llm.with_structured_output(EdgeEvaluation)
        
        evaluation = runnable.invoke({
            "final_edges": critical_edges,
            "data_summary": json.dumps(analysis_parameters, ensure_ascii=False, indent=2),
            "relevant_knowledge": knowledge_excerpt,
            "system_role": evaluate_edge_prompt()
        })
        
        logging.info(f"  修改后列表: {evaluation.decision}, 理由: {evaluation.reason[:50]}...")
        
    except Exception as e:
        logging.error(f"评估边 {critical_edges} 时发生错误: {e}", exc_info=True)
        # 默认保留
        return err_evaluation

    return evaluation
