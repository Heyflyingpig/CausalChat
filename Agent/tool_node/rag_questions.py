from langgraph.func import task
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import logging
import json
from Agent.causal_agent.state import CausalChatState
from Agent.causal_agent.back_prompt import causal_rag_prompt
from langchain_openai import ChatOpenAI


class RagQuestion(BaseModel):
    """用于生成知识库查询问题的模型。"""
    questions: List[str] = Field(
        default_factory=list,
        description="根据对话历史和数据摘要，为知识库生成一个或多个精确、具体的问题列表。"
    )

@task
def get_rag_questions(
    state: CausalChatState,
    llm: ChatOpenAI,
    num_questions: int
) -> List[str]:
    
    logging.info("正在启动 RAG 知识库查询...")
    try:
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """
                system role: {system_role}
            
            你是一个因果数据分析领域的专家，你的任务是根据用户的对话历史和当前的数据摘要，识别出其中需要通过知识库进行澄清的关键概念或潜在问题，提出{num_questions}个问题

            # 数据摘要:
            {data_summary}

            # 数据预处理总结:
            {preprocess_summary}

            # 你的任务:
            综合以上信息，生成一个包含多个个问题的JSON列表，并赋值给 'questions' 字段。这些问题应该简洁、明确，旨在从知识库中检索信息，以帮助用户更好地理解当前分析的背景、方法论或潜在风险。

            **你必须严格按照 `RagQuestion` 的 schema 返回一个 JSON 对象。**
            **绝对不要在你的回复中包含任何Markdown格式或解释性文字。**

            示例输出:
            {{
                "questions": ["什么是混杂因子，以及如何在因果分析中控制它？", "数据缺失在因果分析中会引入哪些类型的偏倚？", "在处理时间序列数据时，PC算法有哪些局限性？"]
            }}
            """),
            ("human", "请根据上述指示和提供的数据摘要，生成问题列表。，注意请只生成json对象，不要包含任何其他文字。")
        ])
        
        question_generator_runnable = rag_prompt | llm | JsonOutputParser()
        
        logging.info("正在调用LLM生成RAG查询问题...")
        
        llm_output = question_generator_runnable.invoke({
            "messages": state["messages"],
            "data_summary": json.dumps(state.get("analysis_parameters", {}), indent=2, ensure_ascii=False),
            "preprocess_summary": state.get("preprocess_summary", ""),
            "system_role": causal_rag_prompt(),
            "num_questions": num_questions
        })

        try:
            response = RagQuestion.model_validate(llm_output)

        except Exception as e:
            logging.error(f"Could not parse JSON from LLM response: {e}\nRaw response: {llm_output}")
            return ["无法生成RAG问题"]
        
        logging.info(f"LLM生成的RAG问题列表: {response.questions}")
        
        logging.info("RAG 知识库查询成功。")
        return response.questions
    except Exception as e:
        logging.error(f"执行 RAG 查询时发生错误: {e}", exc_info=True)
        return ["无法生成RAG问题"]
