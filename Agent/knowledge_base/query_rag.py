import os
import json
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from pandas.core.dtypes.dtypes import str_type
from typing import List
from config.settings import settings

base_dir = os.path.dirname(os.path.abspath(__file__))
# 返回到项目根目录
project_root = os.path.dirname(base_dir) 
MODEL_PATH = os.path.join(base_dir, "models", "bge-small-zh-v1.5")
PERSIST_DIRECTORY = os.path.join(base_dir, "db")


llm = ChatOpenAI(
    api_key=settings.API_KEY,
    base_url=settings.BASE_URL,
    model_name=settings.MODEL, 
)

# 2. 加载与构建时相同的本地Embedding模型
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_function = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 3. 加载已经持久化存储的向量数据库
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_function
)

# 4. 从数据库创建一个检索器 (Retriever)
# 检索器是专门用于根据查询查找相关文档的对象
retriever = db.as_retriever(search_kwargs={"k": 3}) # 设置为返回最相关的3个结果


# 5. 定义我们的Prompt模板
# 这个模板指导LLM如何利用我们提供的上下文来回答问题
template = """
请只根据以下提供的上下文信息来回答问题。
如果根据上下文信息无法回答问题，请直接说"根据提供的知识库，我无法回答该问题"，不要自行编造答案。

上下文:
{context}

问题:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. 使用LCEL构建RAG链
# 这是一个非常优雅和强大的方式来组合不同的组件
rag_chain = (
    # RunnablePassthrough()会将问题同时传给retriever和prompt
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser() # 将LLM的输出解析为字符串
)

def get_rag_response(questions: List[str]) -> str:
    """
    接收一个问题字符串列表，对每个问题使用RAG链进行查询，
    并将所有回答聚合成一个格式化的字符串。
    
    Args:
        questions: 用户或系统提出的问题列表。
        
    Returns:
        一个包含所有问题和回答的、格式化后的字符串。
    """
    if not questions:
        return "没有生成任何需要查询知识库的问题。"
        
    response_parts = []
    # 遍历问题列表，为每个问题调用RAG链
    # 给出索引
    for i, q in enumerate(questions):
        response = rag_chain.invoke(q)
        # 将问答对格式化后存入列表
        response_parts.append(f"**知识库查询 {i+1}: {q}**\n- {response}")
    
    # 使用换行符将所有格式化后的问答对连接成一个单独的字符串
    return "\n\n".join(response_parts)

## 测试
def query(question: str):
    """
    使用RAG链查询并打印答案
    """
    
    print(f"\n用户问题: {question}")
    print("--- RAG响应 ---")
    # 为了兼容测试，我们将单个问题包装成列表
    response = get_rag_response([question])
    print(response)

if __name__ == "__main__":
    # 示例查询
    query("因果推断是什么？它和相关性有什么区别？")
    query("什么是因果推断定律？")
    query("Judea Pearl是谁？") # 这个问题知识库里没有，测试其"不知道"的能力
