import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 使用os.path.join确保路径在Windows和Linux上都能正常工作
base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base_dir, "models", "bge-small-zh-v1.5")
SOURCE_DIRECTORY = os.path.join(base_dir, "source")
PERSIST_DIRECTORY = os.path.join(base_dir, "db")

# 我们将从本地加载模型，避免网络问题
print("正在加载本地Embedding模型...")
model_kwargs = {'device': 'cpu'} # 如果你有支持CUDA的NVIDIA显卡，可以改成 {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # 设置为True，返回归一化的向量，便于余弦相似度计算
embedding_function = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding模型加载完毕。")

def build():
    """
    构建向量知识库：加载文档 -> 切分 -> 向量化 -> 存储
    """
    print("开始构建向量知识库...")
    print(f"将从以下目录加载文档: {SOURCE_DIRECTORY}")

    # --- 修改：手动加载文档，绕过DirectoryLoader ---
    documents = []
    try:
        for root, _, files in os.walk(SOURCE_DIRECTORY):
            print(f"正在扫描文件夹: {root}")
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        doc = Document(page_content=text, metadata={"source": file_path})
                        documents.append(doc)
                        print(f"成功手动加载 TXT 文件: {file_path}")
                    except Exception as e:
                        print(f"加载 TXT 文件 {file_path} 时出错: {e}")

                elif file.endswith(".pdf"):
                    try:
                        # 使用LangChain的PyPDFLoader来处理PDF
                        # 它会为PDF的每一页创建一个Document对象
                        pdf_loader = PyMuPDFLoader(file_path)
                        # .load()返回一个Document列表
                        pdf_docs = pdf_loader.load()
                        documents.extend(pdf_docs) # 将PDF中的所有页面文档添加到主列表中，注意和append的区别，一个加元素，一个加列表
                        print(f"成功加载 PDF 文件: {file_path} (共 {len(pdf_docs)} 页)")
                    except Exception as e:
                        print(f"加载 PDF 文件 {file_path} 时出错: {e}")

    except Exception as e:
        print(f"扫描目录 {SOURCE_DIRECTORY} 时发生错误: {e}")
        return

    if not documents:
        print("警告: 未加载到任何文档。请检查SOURCE_DIRECTORY路径是否正确，以及其中是否包含.txt和.pdf文件。")
        return

    print(f"成功加载 {len(documents)} 篇文档。")

    # 2. 将加载的文档切分成小块 (Chunking)
    # 这对于后续的检索至关重要
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    print(f"文档已切分为 {len(split_docs)} 个小块。")

    # 3. 将切分后的文档块向量化并存入ChromaDB
    # LangChain的Chroma.from_documents会自动处理向量化和存储过程
    print("正在将文档存入向量数据库...")
    db = Chroma.from_documents(
        split_docs,
        embedding_function,
        persist_directory=PERSIST_DIRECTORY  # 指定数据库持久化存储的路径
    )
    # 确保数据被写入磁盘
    db.persist()
    print("知识库构建完成！")

if __name__ == "__main__":
    build()
