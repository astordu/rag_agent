from smolagents import ToolCallingAgent, OpenAIServerModel, tool, GradioUI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import os


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logger.info("正在初始化嵌入模型...")
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

# 加载本地存储的向量数据库
logger.info("正在加载向量数据库...")
vectordb = FAISS.load_local("vector_db", embeddings=embedding_model, allow_dangerous_deserialization=True)
logger.info("向量数据库加载成功")

model = OpenAIServerModel(
    # model_id="google/gemini-2.0-flash-001",
    model_id="google/gemini-2.0-flash-lite-preview-02-05:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  # 从环境变量中获取 API 密钥
)


def retriever(query:str)->str:
    """
    根据用户的查询，执行向量数据库的相似性搜索，并返回结果的字符串表示形式。

    Args:
        query: 要查询的字符串。此字符串将用于在向量数据库中进行相似性搜索。

    """
    logger.info(f"正在查询: {query}")
    results = vectordb.similarity_search(query, k=5)  # k 是返回的结果数量
    logger.info("查询完成")

    # 将结果组合成一个字符串
    combined_results = "\n\n".join([f"资料{i+1}: {result.page_content}" for i, result in enumerate(results)])
    return combined_results


question = "孙悟空有两个师傅,他们分别是谁?"

context = retriever(question)

naive_agent_prompt = f"""
根据下面的问题和支持文档，给出一个全面的答案。
只回答所问的问题，回答应该简洁且与问题相关。
如果你无法找到信息，就如实回答。

Question:
{question}

{context}
    """

logger.info(f"naive_agent_prompt完整提示词:\n {naive_agent_prompt}")



a = model(messages=[{"role": "user", "content": naive_agent_prompt}])
logger.info(a.content)
