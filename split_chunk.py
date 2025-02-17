# 导入必要的包
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain.document_loaders import TextLoader
from tqdm import tqdm
import logging
import os
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载output目录下的所有文本文件
logger.info("开始加载文档...")
output_dir = Path("output")
source_docs = []
for file_path in output_dir.glob("*.txt"):
    try:
        loader = TextLoader(str(file_path), encoding='utf-8')
        source_docs.extend(loader.load())
        logger.info(f"已加载文件: {file_path.name}")
    except Exception as e:
        logger.error(f"加载文件 {file_path} 时出错: {str(e)}")

logger.info(f"共加载了 {len(source_docs)} 个文档")

# Initialize the text splitter
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split documents and remove duplicates
logger.info("开始分割文档...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

logger.info(f"已处理 {len(docs_processed)} 个唯一文档块")

# Initialize the embedding model
logger.info("正在初始化嵌入模型...")
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

# Create the vector database
logger.info("正在创建向量数据库...")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model
)

logger.info("向量数据库创建成功")

vectordb.save_local("vector_db")
logger.info("向量数据库保存成功")