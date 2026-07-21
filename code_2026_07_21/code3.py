from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.runnables import chain

# 嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", base_url="https://api.jiekou.ai/openai")

# Redis 配置
config = RedisConfig(
    index_name="qa",  # 定义索引名
    redis_url="redis://127.0.0.1:6379",
    metadata_schema=[
        {"name": "category", "type": "tag"},   # 添加索引字段：分类
        {"name": "num", "type": "numeric"},    # 添加索引字段：编号
    ]
)

# 初始化 Redis 向量存储实例（建立了索引结构）
vector_store = RedisVectorStore(
    embeddings=embeddings,
    config=config,
)

# # 创建检索器
# retriever = vector_store.as_retriever(search_kwargs={"k" : 2})
# search_result = retriever.invoke("冲突")

# 使用@chain，定义检索器函数，当作具有Runnable属性的"检索器"使用
@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query=query ,k=2)

search_result = retriever.invoke("冲突")
for doc in search_result:
    print("*" * 30)
    print(doc.page_content)