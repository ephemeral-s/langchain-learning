# import redis

# redis_url = "redis://localhost:6379"

# # 定义Redis客户端
# redis_client = redis.from_url(redis_url)

# # Ping
# print(redis_client.ping())

from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore, RedisConfig

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
