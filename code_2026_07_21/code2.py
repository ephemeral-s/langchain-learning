from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 创建Pinecone客户端
pc = Pinecone()
index_name = "qa"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,   # 维度
        metric="cosine",  # 度量方式，cosine余弦相似度
        spec=ServerlessSpec(
            cloud="aws",               # 亚马逊云
            region="us-east-1"         # 区域
        ),
    )

# 获取索引
index = pc.Index(index_name)
# 嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# 定义 pinecone 向量库
vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index,          # pinecone 向量库的索引
)