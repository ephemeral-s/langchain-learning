from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_text_splitters import CharacterTextSplitter
from redisvl.query.filter import Tag, Num

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


loader = UnstructuredMarkdownLoader("../test.md",)
data = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=400,
    chunk_overlap=50,
)

# 文档列表
docs = text_splitter.split_documents(data)
for i, doc in enumerate(docs, start=1):
    doc.metadata["category"] = "QA"
    doc.metadata["num"] = i

# # 添加文档（编制索引）
# ids = vector_store.add_documents(documents=docs)
# print(f"编制了{len(ids)}个索引")
# print(f"前三个索引是：{ids[:3]}")

# # 查
# print(vector_store.get_by_ids(["01K85F9HDK4H98QK221YA2ZQ7E"]))

# # 删除
# vector_store.delete(["01K85F9HDK4H98QK221YA2ZQ7E"])
# print(vector_store.get_by_ids(["01K85F9HDK4H98QK221YA2ZQ7E"]))

# # 全量删除(连带索引结构全部删除)
# vector_store.index.delete(drop=True)

# # 检索
# search_docs = vector_store.similarity_search(query="冲突", k=2)
# # 检索并给结果打分: 分数越低表示相似度越高
# search_docs_results = vector_store.similarity_search_with_score(query="冲突", k=2)

# # for doc in search_docs:
# #     print("*" * 30)
# #     print(f"文档内容：{doc.page_content}")
# #     print(f"文档元数据：{doc.metadata}")

# for doc, score in search_docs_results:
#     print("*" * 30)
#     print(f"打分：{score}")
#     print(f"文档内容：{doc.page_content}")
#     print(f"文档元数据：{doc.metadata}")

# # 过滤类别为QA并且序号大于6的文档
# _filter_function = (Tag("category") == "QA") & (Num("num") > 6)

# search_docs_results = vector_store.similarity_search_with_score(
#     query="冲突", 
#     k=2,
#     filter=_filter_function    
# )

# for doc, score in search_docs_results:
#     print("*" * 30)
#     print(f"打分：{score}")
#     print(f"文档内容：{doc.page_content}")
#     print(f"文档元数据：{doc.metadata}")

# 过滤类别为QA并且序号大于6的文档
_filter_function = (Tag("category") == "QA") & (Num("num") > 6)

# MMR检索
results = vector_store.max_marginal_relevance_search(
    query="冲突",
    k=2,
    filter=_filter_function,
    fetch_k=10,
) # 先筛选十个最相似的，然后按照MMR选出两个返回

for doc in results:
    print("*" * 30)
    print(f"文档内容：{doc.page_content}")
    print(f"文档元数据：{doc.metadata}")
