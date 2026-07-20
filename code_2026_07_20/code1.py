from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", base_url="https://api.jiekou.ai/openai")

# 内存向量存储器
vector_store = InMemoryVectorStore(embedding=embeddings)

# 获取文档列表
loader = UnstructuredMarkdownLoader("../test.md",)
# Document 列表
data = loader.load()

# tiktoken 分词器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=400,
    chunk_overlap=50,
)

# 形成文档列表
docs = text_splitter.split_documents(data)

# 存储文档到内存向量存储器中
# add_documents: 将要存储的文档列表进行编排索引。
ids = vector_store.add_documents(docs)
print(f"共有{len(docs)}个文档，编排了{len(ids)}个索引")
print(f"前三个文档的索引：{ids[:3]}")

# 根据索引获取前两个文档的内容
doc_2 = vector_store.get_by_ids(ids[:2])
print(doc_2)

# # 删除前两个文档
# vector_store.delete(ids=ids[:2])

# # 根据相似度进行检索
# # similarity_search: 根据余弦相似度来捕捉语义
# search_docs = vector_store.similarity_search(query="冲突", k=2) # 找出相关性最大的两个文档
# for doc in search_docs:
#     print("*" * 30)
#     print(doc)

# 限定范围进行检索
def _filter_function(doc: Document) -> bool:
    return doc.metadata.get("source") == "../xxx.md" # 限定原文档是xxx.md（并不存在）

search_docs = vector_store.similarity_search(
    query="冲突",
    k=2,
    filter=_filter_function   # filter 接收一个bool值
)

for doc in search_docs:
    print("*" * 30)
    print(doc)