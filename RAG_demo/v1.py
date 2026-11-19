from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 构建知识库

# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", base_url="https://api.jiekou.ai/openai")
# 定义聊天模型
model = ChatOpenAI(model="gpt-5.6-terra", base_url="https://api.jiekou.ai/openai")

# Redis构建知识库

# Redis 配置
config = RedisConfig(
    index_name="qa",  # 定义索引名
    redis_url="redis://127.0.0.1:6379",
    metadata_schema=[
        {"name": "category", "type": "tag"},   # 添加索引字段：分类
        {"name": "num", "type": "numeric"},    # 添加索引字段：编号
    ]
)

# 向量库
vector_store = RedisVectorStore(
    embeddings=embeddings,
    config=config,
)

loader = UnstructuredLoader("../test.md",)
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

# 2. 从知识库中检索，将检索结果 + 查询语句构建为提示词

# 检索器 -- 默认检索出四个doc
retriever = vector_store.as_retriever()

# 提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """你是负责回答问题的助手。必须使用一下检索到的上下文片段来回答问题。
            如果你不知道答案，就说不知道答案。最多回复三句话的结果，回答要简明扼要。
            Question:{question}\n
            Context:{context}\n
            Answer:"""
        )
    ]
)

# 将检索出来的文档转换成规范化文本，之后传递给提示词模板
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. 发给LLM

# 定义链
chain = (
    # 需要同时传递检索消息和用户问题
    # {检索器检索 -> 规范化文本, RunnablePassthrough() 透传数据，保留原始输入，防止原始问题丢失}
    { "context": retriever | format_docs, "question": RunnablePassthrough()} # 并行执行
    | prompt
    | model
    | StrOutputParser()
)

# 4. 打印结果（流式）

while True:
    # 获取用户输入
    question = input("\n请输入您的问题（输入'退出'或'quit'结束程序）: ").strip()

    # 检查是否退出
    if question.lower() in ["退出", "quit"]:
        print("程序已结束，再见！")
        break

    # 检查输入是否为空
    if not question:
        print("问题不能为空，请重新输入。")
        continue

    # 执行链，流式输出
    print("回答: ", end="", flush=True)
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print()  # 换行