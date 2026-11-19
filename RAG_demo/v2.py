from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, filter_messages
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.tools.retriever import create_retriever_tool
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# 准备工作
# 聊天模型与嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", base_url="https://api.jiekou.ai/openai")
model = init_chat_model("gpt-5.6-terra", base_url="https://api.jiekou.ai/openai")


# 加载文档列表
paths = [
    "../test_docu_1_python速成指南.md",
    "../test_docu_2_二叉搜索树.md",
    "../test_docu_3_Linux参数与环境变量.md",
    "../test_docu_4_Docker代理配置.md"
]
docs = [UnstructuredMarkdownLoader(path).load() for path in paths]
docs_list = [item for sublist in docs for item in sublist]
# from_tiktoken_encoder：使用 tiktoken 编码器来计算长度的文本分割器。
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 使用内存中向量存储和 OpenAI 嵌入
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)

# 使用 LangChain 的预构建 create_retriever_tool 创建检索器工具：
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 创建检索工具
retriever_tool = create_retriever_tool(
    retriever,
    "retriever_tool",
    "搜索并返回文档中的相关信息",
)

# 测试
# print(retriever_tool.invoke({"query": "Python 中的 None 与 C++ 中的哪个概念接近？"}))

# 1. 状态 -- MessagesState已有

# 2. 节点
# 节点1
def generate_query_or_respond(state: MessagesState):
    """调用模型以基于当前状态生成响应。
    给定问题，它将决定使用检索工具检索，或者简单的生成用户响应"""

    result = model.bind_tools([retriever_tool]).invoke(state["messages"])
    # result 是一个 AIMessage
    return {
        "messages": [result]
    }

# 工具节点2：帮助我们去执行工具，可将工具直接封装为节点
retriever_node = ToolNode([retriever_tool])

# 节点3
REWRITE_PROMPT = (
    "查看输入并尝试推断潜在的语义意图/含义。\n"
    "这是最初的问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "提出一个改进后的问题（仅问题内容）："
)
def rewrite_question(state: MessagesState):
    """重写原始用户问题"""

    # state messages 包含 [H, A, T]
    question = state["messages"][0]
    prompt = REWRITE_PROMPT.format(question=question)
    result = model.invoke([HumanMessage(content=prompt)])
    # 把修改后的问题，设置为用户消息，再进行历史消息更新
    return {
        "messages": [HumanMessage(content=result.content)]
    }

# 节点4
GENERATE_PROMPT = (
    "你是负责回答问题的助手。 "
    "使用以下检索到的上下文片段来回答问题。 "
    "如果你不知道答案，就说你不知道。 "
    "回答要简明扼要。\n"
    "Question: {question} \n"
    "Context: {context}"
)
def generate_answer(state: MessagesState):
    """生成答案"""
    # state messages 包含 [H, A, T]

    # 问题+检索结果
    question = state["messages"][0].content # 要根据用户的原始问题生成答案
    context = state["messages"][-1].content # 检索结果
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    return {
        "messages": [model.invoke([HumanMessage(content=prompt)])]
    }

# 3. 构图，添加节点和边
workflow = StateGraph(MessagesState)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retriever", retriever_node)
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,  # 自带函数，自动判断模型是否携带工具调用
    {
        "tools": "retrieve",
        "__end__": END,
    }
)