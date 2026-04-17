from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 定义模型
model = ChatOpenAI(
    model="glm-5.1",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4"
)

# 定义消息
# 用户消息 -- HumanMessage
# 系统消息 -- SystemMessage
message = [
    HumanMessage(content="你是什么模型")
]

# 调用大模型
# result = model.invoke(message)
# print(result)

# 定义输出解析器
parser = StrOutputParser()
# print(parser.invoke(result))

# 定义链
chain = model | parser

#执行链
print(chain.invoke(message))