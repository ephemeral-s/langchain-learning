from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

model = ChatOpenAI(
    model="glm-5.1",
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

# 定义搜索工具
search_tool = TavilySearch(
    max_results=4
)

# 绑定工具
model_with_tools = model.bind_tools([search_tool])

# 定义消息列表
message = [
    HumanMessage(content="北京今天天气怎么样")
]

ai_message = model_with_tools.invoke(message)
message.append(ai_message)

for tool_call in ai_message.tool_calls:
    result = search_tool.invoke(tool_call)
    message.append(result)

print(model_with_tools.invoke(message).content)