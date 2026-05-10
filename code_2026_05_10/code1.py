from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

# 初始化OpenAI模型
model = ChatOpenAI(model="deepseek-v4-flash", temperature=0.5, base_url="https://api.deepseek.com/v1",
                   extra_body={ "thinking": { "type": "disabled"} })

# 定义搜索工具
search_tool = TavilySearch(max_results=3)

# 绑定工具
model_with_tools = model.bind_tools([search_tool])

# 创建消息列表
messages = [
    HumanMessage(content="今天（2026-05-10）纳斯达克100指数是多少"),
]

# 调用模型，让模型选择工具
ai_msg = model_with_tools.invoke(messages)
messages.append(HumanMessage(ai_msg.content))

# 本地调用
for tool_call in ai_msg.tool_calls:
    result = search_tool.invoke(tool_call) # 解析工具调用参数，调用工具并返回结果
    messages.append(result)

# 最后调用模型，让模型总结结果
print(model.invoke(messages).content)
