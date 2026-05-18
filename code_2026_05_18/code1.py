# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

# model = ChatOpenAI(model="gpt-5.5", base_url="https://api.jiekou.ai/openai")

# Messages = [
#     SystemMessage(content="你是一个专业的翻译，请将用户的英文输入翻译成中文"),
#     HumanMessage(content="hello, world")
# ]

# response = model.invoke(Messages)
# print(response.content)

from langchain_core.messages import filter_messages, SystemMessage, HumanMessage, AIMessage

messages = [ 
    SystemMessage("你是一个聊天助手", id="1"), 
    HumanMessage("示例输入", id="2"), 
    AIMessage("示例输出", id="3"), 
    HumanMessage("真实输入", id="4"), 
    AIMessage("真实输出", id="5"), 
]

# 按照类型进行筛选，选出HumanMessage
print(filter_messages(messages, include_types="human"))