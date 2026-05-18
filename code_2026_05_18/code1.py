# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

# model = ChatOpenAI(model="gpt-5.5", base_url="https://api.jiekou.ai/openai")

# Messages = [
#     SystemMessage(content="你是一个专业的翻译，请将用户的英文输入翻译成中文"),
#     HumanMessage(content="hello, world")
# ]

# response = model.invoke(Messages)
# print(response.content)

# from langchain_core.messages import filter_messages, SystemMessage, HumanMessage, AIMessage

# messages = [ 
#     SystemMessage("你是一个聊天助手", id="1"), 
#     HumanMessage("示例输入", id="2"), 
#     AIMessage("示例输出", id="3"), 
#     HumanMessage("真实输入", id="4"), 
#     AIMessage("真实输出", id="5"), 
# ]

# # 按照类型进行筛选，选出HumanMessage
# print(filter_messages(messages, include_types="human"))
# print(filter_messages(include_types="human").invoke(messages))

# # 按照ID进行筛选，排除ID为4的消息
# print(filter_messages(messages, exclude_ids=["4"]))

# # 复合筛选，选出类型为human和ai_message且ID不为4的消息
# print(filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["4"]))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, merge_message_runs

messages = [
    SystemMessage("你是一个聊天助手。"),
    SystemMessage("你总是以笑话回应。"),
    HumanMessage("为什么要使用 LangChain?"),
    HumanMessage("为什么要使用 LangGraph?"),
    AIMessage("因为当你试图让你的代码更有条理时，LangGraph 会让你感到“节点”是个好主意！"),
    AIMessage("不过别担⼼，它不会“分散”你的注意力！"),
    HumanMessage("选择LangChain还是LangGraph?"),
]

merged_messages = merge_message_runs(messages)
print(merged_messages)
