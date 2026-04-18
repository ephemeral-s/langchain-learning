# from langchain_glm import ChatZhipuAI
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.output_parsers import StrOutputParser

# model = ChatZhipuAI(
#     model="glm-5.1", # 模型名称
#     temperature=0.5, # 温度参数，温度越高，模型的输出越随机，温度越低，模型的输出越确定性
#     max_tokens=1024, # 最大输出token数
#     timeout=60, # 请求超时时间，单位为秒
#     max_retries=3, # 最大重试次数
#     # api_key 令牌
#     # base_url 模型的调用1
#     # organization 组织ID
# )

# message = [
#     SystemMessage(content="你是一个专业的翻译，将用户的问题翻译成中文"),
#     HumanMessage(content="hello, i am a student"),
# ]

# parser = StrOutputParser()

# chain = model | parser
# print(chain.invoke(message))

# from langchain.chat_models import init_chat_model
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# model = init_chat_model(
#     model="gpt_4o", 
#     model_provider="openai",
#     max_tokens=1024,
#     configurable_fields=("max_tokens",), # 定义可修改配置的字段
#     config_prefix="first_" # 配置前缀
# )

# message = [
#     SystemMessage(content="你是一个专业的翻译，将用户的问题翻译成中文"),
#     HumanMessage(content="hello, i am a student"),
# ]

# result = model.invoke(
#     message,
#     config={"first_max_tokens": 1024} # 修改配置
# )

# print(result)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model="gemma4:e2b",
    base_url="http://localhost:11434"
)

message = [
    SystemMessage(content="你是一个专业的翻译，将用户的问题翻译成中文"),
    HumanMessage(content="hello, i am a student"),
]

parser = StrOutputParser()

chain = model | parser
print(chain.invoke(message))
