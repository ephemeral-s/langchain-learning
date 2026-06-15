from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# # 方式1
# # prompt_template = PromptTemplate(
# #     template = "介绍{city}的历史",
# #     input_variables = ["city"]
# # )

# #方式2
# prompt_template = PromptTemplate.from_template("将文本从{language_from}翻译为{language_to}")

# # 实例化
# prompt_template.invoke({"language_from" : "英文", "language_to" : "中文"})

# 定义聊天消息提示词模板
chat_prompt_template = ChatPromptTemplate(
    [
        ("system", "将文本从{language_from}翻译为{language_to}"),
        ("user", "{text}")
    ]
)

# 实例化
message = chat_prompt_template.invoke(
    {"language_from" : "英文", 
     "language_to" : "中文", 
     "text" : "hello, what's your name?"
    }
)
