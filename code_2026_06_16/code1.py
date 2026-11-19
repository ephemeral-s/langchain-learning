from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatOpenAI(model="gpt-5.5", base_url="https://api.jiekou.ai/openai")

# 定义聊天消息提示词模板
chat_prompt_template = ChatPromptTemplate(
    [
        ("system", "将文本从{language_from}翻译为{language_to}"),
        MessagesPlaceholder("msgs"),
        ("user", "{text}")
    ]
)

placeholder = [
    HumanMessage(content="hello, what's your name?"),
    AIMessage(content="你好，你叫什么名字？")
]

chain = chat_prompt_template | model

print(chain.invoke(
        {
            "language_from" : "英文", 
            "language_to" : "中文", 
            "text" : "what's your age?",
            "msgs" : placeholder
        }
    ).content
)
