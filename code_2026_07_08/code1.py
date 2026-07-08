from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 定义样本
examples = [
    {"text" : "What's your name?", "output" : "你叫什么名字？"},
    {"text" : "My name is David.", "output" : "我叫戴维。"}
]

# 定义聊天消息模板
examples_prompt_template = ChatPromptTemplate(
    [
        ("user", "{text}"),
        ("ai", "{output}")
    ]
)

# 用少样本提示模板根据聊天消息模板，将样本转换为消息列表
few_shot_prompts = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=examples_prompt_template
)

# 最终消息模板
chat_prompt_template = ChatPromptTemplate(
    [
        ("system", "将文本从{language_from}翻译为{language_to}"),
        few_shot_prompts,
        ("user", "{text}")
    ]
)

# 实例化
print(chat_prompt_template.invoke({
    "language_from" : "英文",
    "language_to" : "中文",
    "text" : "How old are you?"
}))

