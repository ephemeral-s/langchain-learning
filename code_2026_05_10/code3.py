from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Union

# 定义输出结构
class Joke(BaseModel):
    """给用户讲一个笑话"""

    setup: str = Field(description="笑话的开头")
    punchline: str = Field(description="笑话的妙语")

class ConversationResponse(BaseModel):
    """正常对话时模型的回复"""

    content: str = Field(description="正常对话时模型的回复")

class final_Response(BaseModel):
    final: Union[Joke, ConversationResponse]

# 初始化OpenAI Chat模型
model = ChatOpenAI(model="gpt-5.5", temperature=0.5, base_url="https://api.jiekou.ai/openai")

# 绑定结构
model_with_struct = model.with_structured_output(final_Response)

# 调用模型
print(model_with_struct.invoke("讲一个笑话"))
print(model_with_struct.invoke("请介绍一下你自己"))
