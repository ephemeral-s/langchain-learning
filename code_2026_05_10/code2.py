from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated

# # 定义一个输出结构
# class Joke(BaseModel):
#     setup: str = Field(description="笑话的开头")
#     punchline: str = Field(description="笑话的妙语")

# # 定义一个输出结构
# class Joke(TypedDict):
#     setup: Annotated[str, ..., "笑话的开头"]
#     punchline: Annotated[str, ..., "笑话的妙语"]

# 定义一个json输出结构
json_schema = {
    "title": "joke",
    "description": "。",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "这个笑话的开头"
        },
        "punchline": {
            "type": "string",
            "description": "这个笑话的妙语"
        }
    },
    "required": ["setup", "punchline"]
}

# 初始化OpenAI Chat模型
model = ChatOpenAI(model="gpt-5.5", temperature=0.5, base_url="https://api.jiekou.ai/openai")

# 绑定结构
model_with_struct = model.with_structured_output(json_schema)

# 调用模型
joke = model_with_struct.invoke("讲一个笑话")
print(joke)
