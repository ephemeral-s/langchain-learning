# 人物信息提取器，将用户描述的信息转化为格式化输出

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Optional

# 定义一个输出结构
class CharacterInfo(BaseModel):
    """人物信息"""
    name: Optional[str] = Field(default=None, description="人物的姓名")
    age: Optional[int] = Field(default=None, description="人物的年龄")
    height_with_meters: Optional[float] = Field(default=None, description="人物的高度（单位：米）")
    skin_color: Optional[str] = Field(default=None, description="人物的皮肤颜色")

# 定义模型
model = ChatOpenAI(model="gpt-5.5", temperature=0.5, base_url="https://api.jiekou.ai/openai")

# 绑定结构
model_with_struct = model.with_structured_output(CharacterInfo)

messages = [
    {"role": "system", "content": "你是一个人物信息提取器，你的任务是将用户描述的信息转化为格式化输出，如果没有描述完整，就返回null"},
    {"role": "user", "content": "一个18岁的学生，他的身高是1.75米，皮肤颜色是黄色"}
]

# 调用模型
character_info = model_with_struct.invoke(messages)
print(character_info)