from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

model = ChatOpenAI(model="gpt-5.5", base_url="https://api.jiekou.ai/openai")

# Pydantic 对象
class Joke(BaseModel):
    """给用户讲的一个笑话"""

    setup: str = Field(description="这个笑话的开头")
    punchline: str = Field(description="这个笑话的妙语")
    rating: Optional[int] = Field(default=None, description="从1-10分，给这个笑话评分")

# # 定义输出解析器
# parser = PydanticOutputParser(pydantic_object=Joke)

# json格式
parser = JsonOutputParser(pydantic_object=Joke)

# 提示模板
prompt = PromptTemplate(
    template="回复用户问题。\n返回结构说明：{format_instructions}\n用户问题：{query}\n",
    partial_variables={"format_instructions": parser.get_format_instructions()},  # 将parser生成的提示词设置为模板参数（提前设置）
    input_variables=["query"],
)

chain = prompt | model | parser

print(chain.invoke({"query": "讲一个笑话"}))