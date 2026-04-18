# 测试工具调用

# from langchain_core.tools import tool

# @tool
# def add(a: int, b: int) -> int:
#     """Add two numbers together.
    
#     Args:
#         a: The first number to add.
#         b: The second number to add.
    
#     Returns:
#         The sum of the two numbers.
#     """
#     return a + b

# print(add.invoke({"a": 1, "b": 2}))

# print(add.name)
# print(add.description)
# print(add.args)

# from langchain_core.tools import tool
# from pydantic import BaseModel, Field

# class AddInput(BaseModel):
#     """Add two numbers together."""
#     a: int = Field(..., description="The first number to add.")
#     b: int = Field(..., description="The second number to add.")

# @tool(args_schema=AddInput)
# def add(a: int, b: int) -> int:
#     return a + b

# print(add.invoke({"a": 1, "b": 2}))

# print(add.name)
# print(add.description)
# print(add.args)

from langchain_core.tools import tool
from typing_extensions import Annotated

@tool
def add(
    a: Annotated[int, "The first number to add."], 
    b: Annotated[int, "The second number to add."]
) -> int:
    """Add two numbers together."""
    return a + b


print(add.invoke({"a": 1, "b": 2}))

print(add.name)
print(add.description)
print(add.args)
