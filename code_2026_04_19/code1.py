# from langchain_core.tools import StructuredTool
# from typing import Tuple, List

# def add(a: int, b: int) -> Tuple[str, List[int]]:
#     nums = [a, b]
#     content = f"两数相加结果为{a + b}"
#     return content, nums

# add_tool = StructuredTool.from_function(
#     func=add, 
#     name="add", 
#     description="Adds two numbers together",
#     response_format="content_and_artifact" # 返回内容和额外的元数据
# )

# # 模拟大模型调用工具
# print(add_tool.invoke({
#     "name": "add",
#     "args": {"a": 1, "b": 2},
#     "type": "tool_call", # 工具调用类型，必填
#     "id": "1234567890" # 工具调用ID，将结果与调用请求绑定，必填
#    }))

from langchain_core.tools import tool
from langchain_glm import ChatZhipuAI
from langchain_core.messages import HumanMessage

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together
    Args:
        a (int): The first number
        b (int): The second number
    Returns:
        int: The sum of the two numbers
    """
    return a + b

@tool
def mul(a: int, b: int) -> int:
    """Multiply two numbers together
    Args:
        a (int): The first number
        b (int): The second number
    Returns:
        int: The product of the two numbers
    """
    return a * b

model = ChatZhipuAI(model="glm-5.1")

# 绑定工具
tools = [add, mul]
modelWithTools = model.bind_tools(tools) # 接收绑定好工具的新模型
# modelWithTools = model.bind_tools_with(tools=tools, tool_choice="any") # 强制模型选择工具

# # 调用工具
# print(modelWithTools.invoke("计算1加2的和")) # 此时模型会返回工具的选择，但不调用
# print(modelWithTools.invoke("计算1乘2的积")) # 此时模型会返回工具的选择，但不调用
# print(modelWithTools.invoke("你是谁")) # 询问其他问题，不一定会选择工具

# # 调用工具
# msg = modelWithTools.invoke("计算1加2的和")
# print(add.invoke(msg.tool_calls[0])) # 真正调用工具，返回结果

# 将问题、工具选择结果、模型调用结果添加到消息列表中
message = [
    HumanMessage(content="2乘3等于多少？6加2等于多少？") # 问题
]

ai_msg = modelWithTools.invoke(message)
message.append(ai_msg) # 工具选择结果

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add" : add, "mul" : mul}[tool_call["name"].lower()] # 查找对应的工具
    result = selected_tool.invoke(tool_call) # 调用工具
    message.append(result) # 模型调用结果

print(model.invoke(message))

