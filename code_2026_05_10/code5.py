from langchain_openai import ChatOpenAI
import asyncio

model=ChatOpenAI(model="gpt-5.5", temperature=0.5, base_url="https://api.jiekou.ai/openai")

# 创建协程，用于异步流式输出
async def async_stream():
    async for chunk in model.astream("请介绍一下你自己"):
        print(chunk.content, end="")

asyncio.run(async_stream()) # 异步调用协程，输出流式结果


# chunks = []
# for chunk in model.stream("讲一个故事"):
#     chunks.append(chunk.content)
#     print(chunk.content, end="")
