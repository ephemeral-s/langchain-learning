from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-5.5", base_url="https://api.jiekou.ai/openai")

output_parser = StrOutputParser()
chain = model | output_parser

for chunk in chain.stream("请介绍一下你自己"):
    print(chunk, end="")
