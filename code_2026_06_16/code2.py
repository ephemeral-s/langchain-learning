from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

client = Client()
prompt = client.pull_prompt("hardkothari/prompt-maker")

model = ChatOpenAI(model="gpt-5.5", base_url="https://api.jiekou.ai/openai")

chain = prompt | model

chain.invoke({"task" : "总结Python项目", "lazy_prompt" : "你是一个Python专家，你需要对Python项目进行梳理总结"}).pretty_print()