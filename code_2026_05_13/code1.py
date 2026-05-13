from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator, List

model = ChatOpenAI(model="gpt-5.5", base_url="https://api.jiekou.ai/openai")
output_parser = StrOutputParser()

# 自定义生成器 -- 一句一句输出
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    buff = ""
    for chunk in input:
        buff += chunk
        while "。" in buff:
            stop_index = buff.index("。")
            yield [buff[:stop_index].strip()]
            buff = buff[stop_index + 1:]
    yield [buff.strip()]

chain = model | output_parser | split_into_list

for chunk in chain.stream("写一段关于爱情的歌词，五句话，每句话用句号隔开"):
    print(chunk, end="")
