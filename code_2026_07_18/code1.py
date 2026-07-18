from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector, SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 反义词示例集合
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

# 示例模板
example_prompt = PromptTemplate.from_template("Input:{input}\nOutput:{output}")

# # 根据长度进行选择
# example_selector = LengthBasedExampleSelector(
#     examples = examples,
#     example_prompt = example_prompt,
#     max_length=25 # 指定长度，表示格式化示例的最大长度，当前示例集合的长度是20
# )

# 根据语义相似性选择
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples = examples,
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", base_url="https://api.highwayapi.ai/openai"), # 定义嵌入模型
    vectorstore_cls = Chroma,                                      # 指定向量数据库
    k = 2 # 填几就表示筛选最相似的几个
)

# 将示例转换为消息列表
few_short_template = FewShotPromptTemplate(
    example_prompt = example_prompt,
    prefix = "给出每个示例的反义词：", # 在最开头添加字符串
    suffix = "Input:{adjective}\nOutput:",
    input_variables = ["adjective"], # 添加了一个提示词模板，需要注明
    example_selector = example_selector # 指定示例选择器
)

print(few_short_template.invoke({"adjective" : "big"}).to_string())

