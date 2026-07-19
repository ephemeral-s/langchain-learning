from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector, LengthBasedExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.example_selectors import NGramOverlapExampleSelector

examples = [
    {"input": "See Spot run.", "output": "看见Spot跑。"},
    {"input": "My dog barks.", "output": "我的狗叫。"},
    {"input": "Spot can run.", "output": "Spot可以跑。"},
]

# 示例模板
example_prompt = PromptTemplate.from_template("Input:{input}\nOutput:{output}")

# 示例选择器（NGram）（注意使用时要pip install nltk）
example_selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=0.5,  # 阈值.
                    # 负值代表不相干的示例也被筛选出来
                    # 0.0，输出结果是只与输入重叠的示例
                    # 大于等于1.0，排除所有示例，返回空列表
)

# 将示例转换为消息列表
few_short_template = FewShotPromptTemplate(
    example_prompt = example_prompt,
    prefix = "给出每个输入的中文翻译", # 在最开头添加字符串
    suffix = "Input:{adjective}\nOutput:",
    input_variables = ["adjective"], # 添加了一个提示词模板，需要注明
    example_selector = example_selector # 指定示例选择器
)

print(few_short_template.invoke({"adjective" : "See Spot run fast."}).to_string())


# # 反义词示例集合
# examples = [
#     {"input": "happy", "output": "sad"},
#     {"input": "tall", "output": "short"},
#     {"input": "energetic", "output": "lethargic"},
#     {"input": "sunny", "output": "gloomy"},
#     {"input": "windy", "output": "calm"},
# ]


# # 根据长度进行选择
# example_selector = LengthBasedExampleSelector(
#     examples = examples,
#     example_prompt = example_prompt,
#     max_length=25 # 指定长度，表示格式化示例的最大长度，当前示例集合的长度是20
# )

# # 根据语义相似性选择
# example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
#     examples = examples,
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-large", 
#         base_url="https://api.highwayapi.ai/openai"),              # 定义嵌入模型
#     vectorstore_cls = Chroma,                                      # 指定向量数据库
#     k = 3                                                          # 生成示例的数量 
# )


