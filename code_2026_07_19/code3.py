# from langchain_core.documents import Document

# # 手动定义的文档列表
# documents = [

#     # 对于单个Document文档，它一般表示较大的文档的某个块或者某一页
#     Document(
#         # 内容
#         page_content="狗是忠实的伴侣",
#         # 元数据字典
#         # 元数据属性可以包含：文档源，与其他文档的关系以及其他属性信息
#         metadata={"source": "pets-doc"},
#     ),
#     Document(
#         # 内容
#         page_content="猫是独立的宠物",
#         # 元数据字典
#         # 元数据属性可以包含：文档源，与其他文档的关系以及其他属性信息
#         metadata={"source": "pets-doc"},
#     ),
# ]

from langchain_community.document_loaders import PyPDFLoader

# 定义文档加载器
loader = PyPDFLoader(file_path="../test.pdf")

# 进行加载
docs = loader.load()

print(f"文档页数：{len(docs)}")
print(f"第一页文本的内容：{docs[0].page_content}")
print(f"第一页元数据字典：{docs[0].metadata}")