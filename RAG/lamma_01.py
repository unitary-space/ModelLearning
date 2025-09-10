from llama_index.llms.dashscope import DashScope
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-plus-1125"
api_key = os.getenv("QWEN_KEY")
api_base_url = os.getenv("QWEN_URL")

# llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True, max_tokens=1000)
#
# response = llm.complete("帮我推荐一下上海市5天旅游攻略")
# print(response)

Settings.llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True, max_tokens=1000)
Settings.embed_model = HuggingFaceEmbedding(r'./models/BAAI/bge-large-zh-v1___5')

documents = SimpleDirectoryReader("data").load_data() # 自动识别文件格式
index = VectorStoreIndex.from_documents(documents)

# 启动查询引擎
query_engine = index.as_query_engine()
response = query_engine.query("企业事件？")
print(response)
