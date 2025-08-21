from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("QWEN_KEY"),
                 base_url=os.getenv("QWEN_URL"),
                 model_name="qwen-plus")

# 直接提供问题，并调用llm
response = llm.invoke("特朗普可能连任吗？")
print(response)
print("=" * 50)
print(response.content)
