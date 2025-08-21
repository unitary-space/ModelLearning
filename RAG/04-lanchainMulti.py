import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,  # 等价于OpenAI接口中的assistant role AI 模型的回复消息
    HumanMessage,  # 等价于OpenAI接口中的user role  表示用户输入的消息
    SystemMessage  # 等价于OpenAI接口中的system role  系统级指令或背景设定
)

load_dotenv()

llm = ChatOpenAI(model_name='qwen-max',
                 api_key=os.getenv("QWEN_KEY"),
                 base_url=os.getenv("QWEN_URL")  # 默认是gpt-3.5-turbo
)

messages = [
    SystemMessage(content="你是各位老师的个人助理。你叫小戈"),
    HumanMessage(content="我的名字叫小张"),
    AIMessage(content="你好"),
    HumanMessage(content="你是谁？")
    # HumanMessage(content="今天天气怎么样")
]

messages.append(AIMessage(content=llm.invoke(messages).content))
print(messages[-1].content)