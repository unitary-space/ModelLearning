import asyncio
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-max-2025-01-25"
api_key = os.getenv("QWEN_KEY")
api_base_url = os.getenv("QWEN_URL")

def multiply(a: float, b: float) -> float:
    """
    简单计算器
    Args:
        a: 第一个数字
        b: 第二个数字
        operation: 运算符（+, -, *, /）
    Returns:
        计算结果
    :param a:
    :param b:
    :return:
    """
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
tools = [multiply_tool]
agent = FunctionAgent(
    tools=tools,
    llm=DashScope(model_name=model, api_key=api_key),
    verbose=True
)

async def main():
    response = await agent.run(user_msg="2乘以2等于多少？")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())