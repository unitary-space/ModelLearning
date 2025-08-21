from openai import OpenAI
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    load_dotenv()
    client = OpenAI(api_key=os.getenv("QWEN_KEY"), base_url=os.getenv("QWEN_URL"))
