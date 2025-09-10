from langchain.chains.qa_with_sources.map_reduce_prompt import question_prompt_template
from  llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core.ingestion import IngestionPipeline
from dotenv import load_dotenv
import os

from RAG.lamma_01 import api_base_url, documents

load_dotenv()
model = "qwen-turbo"
api_key = os.getenv("QWEN_KEY")
api_base_url = os.getenv("QWEN_URL")

Settings.embed_model = HuggingFaceEmbedding(r'./models/BAAI/bge-large-zh-v1___5')
documents = SimpleDirectoryReader("data").load_data()

text_splitter = TokenTextSplitter(separator='. ', chunk_size=512, chunk_overlap=128)
title_extractor = TitleExtractor(nodes=5, node_template="请为以下文档生成一个简洁的标题：{context_str}")

question_prompt_template = """
以下是参考内容：
{context_str}

请根据上述上下文信息，生成 {num_questions} 个该内容能够具体回答的问题，这些问题的答案最好是该内容独有的，不容易在其他地方找到。

你也可以参考上下文中可能提供的更高层次的总结信息，结合这些总结，尽可能生成更优质、更具有针对性的问题。
"""

qa_extractor = QuestionsAnsweredExtractor(questions=3, prompt_template=question_prompt_template)

pipeline = IngestionPipeline(
    transformations=[text_splitter, title_extractor, qa_extractor]
)
nodes = pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True
)
print(nodes)