from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
import chromadb
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os


load_dotenv()
model = "qwen-plus-1125"
api_key = os.getenv("QWEN_KEY")
api_base_url = os.getenv("QWEN_URL")

embed_model = HuggingFaceEmbedding(r'./models/BAAI/bge-large-zh-v1___5')
# Settings.llm = DashScope(model_name=model, api_key=api_key, api_base_url=api_base_url)
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(input_files=[r"./data/小说.txt"]).load_data()

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

pipeline = IngestionPipeline(
    transformations=[text_splitter, embed_model],
    vector_store=vector_store
)

nodes = pipeline.run(documents=documents)

index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever()
print(retriever.retrieve("萧熏儿的斗气是多少？"))