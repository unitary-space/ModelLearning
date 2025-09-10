from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter


documents = SimpleDirectoryReader(input_dir=r"./data1", filename_as_id=True).load_data()
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(model_name=r'./models/BAAI/bge-large-zh-v1___5')
    ],
    docstore=SimpleDocumentStore()
)

nodes = pipeline.run(documents=documents)
print(f"Ingested {len(nodes)} Nodes")

pipeline.persist(r"./pipeline_storage")
with open("./data1/t4.txt", "w", encodeing='utf-8') as f:
    f.write("这是测试文档3")

documents1 = SimpleDirectoryReader(input_dir='./data1', filename_as_id=True).load_data()
pipeline1 = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(model_name=r'./models/BAAI/bge-large-zh-v1___5')
    ],
    docstore=SimpleDocumentStore()
)

nodes1 = pipeline1.run(documents=documents)
print(f"Ingested {len(nodes)} Nodes")