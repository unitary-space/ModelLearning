from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.json import JSONReader


reader = JSONReader()
documents = reader.load_data(input_file='./data/request.json')

embed_model = HuggingFaceEmbedding(r'./models/BAAI/bge-large-zh-v1___5')
text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

pipeline = IngestionPipeline(
    transformations=[text_splitter, embed_model]
)

nodes = pipeline.run(documents=documents)
for node in nodes:
    print(node, "----------------")
