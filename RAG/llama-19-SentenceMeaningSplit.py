from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

documents = SimpleDirectoryReader(input_files=[r"./data/小说.txt"]).load_data()

embed_model = HuggingFaceEmbedding(r'./models/BAAI/bge-large-zh-v1___5')
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)

nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(node.text, node.metadata, "---------")