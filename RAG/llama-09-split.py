from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()

parser = SentenceSplitter()

nodes = parser.get_nodes_from_documents(documents)
print(nodes)