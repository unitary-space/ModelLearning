from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=80,
    chunk_overlap=20,
    paragraph_separator= "\n\n",
    separator=". "
)

documents = SimpleDirectoryReader(input_files=[r'./data/小说.txt']).load_data()

nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(node.text, "\n", "---"*10)
