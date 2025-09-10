from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

md_docs = FlatReader().load_data(Path(r"./data/小说.txt"))

parser = SimpleFileNodeParser()

nodes = parser.get_nodes_from_documents(md_docs)
print(nodes)