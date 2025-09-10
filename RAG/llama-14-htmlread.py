from llama_index.core.node_parser import HTMLNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

html_docs = FlatReader().load_data(Path(r"data/index.html"))

parser = HTMLNodeParser(tags=["p","h1","li"])
nodes = parser.get_nodes_from_documents(html_docs)

print(nodes)
