from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=[r'data/小说.txt']).load_data()

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
nodes = node_parser.get_nodes_from_documents(documents)
for node in nodes:
    print(f"ID: {node.node_id}, Text:{node.text}...")
    if node.parent_node:
        print(f"parent: {node.parent_node.node_id}")
