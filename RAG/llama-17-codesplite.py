from llama_index.core.node_parser import CodeSplitter
from llama_index.core import SimpleDirectoryReader

document = SimpleDirectoryReader(input_files=[r"./data/demo.py"]).load_data()

splitter = CodeSplitter(
    language="python",
    chunk_lines=50,
    chunk_lines_overlap=10,
    max_chars=300
)
nodes = splitter.get_nodes_from_documents(document)
for node in nodes:
    print(f"Type: {node.metadata}\nText: {node.text}\n{"="*50}")

