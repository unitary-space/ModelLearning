from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document

document = Document(text="这是第一个句子. 这是第二个句子. 这是第三个句子. 这是第四个句子. ")

node_parser = SentenceWindowNodeParser(
    window_size=1,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

nodes = node_parser.get_nodes_from_documents([document])

for node in nodes:
    print(node.text, node.metadata, "\n\n")