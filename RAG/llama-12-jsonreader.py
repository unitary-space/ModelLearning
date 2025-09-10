from llama_index.readers.json import JSONReader
from llama_index.core.node_parser import  JSONNodeParser, SentenceSplitter

reader = JSONReader()
documents = reader.load_data(input_file=r"./data/request.json")
print(documents)
print(JSONNodeParser().get_nodes_from_documents((documents)))
s = SentenceSplitter(chunk_size=10, chunk_overlap=5)
print(s.get_nodes_from_documents(documents))

parser