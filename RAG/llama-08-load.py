from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from pathlib import Path

# documents = SimpleDirectoryReader(r"./data").load_data()
text_list = ["text1", "text2"]
documents = [Document(text=t, metadata={"filename": "文件名称", "category": "类别"}) for t in text_list]
print(documents)


def filename_fn(filename: str):
    return {
        "file_name": filename,
        "category": Path(filename).suffix
    }


documents = SimpleDirectoryReader("./data", file_metadata=filename_fn).load_data()
print(documents)
