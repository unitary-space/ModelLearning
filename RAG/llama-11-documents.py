from llama_index.core import SimpleDirectoryReader


def main():
    reader = SimpleDirectoryReader(input_dir=r'./data', recursive=True, exclude=['wodima.txt']) # 读取子目录

    all_docs = []
    for docs in reader.iter_data():
        all_docs.extend(docs)
    print(all_docs)

if __name__ == '__main__':
    main()