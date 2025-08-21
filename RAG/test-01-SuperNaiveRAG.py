from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

'''
def load_pdf(ppath):
    loader = PyPDFLoader(ppath)
    doc = loader.load_and_split()
    return doc


def split(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len
    )
    _texts = text_splitter.create_documents([page.page_content for page in doc if page])
    text_contents = [doc.page_content for doc in _texts if doc]
    return text_contents


def embed(text_list, key, model):
    embeddings = DashScopeEmbeddings(dashscope_api_key=key, model=model)
    doc_emb = embeddings.embed_documents(text_list)
    return doc_emb

'''


def load_spl_emb_save_to_chroma(doc_path, dpath):
    loader = PyPDFLoader(doc_path)
    paragraphs = loader.load_and_split(RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len
    ))
    if os.path.exists(dpath) and os.listdir(dpath):
        print('目标文件夹已有向量库，正在加载：')
        db = Chroma(
            persist_directory=dpath,
            embedding_function=DashScopeEmbeddings(dashscope_api_key=os.getenv('QWEN_KEY'))
        )
    else:
        print('目标文件夹无向量库，正在创建（消耗tokens）：')
        db = Chroma.from_documents(paragraphs,
                                   DashScopeEmbeddings(dashscope_api_key=os.getenv('QWEN_KEY')),
                                   persist_directory=dpath)
    return db


if __name__ == '__main__':
    load_dotenv()

    pdf_path = r"./RAG_documents/example.pdf"
    db_path = r"./chroma_db./example"
    chroma_db = load_spl_emb_save_to_chroma(pdf_path, db_path)

    query = '清华'
    retriever = chroma_db.as_retriever()
    docs = retriever.invoke(query)
    print('开始按顺序打印：')
    for index, doc in enumerate(docs):
        print(f"第 {index + 1} 相关内容：\n{doc.page_content}\n---------------\n")
    # print(documents[0].page_content)
    # texts_list = split(documents)
    # print(texts_list[0].page_content)
    # key = os.getenv('QWEN_KEY')
    # url = os.getenv('QWEN_URL')
    # embeddings_list = embed(texts_list, key, model='text-embedding-v3')
    # print(embeddings_list[0][:4])
