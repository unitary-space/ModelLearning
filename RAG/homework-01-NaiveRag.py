import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

def load_web_and_split(url):
    web_loader = WebBaseLoader('https://www.gov.cn')
    doc_list = web_loader.load_and_split(RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len
    ))
    return doc_list


def embed_and_retriever(dlist, key, emb_path):
    db = Chroma.from_documents(dlist,
    DashScopeEmbeddings(dashscope_api_key=key),
    persist_directory=emb_path)

    retriever = db.as_retriever()
    return retriever

if __name__ == '__main__':
    load_dotenv()
    doc_list = load_web_and_split("https://www.gov.cn")
    emb_ret = embed_and_retriever(doc_list, os.getenv("QWEN_KEY"), "./chroma_db/gov_net")

    llm = ChatOpenAI(api_key=os.getenv("QWEN_KEY"),
                    base_url=os.getenv("QWEN_URL"),
                    model="qwen-plus")

    template = """你是一个政府网站信息助手，请根据以下上下文回答用户的问题。
       如果你不知道答案，就说你不知道，不要编造信息。

       上下文：{context}

       问题：{question}

       请提供有帮助的回答："""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
            {
                "context": emb_ret | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    quest = '今日要闻是什么？'
    answer = rag_chain.invoke(quest)

    print(f"问题: {quest}")
    print(f"回答: {answer}")