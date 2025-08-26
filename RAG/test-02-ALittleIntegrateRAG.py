import os

from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter


class SimplePDFRAG:
    def __init__(self, ppath):
        self.embeddings = HuggingFaceEmbeddings(model_name=r"./models/BAAI/bge-large-zh-v1___5", model_kwargs={'device': 'cpu'})

        # 加载和处理PDF
        loader = PyPDFLoader(ppath)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = splitter.split_documents(documents)

        texts = [doc.page_content for doc in split_docs]
        self.vectorstore = InMemoryVectorStore.from_texts(texts, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(k=3)

        # 构建RAG链
        self.chain = (
                RunnableParallel({
                    "context": self.retriever | (lambda docs: "\n".join([d.page_content for d in docs])),
                    "question": RunnablePassthrough()
                })
                | ChatPromptTemplate.from_template("基于以下旅游信息回答：{context}\n\n问题：{question}\n\n回答：")
                | ChatOpenAI(api_key=os.getenv("QWEN_KEY"),
                             base_url=os.getenv("QWEN_URL"),
                             model="qwen-plus")
                | StrOutputParser()
        )

    def ask(self, question):
        return self.chain.invoke(question)


if __name__ == '__main__':
    # 使用
    load_dotenv()
    pdf_path = r"./RAG_documents/example.pdf"
    rag = SimplePDFRAG(pdf_path)
    print(rag.ask("故宫的天气怎么样？"))
