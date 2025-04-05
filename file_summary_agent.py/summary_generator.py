from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os

LAAMA_3B = os.getenv("LAAMA_3B")
OLLAMA_URL = os.getenv("OLLAMA_URL")


def generate_summary(df):
    csv_text = df.to_csv(index=False)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = splitter.create_documents([csv_text])
    embeddings = OllamaEmbeddings(model=LAAMA_3B, base_url=OLLAMA_URL)
    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever()
    llm = Ollama(model=LAAMA_3B, base_url=OLLAMA_URL)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    summary = qa_chain.run("Give me a professional summary of this dataset. Highlight trends, stats, and insights.")
    return summary
