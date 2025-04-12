from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.document_loaders import TextLoader, PyPDFLoader

import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Model and base URL config
OLLAMA_URL = os.getenv("OLLAMA_URL")
DEEPSEEK = os.getenv("DEEPSEEK")
GEMMA = os.getenv("GEMMA")
LAAMA_3B = os.getenv("LAAMA_3B")

# ------- File Handling Function -------- #
def load_file_as_documents(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    docs = []

    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path)
        return loader.load()
    elif ext == '.pdf':
        loader = PyPDFLoader(file_path)
        return loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    for index, row in df.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(page_content=text, metadata={"source": f"row_{index}"})
        docs.append(doc)

    return docs

file_path = rf'files/sample.txt'
docs = load_file_as_documents(file_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)


ollama_model = Ollama(model=LAAMA_3B, base_url=OLLAMA_URL)
ollama_embeddings = OllamaEmbeddings(model=LAAMA_3B, base_url=OLLAMA_URL)
db = FAISS.from_documents(documents, ollama_embeddings)

# ------- Retrieval + QA -------- #
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_model,
    chain_type="stuff",
    retriever=db.as_retriever()
)


query = "total words in the text file"
answer = qa_chain.run(query)
print(answer)

