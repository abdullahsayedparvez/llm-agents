# from langchain.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import json
import hashlib
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
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")


# ------------------- Helpers ------------------- #

def get_file_hash(file_path):
    """Generate MD5 hash of the file's content."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def update_registry_json(file_path, file_hash, registry_path="vector_registry.json"):
    """Log the file and its hash to a JSON file."""
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}

    registry[file_path] = file_hash

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)


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


# ------------------- Main Logic ------------------- #

def process_file(file_path,ollama_model,ollama_embeddings,query):
    file_hash = get_file_hash(file_path)
    vector_store_path = f'vectors/{file_hash}'

    if not os.path.exists('vectors'):
        os.makedirs('vectors')

    if os.path.exists(vector_store_path):
        print("🔁 Loading existing vector store...")
        db = FAISS.load_local(vector_store_path, ollama_embeddings, allow_dangerous_deserialization=True)
    else:
        print("🆕 Creating new vector store...")
        docs = load_file_as_documents(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = text_splitter.split_documents(docs)
        print(documents,f'--> length {len(documents)}')
        db = FAISS.from_documents(documents, ollama_embeddings)
        db.save_local(vector_store_path)
        update_registry_json(file_path, file_hash)
        print("✅ Vector store saved and registered.")

    # Retrieval + QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama_model,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    # query = "total words in the text file"
    answer = qa_chain.invoke(query)
    print(f"\n💬 LLM Answer: {answer['result']}")


# ------------------- Run ------------------- #

if __name__ == "__main__":
    file_path = rf'files/sample.pdf'
    ollama_model = OllamaLLM(model=LAAMA_3B, base_url=OLLAMA_URL)
    ollama_embeddings = OllamaEmbeddings(model=LAAMA_3B, base_url=OLLAMA_URL)
    query = "give me the summary of the document pdf"
    process_file(file_path,ollama_model,ollama_embeddings,query)
