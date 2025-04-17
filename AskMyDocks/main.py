from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from pinecone import Pinecone
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import pinecone
import os
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
import streamlit as st


# Model and base URL config
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
DEEPSEEK = os.getenv("DEEPSEEK")
GEMMA = os.getenv("GEMMA")
LAAMA_3B = os.getenv("LAAMA_3B")
PINCONE_API = os.getenv("PINCONE_API")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")


# --- Load PDF ---
def load_pdf(file_path,file_name):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["file_name"] = file_name
    print(f"Loaded {len(documents)} pages from PDF.")
    return documents


# --- Load TXT ---
def load_txt(file_path, file_name):
    loader = TextLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["file_name"] = file_name
    print(f"Loaded TXT file.")
    return documents


# --- Split Text into Chunks ---
def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks



# --- Initialize Embeddings ---
def get_ollama_embeddings(model_name, base_url):
    ollama_embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
    return ollama_embeddings


# --- Create RetrievalQA Chain ---
def create_qa_chain(vectorstore, model_name,ollama_url):
    llm = OllamaLLM(model=model_name, base_url=ollama_url)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return chain


# --- Ask a Question ---
def query_chain(chain, question):
    result = chain({"query": question})
    import re
    cleaned_result = re.sub(r'<think>.*?</think>', '', result["result"], flags=re.DOTALL).strip()
    return cleaned_result


def main():
    st.title("📄 Document Query & Summary App")
    uploaded_file = st.file_uploader("Upload a PDF file",type=["pdf", "txt"])
    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.write(f"Processing file: {file_name}")
        st.write(f"🔍Document Scanning **{file_name}**...")
        
         # Save the uploaded PDF file locally
        file_path = f"files/{file_name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # load,convert,embedd
        embedding_model = get_ollama_embeddings(DEEPSEEK,OLLAMA_URL)
        load_vector_file = f'vector_db/{file_name}'
        file_extension = os.path.splitext(file_name)[1].lower()
        if os.path.exists(load_vector_file):
            st.success("✅ Ready to answer")
            vectorstore = FAISS.load_local(load_vector_file, embedding_model, allow_dangerous_deserialization=True)
        else:
            with st.spinner("🔄 Processing document... This might take a few moments."):
                # Convert the file to docs and chunks
                PDF_PATH = f"files/{file_name}"
                if file_extension == ".pdf":
                    docs = load_pdf(file_path, file_name)
                elif file_extension == ".txt":
                    docs = load_txt(file_path, file_name)
                else:
                    st.error("Unsupported file type.")
                    return
                chunks = chunk_documents(docs)

                # Save the vector database
                vectorstore = FAISS.from_documents(chunks, embedding_model)
                vectorstore.save_local(load_vector_file)
        qa_chain = create_qa_chain(vectorstore,DEEPSEEK,OLLAMA_URL)

        # Display a text box to input the question
        question = st.text_input("Enter your question:")
        if question:
            with st.spinner("🧠 Thinking..."):
                answer = query_chain(qa_chain, question)
                st.write(f"Answer: {answer}")
        if st.button("Get Summary of Topics"):
            with st.spinner("📖 Generating summary..."):
                summary = query_chain(qa_chain,  "Summarize the document into 5 key bullet points. Avoid repeating the same info. Start each point with a number.Give me the information from file")
                st.write(f"{summary}")

if __name__ == "__main__":
    main()
