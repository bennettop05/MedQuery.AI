from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Use HF instead of Ollama
import os
import pickle

def create_vector_store(documents, persist_path="data/faiss_store"):
    print("[INFO] Creating embeddings...")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[INFO] Creating FAISS vector store...")
    vector_store = FAISS.from_documents(documents, embedding)

    print("[INFO] Saving vector store...")
    os.makedirs(persist_path, exist_ok=True)
    faiss_path = os.path.join(persist_path, "index.faiss")
    store_path = os.path.join(persist_path, "store.pkl")
    vector_store.save_local(faiss_path)
    with open(store_path, "wb") as f:
        pickle.dump(vector_store, f)

    print("[INFO] ✅ Vector store saved!")
    return vector_store

def get_retriever(persist_path="data/faiss_store"):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_path = os.path.join(persist_path, "index.faiss")
    store_path = os.path.join(persist_path, "store.pkl")
    if os.path.exists(faiss_path) and os.path.exists(store_path):
        with open(store_path, "rb") as f:
            return pickle.load(f).as_retriever()
    else:
        raise ValueError("Vector store not found. Please run ingestion first.")
