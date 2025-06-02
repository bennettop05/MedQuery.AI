from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def load_all_pdfs(data_dir="./data"):
    all_chunks = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            chunks = load_and_split_pdf(pdf_path)
            all_chunks.extend(chunks)
    return all_chunks
