from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(documents):
    texts = [doc.page_content for doc in documents]
    return embedding_model.embed_documents(texts)
