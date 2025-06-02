from langchain.embeddings import OllamaEmbeddings

# Instantiate Ollama embedding model (local)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

def get_embeddings(documents):
    texts = [doc.page_content for doc in documents]
    return embedding_model.embed_documents(texts)
