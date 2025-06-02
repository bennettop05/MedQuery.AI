from langchain.vectorstores import FAISS  # or swap with ObjectBox if configured

vector_db = None

# Create FAISS retriever

def create_retriever(documents, embeddings):
    global vector_db
    vector_db = FAISS.from_texts([doc.page_content for doc in documents], embeddings)
    return vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def retrieve_docs(query, retriever):
    return retriever.get_relevant_documents(query)