from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0, "max_length":512})

def create_qa_chain(retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
