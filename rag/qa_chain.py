from langchain.chains import RetrievalQA
from langchain.llms import Ollama  # Or Llama/Groq wrappers as you prefer

llm = Ollama(model="llama3")  # local LLM

def create_qa_chain(retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
