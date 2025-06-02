import streamlit as st
from rag.loader import load_all_pdfs
from rag.embedder import get_embeddings
from rag.vector_store import create_vector_store, get_retriever
from rag.qa_chain import create_qa_chain
from utils.helpers import timer

st.set_page_config(page_title="MedQuery.AI")
st.title("MedQuery.AI 🏥 Medical Document Assistant")
st.sidebar.header("📂 Document Control")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.retriever = None
    st.session_state.vector_store = None

@timer
def load_and_prepare():
    chunks = load_all_pdfs("./data")
    embeddings = get_embeddings(chunks)
    vector_store = create_vector_store(chunks, embeddings)
    retriever = get_retriever(vector_store)
    qa_chain = create_qa_chain(retriever)
    return qa_chain, retriever, vector_store

if st.sidebar.button("Load PDFs & Build Index"):
    with st.spinner("Loading, embedding and indexing documents..."):
        qa_chain, retriever, vector_store = load_and_prepare()
        st.session_state.qa_chain = qa_chain
        st.session_state.retriever = retriever
        st.session_state.vector_store = vector_store
    st.success("✅ Documents loaded and indexed!")

if st.session_state.qa_chain:
    query = st.text_input("Ask a medical question:")
    if query:
        with st.spinner("Fetching answer..."):
            result = st.session_state.qa_chain.run(query)
            st.markdown("### 💬 Answer:")
            st.write(result['result'] if isinstance(result, dict) else result)
            if 'source_documents' in result:
                st.markdown("---")
                st.markdown("### 📄 Source document chunks:")
                for i, doc in enumerate(result['source_documents']):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(doc.page_content[:500])

else:
    st.info(" Click 'Load PDFs & Build Index' to start.")
