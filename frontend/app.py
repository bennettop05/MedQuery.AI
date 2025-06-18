# frontend/app.py
import streamlit as st
import requests

st.set_page_config(page_title="MedQuery.ai", layout="wide")
st.title("💬 MedQuery.ai - Medical Assistant Chatbot")

query = st.text_input("Enter your medical question:")
if query:
    with st.spinner("Thinking..."):
        res = requests.post("http://localhost:8000/query", json={"query": query})
        if res.status_code == 200:
            data = res.json()
            st.success(f"**Answer:** {data['answer']}")
            with st.expander("🔍 Context Used"):
                st.write(data['context'])
        else:
            st.error("Something went wrong. Please try again.")
