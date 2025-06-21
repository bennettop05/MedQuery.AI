# frontend/app.py
import streamlit as st
import requests
import time

st.set_page_config(page_title="MedQuery.ai", layout="wide")
st.title("💬 MedQuery.ai - Medical Assistant Chatbot")

query = st.text_input("Enter your medical question:")

if query:
    with st.spinner("Thinking..."):

        backend_url = "http://backend:8000/query"


        max_retries = 15
        delay_seconds = 4

        for i in range(max_retries):
            try:
                res = requests.post(backend_url, json={"query": query})
                break  # Exit loop if successful
            except requests.exceptions.ConnectionError:
                st.warning(f"Backend not ready, retrying ({i+1}/{max_retries})...")
                time.sleep(delay_seconds)
        else:
            st.error("❌ Could not connect to backend after several attempts.")
            st.stop()

        if res.status_code == 200:
            data = res.json()
            st.success(f"**Answer:** {data['answer']}")
            with st.expander("🔍 Context Used"):
                st.write(data['context'])
        else:
            st.error("Something went wrong. Please try again.")
