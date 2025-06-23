# 💬 MedQuery.AI — Medical Assistant Chatbot

MedQuery.AI is a locally hosted **GenAI-powered medical assistant** built using Retrieval-Augmented Generation (RAG). It leverages a custom dataset of medical documents to provide reliable answers to health-related queries via a natural language interface.

---

## 🚀 Features

- 🔎 **Retrieval-Augmented Generation (RAG):** Combines semantic search with language generation.
- ⚡ **FastAPI Backend:** Handles embeddings, retrieval, and response generation.
- 🌐 **Streamlit Frontend:** Lightweight, interactive UI for users.
- 🐳 **Dockerized Setup:** Easy to deploy locally using `docker-compose`.
- 🧠 **Custom-Trained Embeddings:** Uses models like `sentence-transformers` for document indexing.
- 📚 **Local Medical Dataset:** Domain-specific PDF/text content for personalized response generation.

---

## 🧱 Tech Stack

| Layer     | Tech Used                         |
|-----------|-----------------------------------|
| Frontend  | Streamlit                         |
| Backend   | FastAPI, Uvicorn                  |
| GenAI     | OpenAI / HuggingFace Transformers |
| RAG       | FAISS (or similar vector DB), Transformers |
| Embeddings| Sentence Transformers             |
| Container | Docker + Docker Compose           |
| Language  | Python 3.10+                      |

---

## 📂 Dataset

A curated collection of medical documents, articles, and notes (in `.pdf` or `.txt` form), embedded using Sentence Transformers. These documents are indexed and queried via semantic search (FAISS or similar).

---

## 🛠️ How to Run Locally

### 🐳 With Docker (recommended)

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourname/medquery_ai_full.git
   cd medquery_ai_full
Build and run:

bash
Copy
Edit
docker-compose up --build
Open in browser:

Frontend: http://localhost:8501

Backend Docs: http://localhost:8000/docs

To stop:

bash
Copy
Edit
docker-compose down
💬 Sample Questions to Ask
Here are some example queries MedQuery.AI can answer:

“What are the symptoms of Type 2 diabetes?”

“How does insulin resistance affect the body?”

“What is the difference between viral and bacterial infections?”

“Can you explain how blood pressure medication works?”

“What should I do if I have chest pain?”

“Tell me about post-COVID fatigue.”

“What are the side effects of metformin?”

🧠 Future Improvements
Add multilingual support (Hindi/English medical queries)

Enable PDF upload for patient-specific document QA

Add authentication for secure medical queries

Extend RAG with hybrid search (BM25 + dense vectors)

🧑‍💻 Author
Aryan Singh
Email: aryansinghballer@gmail.com
LinkedIn: linkedin.com/in/aryan-singh-aa461a259

