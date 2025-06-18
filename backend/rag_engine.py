import faiss
import pickle
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# === Load FAISS index safely ===
try:
    faiss_index = faiss.read_index("faiss_index/faiss.index")
    print("[INFO] FAISS index loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load FAISS index: {e}")
    faiss_index = None

# === Load documents associated with FAISS ===
try:
    with open("faiss_index/doc_embeddings.pkl", "rb") as f:
        documents = pickle.load(f)
    print(f"[INFO] Loaded {len(documents)} documents.")
except Exception as e:
    print(f"[ERROR] Failed to load document embeddings: {e}")
    documents = []

# === Load sentence embedding model ===
try:
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("[INFO] SentenceTransformer model loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load SentenceTransformer model: {e}")
    embedder = None

# === Load QA pipeline model ===
try:
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    print("[INFO] QA pipeline loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load QA pipeline: {e}")
    qa_pipeline = None

# === Document retrieval function ===
def retrieve_docs(query, top_k=5):
    if not faiss_index or not documents:
        print("[WARN] FAISS index or document list missing.")
        return ["No documents available to retrieve."]

    try:
        query_embedding = embedder.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        _, indices = faiss_index.search(query_embedding, top_k)
        return [documents[i] for i in indices[0]]
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return ["Error retrieving documents."]

# === Answer generation function ===
def generate_answer(query):
    if not embedder or not qa_pipeline:
        return {
            "answer": "Required models are not loaded properly.",
            "score": 0.0,
            "context": "N/A"
        }

    context_docs = retrieve_docs(query)
    combined_context = " ".join(context_docs)

    try:
        response = qa_pipeline({
            "question": query,
            "context": combined_context
        })

        return {
            "answer": response.get("answer", "No answer found."),
            "score": response.get("score", 0.0),
            "context": combined_context
        }
    except Exception as e:
        print(f"[ERROR] QA generation failed: {e}")
        return {
            "answer": "Error during QA generation.",
            "score": 0.0,
            "context": combined_context
        }
