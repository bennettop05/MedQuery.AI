# test_embed.py
from rag.loader import load_all_pdfs
from rag.vector_store import create_vector_store

print("[TEST] Loading PDFs...")
docs = load_all_pdfs("./data")

print(f"[TEST] Loaded {len(docs)} document chunks.")
print("[TEST] Creating vector store...")
create_vector_store(docs)

print("[TEST] Done.")
