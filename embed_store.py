from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from document_loader import extract_and_chunk
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# --- CONFIG ---
EMBEDDING_DIR = "vectordb"
PDF_DIR = Path("./Documents/")

# --- Load PDF and Chunk ---
all_docs = []
for pdf_path in PDF_DIR.glob("*.pdf"):
    all_docs.extend(extract_and_chunk(pdf_path))

print(f"Loaded {len(all_docs)} chunks for embedding.")

# --- Embedding Model Setup ---
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

# --- Create Vector Store ---
db = FAISS.from_documents(all_docs, embedding_function)

# --- Persist Vector Store ---
os.makedirs(EMBEDDING_DIR, exist_ok=True)
db.save_local(EMBEDDING_DIR)
print(f"FAISS vector store saved to: {EMBEDDING_DIR}")