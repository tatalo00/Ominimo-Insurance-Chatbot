from langchain_community.embeddings import OpenAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


load_dotenv()

# --- CONFIG ---
EMBEDDING_DIR = "vectordb"
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Load FAISS Index and Embedding Model ---
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

def get_retriever():
    if not Path(EMBEDDING_DIR).exists():
        raise FileNotFoundError(f"Vector store not found at {EMBEDDING_DIR}. Run embed_store.py first.")

    db = FAISS.load_local(EMBEDDING_DIR, embedding_function, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever

def load_faiss_index(index_path="vectordb"):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True), embedding_model

# --- Example usage ---
if __name__ == "__main__":
    retriever = get_retriever()
    print("✅ Retriever loaded and ready to use.")