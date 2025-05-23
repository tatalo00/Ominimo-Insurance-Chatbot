import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from load_db import get_retriever
import time
from load_db import load_faiss_index


db, embedding_model = load_faiss_index()


# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"

prompt_template_english = """
You are an insurance expert assistant. Answer the question below as accurately as possible using only the provided context and politely.
If the answer is not in the context, say "This question is outside the scope of Ominimo insurance topics."

<context>
{context}
</context>

Question: {question}
Answer in English:
"""

prompt_template_hungarian = """
Biztosítási szakértő asszisztens vagy. Válaszolj az alábbi kérdésre a lehető legpontosabban, kizárólag a megadott kontextus felhasználásával. Úgy hangzik, mintha egy valós személy lennél, aki egy biztosítótársaságnál dolgozik.
Ha a válasz nem a szövegkörnyezetben van, mondd "A kérdés nem tartozik az Ominimo biztosítási témakörébe."

<kontextus>
{context}
</kontextus>

Kérdés: {question}
Válasz magyarul:
"""


# --- Load Retriever and LLM ---
retriever = get_retriever()
llm = ChatOpenAI(model_name=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

prompt = PromptTemplate(
    template=prompt_template_english,
    input_variables=["context", "question"]
)

# --- Create RAG QA Chain with Source Tracking ---
def get_qa_chain(language: str = "en"):
    if language == "hu":
        template = prompt_template_hungarian
    else:
        template = prompt_template_english

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# --- Example usage ---
if __name__ == "__main__":
    query = input("Ask a question: ")

    # Show retrieved chunks first
    docs = retriever.invoke(query)

    print("\n🔍 Retrieved Chunks:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk #{i} ---")
        print(doc.page_content[:1000])
        print(f"[source: {doc.metadata['source']}, page: {doc.metadata['page']}, header: {doc.metadata.get('header', 'N/A')}]")


    start = time.time()
    result = get_qa_chain().invoke({"query": query})
    end = time.time()

    print("\nAnswer:")
    print(result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['source']} (page {doc.metadata['page']}, section: {doc.metadata.get('header', 'N/A')})")

    response_time = end - start
    print(f"Response time: {response_time:.2f} sec")