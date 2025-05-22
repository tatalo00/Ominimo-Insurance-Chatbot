import os
import streamlit as st
import time
from qa_chain import get_qa_chain

# --- Streamlit UI Config ---
st.set_page_config(page_title="Ominimo Insurance Chatbot", page_icon="📄")
st.title("📄 Ominimo Insurance Chatbot")

# --- Initialize Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Language Toggle ---
language_display = st.radio("Válasszon nyelvet / Choose language:", ["Magyar", "English"], horizontal=True)
language = "hu" if language_display == "Magyar" else "en"
st.session_state["language"] = language

# --- User Input ---
placeholder = "Tegye fel a kérdését a biztosítással kapcsolatban..." if language == "hu" else "Ask your insurance-related question..."
user_query = st.chat_input(placeholder)


thinking_placeholder = "Gondolkodom..." if language == "hu" else "Thinking..."
if user_query:
    start_time = time.time()
    with st.spinner(thinking_placeholder):
        result = get_qa_chain(language).invoke(user_query)
    end_time = time.time()

    answer = result["result"]
    sources = result["source_documents"]
    response_time = round(end_time - start_time, 2)

    st.session_state.chat_history.append({
        "query": user_query,
        "answer": answer,
        "sources": sources,
        "time": response_time
    })


thinking_placeholder = "Gondolkodom..." if language == "hu" else "Thinking..."
# --- Display Chat History with Feedback ---
for i, entry in enumerate(st.session_state.chat_history):
    index = len(st.session_state.chat_history)- 1 - i  # to keep reference aligned

    with st.chat_message("user"):
        st.write(entry["query"])

    with st.chat_message("assistant"):
        st.write(entry["answer"])
        st.caption(f"⏱ Válaszidő: {entry['time']} másodperc" if language=="hu" else f"⏱ Response time: {entry['time']} seconds")

        if entry["sources"]:
            with st.expander("Források" if language=="hu" else "Resources"):
                for doc in entry["sources"]:
                    st.markdown(f"- `{doc.metadata['source']}` (oldal {doc.metadata.get('page', '?')}, szakasz: {doc.metadata.get('header', 'N/A')})"
                                if language=="hu" else
                                f"- `{doc.metadata['source']}` (page {doc.metadata.get('page', '?')}, header: {doc.metadata.get('header', 'N/A')})")

        # Feedback
        feedback_key = f"feedback_{index}"
        if feedback_key not in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Hasznos" if language=="hu" else "👍 Useful", key=f"up_{index}"):
                    st.session_state[feedback_key] = "positive"
            with col2:
                if st.button("👎 Nem hasznos" if language=="hu" else "👎 Not useful", key=f"down_{index}"):
                    st.session_state[feedback_key] = "negative"
        else:
            fb = st.session_state[feedback_key]
            icon = "👍" if fb == "positive" else "👎"
            st.caption(f"Visszajelzés: {icon} ({'Hasznos' if fb == 'positive' else 'Nem hasznos'})" if language=="hu" else
                       f"Feedback: {icon} ({'Useful' if fb == 'positive' else 'Not useful'})")