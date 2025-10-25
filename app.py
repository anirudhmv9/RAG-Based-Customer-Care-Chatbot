import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

from rag_engine import get_relevant_faqs, get_relevant_products, generate_response

st.set_page_config(page_title="🧠 RAG Support Chatbot", layout="centered")
st.title("💬 Contextual Support Chatbot")
st.markdown("Ask any *product* or *service-related* question below.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faq_references" not in st.session_state:
    st.session_state.faq_references = []

for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if chat["sender"] == "user" else "assistant"):
        st.markdown(chat["message"])
        st.caption(f"{chat['time']}")
        
        if chat["sender"] == "bot" and i // 2 < len(st.session_state.faq_references):
            referenced_faqs = st.session_state.faq_references[i // 2]
            if referenced_faqs:
                with st.expander("📚 FAQs Referenced"):
                    for q in referenced_faqs:
                        st.markdown(f"- {q}")

user_input = st.chat_input("e.g. How do I return a product?")
if user_input:
    timestamp = datetime.now().strftime("%H:%M:%S")

    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({
        "sender": "user",
        "message": user_input,
        "time": timestamp
    })

    top_faqs = get_relevant_faqs(user_input)
    top_products = get_relevant_products(user_input)

    bot_response = generate_response(user_input, top_faqs, top_products)

    st.chat_message("assistant").markdown(bot_response)
    st.session_state.chat_history.append({
        "sender": "bot",
        "message": bot_response,
        "time": datetime.now().strftime("%H:%M:%S")
    })

    st.session_state.faq_references.append(
        [faq["question"] for faq in top_faqs]
    )
