# 🧠 Contextual Customer Support Chatbot (RAG-based)

## 🚀 Overview
This project is a **Retrieval-Augmented Generation (RAG)**-based **Contextual Customer Support Chatbot** designed to deliver **instant, accurate, and context-aware responses** to customer queries related to products and FAQs.

It integrates **FAISS vector search** for fast semantic retrieval and **LLaMA 3** (via **Ollama**) for natural language response generation, all served through an intuitive **Streamlit web interface**.  
The chatbot effectively reduces repetitive customer support queries by automating common responses with citations and context.

---

 🧩 System Architecture


User Query → Text Embedding (Sentence Transformers)
           → FAISS Vector Search (Retrieval)
           → Context Augmentation (RAG)
           → LLaMA 3 (via Ollama) Response Generation
           → Streamlit Chat UI (Response Display)
