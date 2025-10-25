import os
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FAQ_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faq_index.faiss")
FAQ_MAPPING_PATH = os.path.join(EMBEDDINGS_DIR, "faq_mapping.pkl")
PRODUCT_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "product_index.faiss")
PRODUCT_MAPPING_PATH = os.path.join(EMBEDDINGS_DIR, "product_mapping.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

faq_index = faiss.read_index(FAQ_INDEX_PATH)
with open(FAQ_MAPPING_PATH, "rb") as f:
    faq_mapping = pickle.load(f)

product_index = faiss.read_index(PRODUCT_INDEX_PATH)
with open(PRODUCT_MAPPING_PATH, "rb") as f:
    product_mapping = pickle.load(f)

def get_relevant_faqs(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = faq_index.search(query_embedding, k)
    results = [faq_mapping[i] for i in indices[0] if i < len(faq_mapping)]
    return results

def get_relevant_products(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = product_index.search(query_embedding, k)
    results = [product_mapping[i] for i in indices[0] if i < len(product_mapping)]
    return results

def generate_response(query, faqs, products):
    faq_context = "\n".join([f"Q: {f['question']}\nA: {f['answer']}" for f in faqs])
    product_context = "\n".join(
        [f"Product: {p['name']}\nFeatures: {p['features']}\nPrice: ₹{p['price']}" for p in products]
    )

    prompt = f"""You are a helpful customer support assistant.
Use the following FAQs and product info to answer the user's question.

FAQs:
{faq_context}

Products:
{product_context}

User: {query}
Support:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]