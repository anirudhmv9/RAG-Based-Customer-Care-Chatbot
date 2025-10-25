
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

faq_path = os.path.join("..", "data", "faqs_500.csv")
df = pd.read_csv(faq_path)

texts = (df['question'] + " " + df['answer']).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = embeddings.astype('float32')

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join("..", "embeddings", "faq_index.faiss"))

id_to_faq = df.to_dict(orient='records')
with open(os.path.join("..", "embeddings", "faq_mapping.pkl"), 'wb') as f:
    pickle.dump(id_to_faq, f)

print("✅ FAQ Embeddings stored successfully.")
