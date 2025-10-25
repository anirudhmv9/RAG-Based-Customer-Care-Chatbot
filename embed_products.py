import os
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

product_path = Path("..") / "data" / "product_catalog_1000.csv"
embedding_dir = Path("..") / "embeddings"
embedding_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(product_path)
df["text"] = df["name"] + " - Features: " + df["features"] + f" | Price: ₹" + df["price"].astype(str)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist())

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, str(embedding_dir / "product_index.faiss"))

id_to_product = df.to_dict(orient="records")
with open(embedding_dir / "product_mapping.pkl", "wb") as f:
    pickle.dump(id_to_product, f)

print("✅ Product embeddings stored successfully.")