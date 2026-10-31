import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Generate embeddings for each chunk's text
texts = [f"{chunk['section']}: {chunk['text']}" for chunk in chunks]
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
embeddings = np.array(embeddings, dtype="float32")

# Build FAISS index (cosine similarity via normalized inner product)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and chunk metadata
faiss.write_index(index, "knowledge_base.index")
with open("chunk_metadata.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Index built with {index.ntotal} vectors, dimension {dimension}")
print("Saved: knowledge_base.index, chunk_metadata.json")