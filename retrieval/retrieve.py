import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Load once at startup
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
index = faiss.read_index(str(BASE / "knowledge_base.index"))
with open(BASE / "chunk_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def retrieve(query: str, top_k: int = 4):
    query_embedding = model.encode(
        ["Represent this sentence for searching relevant passages: " + query],
        normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results

# Quick test
if __name__ == "__main__":
    test_queries = [
        "How long does a Public Information Officer have to respond to my RTI?",
        "How do I file a complaint against a government department?",
        "What's the weather today?"  # out-of-scope test
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['section']} — {r['text'][:80]}...")