"""Week 2/3 check: run real questions through retrieve -> guards -> model.

Run from the backend folder (Ollama must be running):
    python test_rag.py

Look at: correct answers, citations, and which questions get refused.
Save the output for your test log.
"""
import sys
import time
from pathlib import Path

from rag import REFUSAL_THRESHOLD, answer_question, build_messages, call_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
try:
    from retrieve import retrieve
    print("Using REAL retriever (Member A)")
except ImportError as exc:
    from stub_retriever import retrieve
    print(f"!!! USING STUB RETRIEVER ({exc}) !!!")

QUESTIONS = [
    "What is a first appeal under the RTI Act?",
    "My RTI request has been pending for 45 days. What can I do?",
    "How long does it take for a grievance to be resolved?",
    "My neighbor plays loud music at night, what can I do?",
    "Can RTI be used to settle a property dispute?",
]
BASELINE = QUESTIONS[:1]  # also run without context, for the before/after

print(f"Refusal threshold: {REFUSAL_THRESHOLD}")
for q in QUESTIONS:
    print("=" * 70)
    print(f"Q: {q}\n")

    if q in BASELINE:
        start = time.time()
        plain = call_model(build_messages(q, None))
        print(f"WITHOUT context ({time.time() - start:.1f}s):\n{plain}\n")

    chunks = retrieve(q, top_k=4)
    print("Retrieved:", [(c["id"], round(c["score"], 3)) for c in chunks])
    start = time.time()
    result = answer_question(q, chunks)
    print(f"RESULT ({time.time() - start:.1f}s) refused={result['refused']}:")
    print(result["answer"])
    for s in result["sources"]:
        print(f"  - {s['source_doc']}, {s['section']}")
    print()