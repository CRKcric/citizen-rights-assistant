"""Rehearsal check: run the demo questions through the real pipeline.

Run from the backend folder, with Ollama running:
    python demo_check.py

For each question you want: the right answer, only relevant sources,
and refused=True with the fixed message for the out-of-scope ones.
"""
import sys
import time
from pathlib import Path

import rag

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
try:
    from retrieve import retrieve
    print("Using REAL retriever (Member A)")
except ImportError as exc:
    from stub_retriever import retrieve
    print(f"!!! USING STUB RETRIEVER ({exc}) !!!")

DEMO = [
    ("IN ", "How long does a Public Information Officer have to respond to my RTI?"),
    ("IN ", "How long do I have to file a first appeal?"),
    ("IN ", "What penalty does an officer face for delaying my RTI response?"),
    ("IN ", "What is a first appeal under the RTI Act?"),
    ("OUT", "What's the weather today?"),
    ("OUT", "My neighbor plays loud music at night, what can I do?"),
]

print(f"Refusal threshold: {rag.REFUSAL_THRESHOLD}\n")
for kind, q in DEMO:
    start = time.time()
    chunks = retrieve(q, top_k=4)
    result = rag.answer_question(q, chunks)
    elapsed = time.time() - start
    print("=" * 70)
    print(f"[{kind}] {q}")
    print(f"refused={result['refused']}  top_score={chunks[0]['score']:.3f}  time={elapsed:.1f}s")
    print(result["answer"])
    for s in result["sources"]:
        print(f"  - {s['source_doc']}, {s['section']}")
    print()