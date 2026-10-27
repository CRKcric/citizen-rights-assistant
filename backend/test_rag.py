"""Week 2 check: compare answers WITHOUT and WITH retrieved context.

Run from the backend folder (Ollama must be running):
    python test_rag.py

Save the output. The 'first appeal' before/after is your baseline
for the report and the demo.
"""
import time

from rag import answer_question, build_messages, call_model
from stub_retriever import retrieve

QUESTIONS = [
    "What is a first appeal under the RTI Act?",
    "My RTI request has been pending for 45 days. What can I do?",
    "My neighbor plays loud music at night, what can I do?",
]

for q in QUESTIONS:
    print("=" * 70)
    print(f"Q: {q}\n")

    start = time.time()
    plain = call_model(build_messages(q, None))
    print(f"WITHOUT context ({time.time() - start:.1f}s):\n{plain}\n")

    chunks = retrieve(q, top_k=4)
    print(f"Retrieved: {[(c['section'], c['score']) for c in chunks]}")
    start = time.time()
    result = answer_question(q, chunks)
    print(f"WITH context ({time.time() - start:.1f}s) refused={result['refused']}:")
    print(result["answer"])
    for s in result["sources"]:
        print(f"  - {s['source_doc']}, {s['section']}")
    print()