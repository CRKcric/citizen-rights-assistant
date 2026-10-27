"""Week 1 check: is Phi-3-mini running, and how fast is it?

Prereqs:
  1. Install Ollama (https://ollama.com)
  2. ollama pull phi3:mini
  3. pip install requests
Run:  python test_phi3.py
"""
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"

QUESTIONS = [
    "In one sentence, what is the RTI Act in India?",
    "Explain what a first appeal is, in simple words.",
]

for q in QUESTIONS:
    start = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": q, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    elapsed = time.time() - start
    print(f"Q: {q}")
    print(f"A: {resp.json()['response'].strip()}")
    print(f"Time: {elapsed:.1f}s\n")