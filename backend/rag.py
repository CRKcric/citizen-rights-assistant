"""RAG generation for the Citizen Rights Assistant (Member B, Week 2).

Takes the question plus retrieved chunks, asks Phi-3-mini to answer ONLY
from those chunks, and builds the `sources` list from chunk metadata
(never from model text, so citations cannot be invented).
"""
import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "phi3:mini"
MODEL_TIMEOUT_SECONDS = 120

# The model is told to use this exact sentence when the context is not enough.
# Week 3 guardrail: if the answer contains it, treat the response as a refusal.
NO_ANSWER = "I cannot answer this from the available information."

REFUSAL_MESSAGE = (
    "I don't have verified information on that. "
    "Please check rti.gov.in or contact the relevant authority."
)

SYSTEM_PROMPT = f"""You are an assistant that explains Indian RTI Act 2005 rules \
and government grievance procedures to ordinary citizens.

Rules you must follow:
1. Answer ONLY using the numbered context passages below. Do not use any \
outside knowledge.
2. If the context does not contain the answer, reply with exactly: {NO_ANSWER}
3. Write in plain, simple language, in at most 4 sentences.
4. Mention the section number (for example Section 7(1)) when you state a \
rule, deadline or procedure.
5. Never invent section numbers, deadlines, fees or authorities."""


class ModelUnavailableError(Exception):
    """Ollama is not reachable (not running, wrong port)."""


class ModelTimeoutError(Exception):
    """The model took too long to answer."""


def format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c['source_doc']}, {c['section']}: {c['text']}")
    return "\n\n".join(lines)


def build_messages(question: str, chunks: list[dict] | None) -> list[dict]:
    if chunks:
        user = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    else:
        user = f"Question: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call_model(messages: list[dict]) -> str:
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",  # keep the model loaded between questions
                   "options": {"temperature": 0, "num_predict": 250, "stop": ["\n\nQuestion:", "Question:"]},
            },
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise ModelUnavailableError("Cannot reach Ollama on localhost:11434") from exc
    except requests.exceptions.Timeout as exc:
        raise ModelTimeoutError("Model timed out") from exc
    return resp.json()["message"]["content"].strip()


def build_sources(chunks: list[dict]) -> list[dict]:
    """Citations come from chunk metadata, not from the model's text."""
    sources = []
    for c in chunks:
        snippet = c["text"]
        if len(snippet) > 200:
            snippet = snippet[:200].rstrip() + "..."
        sources.append(
            {
                "source_doc": c["source_doc"],
                "section": c["section"],
                "snippet": snippet,
                "source_url": c["source_url"],
            }
        )
    return sources


def answer_question(question: str, chunks: list[dict]) -> dict:
    """Return a dict in the /chat response shape."""
    if not chunks:
        return {"answer": REFUSAL_MESSAGE, "refused": True, "sources": []}

    answer = call_model(build_messages(question, chunks))
    return {
        "answer": answer,
        "refused": False,
        "sources": build_sources(chunks),
    }