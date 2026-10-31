"""RAG generation for the Citizen Rights Assistant (Member B).

Takes the question plus retrieved chunks, asks Phi-3-mini to answer ONLY
from those chunks, and builds the `sources` list from chunk metadata
(never from model text, so citations cannot be invented).

Guards (in order):
  1. top retrieval score below REFUSAL_THRESHOLD -> refuse, model not called
  2. the model says it cannot answer (exact sentence or a similar phrase)
  3. the answer cites a section number that is not in the retrieved context
"""
import re

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "phi3:mini"
MODEL_TIMEOUT_SECONDS = 120

# The model is told to use this exact sentence when the context is not enough.
NO_ANSWER = "I cannot answer this from the available information."

# Phi-3-mini often refuses in its own words instead of the exact sentence,
# so Guard 2 also looks for these phrases. Kept narrow on purpose: phrases
# like "unable to" appear in legitimate answers.
REFUSAL_MARKERS = (
    NO_ANSWER.lower(),
    "cannot answer",
    "can't answer",
    "i'm sorry",
    "i am sorry",
    "does not contain",
    "do not contain",
    "cannot assist",
    "can only provide information",
)

REFUSAL_MESSAGE = (
    "I don't have verified information on that. "
    "Please check rti.gov.in or contact the relevant authority."
)

# Provisional: lowered from 0.65 after the model-in-the-loop run (pipeline_eval.py):
# 0.65 refused two valid questions, and the model + guards refused every
# near-topic out-of-scope question scoring 0.58-0.68. Re-check on 30+ questions.
REFUSAL_THRESHOLD = 0.55

SYSTEM_PROMPT = f"""You are an assistant that explains Indian RTI Act 2005 rules \
and government grievance procedures to ordinary citizens.

Rules you must follow:
1. Answer ONLY using the numbered context passages below. Do not use any \
outside knowledge.
2. If the context does not contain the answer, reply with exactly: {NO_ANSWER}
3. Write in plain, simple language, in at most 4 sentences.
4. When you state a rule, deadline or procedure, name its source exactly as \
labelled in the context (for example Section 7(1), or Grievance Resolution \
Process). Never write a section number that does not appear in the context.
5. Passages are ordered by relevance. If they describe different processes \
(for example an RTI application versus a CPGRAMS grievance), use the one that \
matches the user's situation, and prefer the lower-numbered passage when unsure."""


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
                "options": {
                    "temperature": 0,
                    "num_predict": 250,
                    # the model sometimes keeps going with invented follow-ups
                    "stop": ["\n\nQuestion:", "Question:", "\n\n##"],
                },
            },
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise ModelUnavailableError("Cannot reach Ollama on localhost:11434") from exc
    except requests.exceptions.Timeout as exc:
        raise ModelTimeoutError("Model timed out") from exc
    return resp.json()["message"]["content"].strip()


def clean_answer(text: str) -> str:
    """Drop trailing junk such as '## Instruction 2 ...' the model sometimes adds."""
    return re.split(r"\n\s*#{2,}", text)[0].strip()


SECTION_RE = re.compile(r"[Ss]ection\s+(\d+(?:\([a-zA-Z0-9]+\))*)")
WORD_RE = re.compile(r"[a-z]{5,}")


def _label_matches(label: str, cited: set[str]) -> bool:
    # "Section 7" counts as citing "7(1)"
    return any(label == f or label.startswith(f + "(") for f in cited)


def unsupported_sections(answer: str, chunks: list[dict]) -> list[str]:
    """Section numbers the answer cites that appear nowhere in the context."""
    allowed = set()
    for c in chunks:
        for m in SECTION_RE.finditer(c["section"] + " " + c["text"]):
            allowed.add(m.group(1).lower())
    bad = []
    for m in SECTION_RE.finditer(answer):
        found = m.group(1).lower()
        if not any(a == found or a.startswith(found + "(") for a in allowed):
            bad.append(m.group(1))
    return bad


def select_chunks(answer: str, chunks: list[dict]) -> list[dict]:
    """Chunks the answer actually relies on, so we do not cite unused ones."""
    cited = {m.group(1).lower() for m in SECTION_RE.finditer(answer)}
    by_label = []
    for c in chunks:
        m = SECTION_RE.search(c["section"])
        if m and _label_matches(m.group(1).lower(), cited):
            by_label.append(c)
    if by_label:
        return by_label

    # No section number in the answer (e.g. CPGRAMS answers): pick the chunks
    # sharing the most content words with the answer.
    answer_words = set(WORD_RE.findall(answer.lower()))
    scored = []
    for c in chunks:
        words = set(WORD_RE.findall((c["section"] + " " + c["text"]).lower()))
        scored.append((len(answer_words & words), c))
    best = max(s for s, _ in scored)
    if best == 0:
        return chunks[:1]
    return [c for s, c in scored if s >= 0.6 * best][:2]


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


def refusal(reason: str, debug: bool) -> dict:
    result = {"answer": REFUSAL_MESSAGE, "refused": True, "sources": []}
    if debug:  # extra key only for evaluation scripts, never in the API
        result["reason"] = reason
    return result


def answer_question(question: str, chunks: list[dict], debug: bool = False) -> dict:
    """Return a dict in the /chat response shape."""
    # Guard 1: weak retrieval -> refuse without calling the model
    if not chunks or chunks[0]["score"] < REFUSAL_THRESHOLD:
        return refusal("low_score", debug)

    answer = clean_answer(call_model(build_messages(question, chunks)))

    # Guard 2: the model says the context is not enough -> refuse
    lowered = answer.lower()
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return refusal("model_refused", debug)

    # Guard 3: the answer cites a section number that is not in the context
    # (invented citation) -> refuse
    if unsupported_sections(answer, chunks):
        return refusal("invented_section", debug)

    result = {
        "answer": answer,
        "refused": False,
        "sources": build_sources(select_chunks(answer, chunks)),
    }
    if debug:
        result["reason"] = ""
    return result