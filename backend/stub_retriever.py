"""Stub retriever for Week 2 (Member B).

Mimics Member A's retrieve(query, top_k=4) so the backend is never blocked.
The chunk text below is a PARAPHRASE written for testing only. Member A's
real chunks must be verified against the original Act text.

Scores here come from simple keyword hits, NOT embeddings, so they say
nothing about what real scores will look like. Set the refusal threshold
from Member A's evaluation data in Week 3, not from these numbers.

When A's retrieve() is ready, the swap is one import line in main.py.
"""

import re

STUB_CHUNKS = [
    {
        "id": "rti_act_s6_1",
        "text": (
            "A person who wishes to obtain information under the RTI Act must "
            "make a request in writing or electronically, in English, Hindi or "
            "the official language of the area, to the Public Information "
            "Officer (PIO) of the concerned public authority, along with the "
            "prescribed fee."
        ),
        "source_doc": "RTI Act 2005",
        "section": "Section 6(1)",
        "source_url": "https://rti.gov.in",
        "doc_type": "act",
        "keywords": ["file", "filing", "apply", "request", "fee", "fees", "pio"],
    },
    {
        "id": "rti_act_s7_1",
        "text": (
            "The PIO must either provide the information or reject the request "
            "within 30 days of receiving it. Where the information concerns "
            "the life or liberty of a person, it must be provided within 48 "
            "hours."
        ),
        "source_doc": "RTI Act 2005",
        "section": "Section 7(1)",
        "source_url": "https://rti.gov.in",
        "doc_type": "act",
        "keywords": ["deadline", "days", "pending", "response", "reply", "no response", "48 hours", "time"],
    },
    {
        "id": "rti_act_s8_1",
        "text": (
            "Section 8(1) lists categories of information that are exempt from "
            "disclosure, such as information that would prejudicially affect "
            "the sovereignty, integrity or security of the State, or "
            "information that a court has expressly forbidden to be published."
        ),
        "source_doc": "RTI Act 2005",
        "section": "Section 8(1)",
        "source_url": "https://rti.gov.in",
        "doc_type": "act",
        "keywords": ["exempt", "exemption", "reject", "rejected", "denied", "refuse", "refused", "denial"],
    },
    {
        "id": "rti_act_s19_1",
        "text": (
            "A person who does not receive a decision within the time limit, "
            "or who is dissatisfied with the PIO's decision, may file a first "
            "appeal with an officer senior in rank to the PIO in the same "
            "public authority (the First Appellate Authority). The appeal must "
            "be filed within 30 days from the expiry of the time limit or from "
            "receiving the decision."
        ),
        "source_doc": "RTI Act 2005",
        "section": "Section 19(1)",
        "source_url": "https://rti.gov.in",
        "doc_type": "act",
        "keywords": ["first appeal", "appeal", "appellate", "dissatisfied", "pending", "no response"],
    },
    {
        "id": "rti_act_s19_3",
        "text": (
            "A second appeal against the decision of the First Appellate "
            "Authority lies to the Central Information Commission or the "
            "relevant State Information Commission, within 90 days of the "
            "date the decision was made or should have been made."
        ),
        "source_doc": "RTI Act 2005",
        "section": "Section 19(3)",
        "source_url": "https://rti.gov.in",
        "doc_type": "act",
        "keywords": ["second appeal", "commission", "cic", "sic", "information commission"],
    },
]


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """Return up to top_k chunks (best first), each with a 0-1 'score'.

    Only chunks with at least one keyword hit are returned, so an
    out-of-scope question gets an empty list.
    """
    q = query.lower()
    results = []
    for chunk in STUB_CHUNKS:
        # whole-word match, so e.g. "sic" does not match "music"
        hits = sum(
            1
            for kw in chunk["keywords"]
            if re.search(r"\b" + re.escape(kw) + r"\b", q)
        )
        if hits == 0:
            continue
        result = {k: v for k, v in chunk.items() if k != "keywords"}
        result["score"] = min(1.0, hits / 2)
        results.append(result)
    results.sort(key=lambda c: c["score"], reverse=True)
    return results[:top_k]