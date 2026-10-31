"""Model-in-the-loop evaluation (run from the backend folder, Ollama running).

Runs every question through retrieve -> model with the score threshold
switched OFF, so we can see where the model/guards refuse on their own,
then shows what each candidate threshold would do. Takes several minutes.

Output: a summary table here, and pipeline_eval_results.csv with every
answer and cited section so you can READ the answers (this script only
measures answered vs refused, not whether an answer is correct).
"""
import csv
import sys
from pathlib import Path

import rag

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
try:
    from retrieve import retrieve
    print("Using REAL retriever (Member A)")
except ImportError as exc:
    from stub_retriever import retrieve
    print(f"!!! USING STUB RETRIEVER ({exc}) !!!")

IN_SCOPE = [
    # Member A's evaluation questions
    "How do I file an RTI request?",
    "In which languages can I submit an RTI application?",
    "Do I need to give a reason for requesting information?",
    "What happens if my application goes to the wrong department?",
    "How long does a Public Information Officer have to respond to my RTI?",
    "What's the deadline if my request concerns life or liberty?",
    "What happens if I don't get a response in time?",
    "When am I not required to pay a fee for the information?",
    "Can the government refuse to give me information about national security?",
    "Is my personal information protected from being disclosed to others under RTI?",
    "Can I get cabinet meeting records through RTI?",
    "Can old government records be withheld forever?",
    "What do I do if my RTI request is rejected?",
    "How long do I have to file a first appeal?",
    "What's the difference between a first and second appeal?",
    "What penalty does an officer face for delaying my RTI response?",
    "Who has to prove the denial was justified in an appeal?",
    "How do I file a complaint against a government department?",
    "Can I file a grievance through WhatsApp?",
    "Who handles my grievance after I submit it?",
    "How long does it take for a grievance to be resolved?",
    "How can I check the status of my complaint?",
    "What happens if my grievance takes longer than expected?",
    "What do I do if I'm not satisfied with how my grievance was resolved?",
    "How long does an appeal on CPGRAMS take?",
    # situation-style phrasing, the way real users write
    "My RTI request has been pending for 45 days. What can I do?",
    "I filed an RTI request about road repair funds. It's been 45 days, no response.",
    "What is a first appeal under the RTI Act?",
    "The officer rejected my RTI application. What now?",
]
OUT_OF_SCOPE = [
    "What's the weather today?",
    "Can you help me write a resume?",
    "What's the punishment for theft under IPC?",
    "How do I apply for a passport?",
    "Can RTI be used to settle a property dispute?",
    "How do I file an FIR with the police?",
    "How do I complain about a defective product to a consumer court?",
    "My neighbor plays loud music at night, what can I do?",
]

rag.REFUSAL_THRESHOLD = 0.0  # measure the model/guards on their own
rows = []
questions = [(q, True) for q in IN_SCOPE] + [(q, False) for q in OUT_OF_SCOPE]
for n, (q, in_scope) in enumerate(questions, start=1):
    chunks = retrieve(q, top_k=4)
    try:
        result, reason = rag.answer_with_reason(q, chunks)
        refused, answer = result["refused"], result["answer"]
        cited = "; ".join(s["section"] for s in result["sources"])
    except (rag.ModelUnavailableError, rag.ModelTimeoutError) as exc:
        refused, answer, cited, reason = True, f"ERROR: {exc}", "", "error"
    top = chunks[0]["score"] if chunks else 0.0
    rows.append(
        {
            "question": q,
            "in_scope": in_scope,
            "top_score": round(top, 3),
            "top_ids": ", ".join(c["id"] for c in chunks),
            "refused_by_model_or_guards": refused,
            "refusal_reason": reason,
            "reason": reason or "",
            "answer": answer,
            "cited": cited,
        }
    )
    tag = "REFUSED " if refused else "ANSWERED"
    print(f"[{n}/{len(questions)}] {tag} {top:.3f} {'IN ' if in_scope else 'OUT'} {q}"
          + (f"  <- {reason}" if refused else ""))

with open("pipeline_eval_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

n_in = sum(r["in_scope"] for r in rows)
n_out = len(rows) - n_in
print(f"\nBefore any score threshold: in-scope answered "
      f"{sum(r['in_scope'] and not r['refused_by_model_or_guards'] for r in rows)}/{n_in}, "
      f"out-of-scope answered "
      f"{sum(not r['in_scope'] and not r['refused_by_model_or_guards'] for r in rows)}/{n_out}")

print("\nthreshold  in-scope answered  out-of-scope answered")
for t in [0.0, 0.50, 0.55, 0.60, 0.65, 0.70]:
    ok_in = sum(r["in_scope"] and r["top_score"] >= t and not r["refused_by_model_or_guards"] for r in rows)
    bad_out = sum(not r["in_scope"] and r["top_score"] >= t and not r["refused_by_model_or_guards"] for r in rows)
    print(f"{t:9.2f}  {ok_in:3d}/{n_in:<14d}  {bad_out:3d}/{n_out}")

print("\nWhy things were refused (threshold is off, so low_score = no chunks):")
for why in ["model_refused", "invented_section", "low_score", "error"]:
    n_in_r = sum(r["in_scope"] and r["refusal_reason"] == why for r in rows)
    n_out_r = sum(not r["in_scope"] and r["refusal_reason"] == why for r in rows)
    print(f"  {why:17s} in-scope {n_in_r:2d}   out-of-scope {n_out_r:2d}")

print("\nOut-of-scope questions that were ANSWERED (read these for hallucinations):")
for r in rows:
    if not r["in_scope"] and not r["refused_by_model_or_guards"]:
        print(f"- ({r['top_score']}) {r['question']}\n  {r['answer']}")
print("\nFull answers: pipeline_eval_results.csv")