"""Choose the refusal threshold from Member A's evaluation scores (Week 3).

1. Ask Member A for the TOP-1 retrieval score of every evaluation question.
2. Paste them below (currently Member A's first evaluation run; update after re-runs).
3. Run:  python pick_threshold.py

A question is ANSWERED when top-1 score >= threshold, otherwise REFUSED.
  false answer  = out-of-scope question that would be answered (bad: hallucination risk)
  false refusal = in-scope question that would be refused (annoying, but safe)
"""

# Member A's FINAL retrieval run (56 chunks): top-1 scores from eval.py, 25 in-scope, 5 out-of-scope
IN_SCOPE = [0.898, 0.819, 0.783, 0.714, 0.860, 0.692, 0.590, 0.775, 0.716, 0.835, 0.707, 0.661, 0.714,
            0.808, 0.717, 0.714, 0.840, 0.686, 0.723, 0.761, 0.915, 0.671, 0.784, 0.736, 0.828]
OUT_OF_SCOPE = [0.445, 0.530, 0.646, 0.586, 0.684]

# Wrongly answering is worse than wrongly refusing for this project.
FALSE_ANSWER_WEIGHT = 3

scores = sorted(set(IN_SCOPE + OUT_OF_SCOPE))
# candidate thresholds: midpoints between neighbouring scores
candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])]

print(f"in-scope:     min={min(IN_SCOPE):.2f}  max={max(IN_SCOPE):.2f}  (n={len(IN_SCOPE)})")
print(f"out-of-scope: min={min(OUT_OF_SCOPE):.2f}  max={max(OUT_OF_SCOPE):.2f}  (n={len(OUT_OF_SCOPE)})")
if min(IN_SCOPE) > max(OUT_OF_SCOPE):
    print("Clean gap: every in-scope score is above every out-of-scope score.")
else:
    print("Overlap: some in-scope and out-of-scope scores share a range.")
print()

print("threshold  false_answers  false_refusals  cost")
best = None
for t in candidates:
    false_answers = sum(s >= t for s in OUT_OF_SCOPE)
    false_refusals = sum(s < t for s in IN_SCOPE)
    cost = FALSE_ANSWER_WEIGHT * false_answers + false_refusals
    print(f"{t:9.3f}  {false_answers:13d}  {false_refusals:14d}  {cost:4d}")
    if best is None or cost < best[0]:
        best = (cost, t)

print()
print(f"Lowest-cost threshold: {best[1]:.3f}")
print("With few out-of-scope questions, treat this as a starting point:")
print("re-check it on 30+ tricky questions in Week 4 and note it in your test log.")