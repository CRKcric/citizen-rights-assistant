from retrieve import retrieve

# (question, acceptable chunk ids). Empty list = should be refused
tests = [
    ("How do I file an RTI request?", ["rti_act_s6_1"]),
    ("In which languages can I submit an RTI application?", ["rti_act_s6_1_language"]),
    ("Do I need to give a reason for requesting information?", ["rti_act_s6_2"]),
    ("What happens if my application goes to the wrong department?", ["rti_act_s6_3"]),
    ("How long does a Public Information Officer have to respond to my RTI?", ["rti_act_s7_1"]),
    ("What's the deadline if my request concerns life or liberty?", ["rti_act_s7_1_life"]),
    ("What happens if I don't get a response in time?", ["rti_act_s7_2", "rti_act_s19_1"]),
    ("When am I not required to pay a fee for the information?", ["rti_act_s7_5", "rti_act_s7_6"]),
    ("Can the government refuse to give me information about national security?", ["rti_act_s8_1a"]),
    ("Is my personal information protected from being disclosed to others under RTI?", ["rti_act_s8_1j"]),
    ("Can I get cabinet meeting records through RTI?", ["rti_act_s8_1i"]),
    ("Can old government records be withheld forever?", ["rti_act_s8_3"]),
    ("What do I do if my RTI request is rejected?", ["rti_act_s19_1", "rti_act_s7_8"]),
    ("How long do I have to file a first appeal?", ["rti_act_s19_1"]),
    ("What's the difference between a first and second appeal?", ["rti_act_s19_1", "rti_act_s19_3"]),
    ("What penalty does an officer face for delaying my RTI response?", ["rti_act_s20_1"]),
    ("Who has to prove the denial was justified in an appeal?", ["rti_act_s19_5"]),
    ("How do I file a complaint against a government department?", ["cpgrams_filing_1"]),
    ("Can I file a grievance through WhatsApp?", ["cpgrams_scope_1"]),
    ("Who handles my grievance after I submit it?", ["cpgrams_resolution_process_1", "cpgrams_filing_1"]),
    ("How long does it take for a grievance to be resolved?", ["cpgrams_timeline_1", "cpgrams_resolution_process_1"]),
    ("How can I check the status of my complaint?", ["cpgrams_tracking_1"]),
    ("What happens if my grievance takes longer than expected?", ["cpgrams_resolution_process_1"]),
    ("What do I do if I'm not satisfied with how my grievance was resolved?",
     ["cpgrams_feedback_reopen_1", "cpgrams_appeal_trigger_1", "cpgrams_escalation_1"]),
    ("How long does an appeal on CPGRAMS take?", ["cpgrams_escalation_1"]),
    ("What's the weather today?", []),
    ("Can you help me write a resume?", []),
    ("What's the punishment for theft under IPC?", []),
    ("How do I apply for a passport?", []),
    ("Can RTI be used to settle a property dispute?", []),
]

top1_hits = top3_hits = in_scope = 0
in_scores, out_scores = [], []

for q, expected in tests:
    results = retrieve(q, top_k=3)
    top_score = results[0]["score"]
    ids = [r["id"] for r in results]
    if expected:
        in_scope += 1
        in_scores.append(top_score)
        hit1 = ids[0] in expected
        hit3 = any(i in expected for i in ids)
        top1_hits += hit1
        top3_hits += hit3
        status = "TOP1" if hit1 else ("TOP3" if hit3 else "MISS")
        print(f"[{status}] {top_score:.3f} | {q}")
        if not hit1:
            print(f"        got: {ids}  wanted: {expected}")
    else:
        out_scores.append(top_score)
        print(f"[OUT ] {top_score:.3f} | {q}")

print(f"\nIn-scope: top1 {top1_hits}/{in_scope}, top3 {top3_hits}/{in_scope}")
print(f"In-scope top scores: min {min(in_scores):.3f}")
print(f"Out-of-scope top scores: max {max(out_scores):.3f}")