#!/usr/bin/env python3
"""One-shot script: add missing instructor qualifications for 7 bottleneck courses.

Courses fixed:
  CT654 practical (0 qualified) → I72, I112
  EX654 practical (0 qualified) → I103, I95
  AM654 theory (insufficient)   → I5
  CT753 theory (insufficient)   → I76
  IE651 theory (insufficient)   → I192
  IE653 theory (insufficient)   → I193
  SH653 theory (insufficient)   → I225
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "Instructors.json"

ADDITIONS = [
    # (instructor_id, coursecode, coursetype)
    # CT654 practical: 0 qualified → add CT dept instructors
    ("I72",  "CT654", "Practical"),   # Pravin Sangroula (full-time, teaches CT651)
    ("I112", "CT654", "Practical"),   # Rajad Shakya (part-time 5 days, teaches CT652)
    # EX654 practical: 0 qualified → add EX dept instructors
    ("I103", "EX654", "Practical"),   # Shreekar Tiwari (part-time 5 days, teaches EX653)
    ("I95",  "EX654", "Practical"),   # Suramya Sharma Dahal (part-time 3 days, teaches EX656)
    # AM654 theory: 1 qualified (I239 part-time Sun, supply=3 < demand=4)
    ("I5",   "AM654", "Theory"),      # Bishal Kumar (full-time, teaches AM651/652)
    # CT753 theory: 1 qualified (I249 part-time Fri/Tue, supply=3 < demand=4)
    ("I76",  "CT753", "Theory"),      # Praches Acharya (full-time, teaches CT751)
    # IE651 theory: 1 qualified (I199 part-time Sun, supply=2 < demand=3)
    ("I192", "IE651", "Theory"),      # Aayush Pudasaini (part-time 3 days, teaches IE654)
    # IE653 theory: 1 qualified (I201 part-time Tue, supply=1 < demand=2)
    ("I193", "IE653", "Theory"),      # Akhalesh Yadav (part-time 5 days, teaches IE655)
    # SH653 theory: 1 qualified (I236 part-time Sun, supply=1 < demand=2)
    ("I225", "SH653", "Theory"),      # Dr. Shree Hari Thapa (full-time, teaches SH652)
]


def main() -> None:
    with open(DATA) as f:
        data = json.load(f)

    by_id = {inst["id"]: inst for inst in data}

    for iid, cc, ct in ADDITIONS:
        inst = by_id[iid]
        existing = {(q["coursecode"], q["coursetype"]) for q in inst["courses"]}
        if (cc, ct) not in existing:
            inst["courses"].append({"coursecode": cc, "coursetype": ct})
            print(f"  Added {cc}/{ct} → {iid} ({inst['name']})")
        else:
            print(f"  Already exists: {cc}/{ct} in {iid}")

    with open(DATA, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {DATA}")


if __name__ == "__main__":
    main()
