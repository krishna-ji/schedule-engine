#!/usr/bin/env python3
"""Parse rl_02_llh_differentiation log into structured output files.

Reads the raw terminal log and produces:
  - llh_differentiation_trajectory.csv  (per-gen, per-action best_hard)
  - llh_differentiation_results.json    (summary + checkpoint table + verdicts)
"""

import csv
import json
import re
import sys
from pathlib import Path

LOG_PATH = Path("output/rl_llh__differentiation.log")
OUT_DIR = Path("output/rl_llh_differentiation")

ACTION_NAMES = {
    0: "Conservative",
    1: "Aggressive",
    2: "Memetic",
    3: "SoftFocus",
    4: "Destructive",
    5: "Intensified",
}


def parse_log(text: str) -> dict:
    """Extract all structured data from the log text."""

    # --- Per-action summary lines: " hard=73  soft=436  mean_hard=285  1866.9s"
    action_pattern = re.compile(
        r"Running Action (\d) \((\w+)\).*?"
        r"hard=(\d+)\s+soft=(\d+)\s+mean_hard=(\d+)\s+([\d.]+)s",
        re.DOTALL,
    )
    actions = {}
    for m in action_pattern.finditer(text):
        aid = int(m.group(1))
        actions[aid] = {
            "action_id": aid,
            "name": m.group(2),
            "best_hard": int(m.group(3)),
            "best_soft": int(m.group(4)),
            "mean_hard": int(m.group(5)),
            "time_s": float(m.group(6)),
        }

    # --- Total time
    total_m = re.search(r"Total time:\s+([\d.]+)s", text)
    total_time = float(total_m.group(1)) if total_m else 0.0

    # --- Per-generation trajectory (best_hard)
    traj_section = re.search(
        r"PER-GENERATION TRAJECTORY \(best_hard\)\n=+\n(.+?)\n\n",
        text,
        re.DOTALL,
    )
    trajectory = []
    if traj_section:
        lines = traj_section.group(1).strip().split("\n")
        # Skip header and separator
        for line in lines:
            if "Gen" in line or line.startswith("---"):
                continue
            parts = line.split("<-")[0].split()
            if len(parts) < 7:
                continue
            try:
                gen = int(parts[0])
            except ValueError:
                continue
            vals = [int(x) for x in parts[1:7]]
            winner_match = re.search(r"<-\s+(\w+)", line)
            winner = winner_match.group(1) if winner_match else ""
            trajectory.append(
                {
                    "gen": gen,
                    "Conservative": vals[0],
                    "Aggressive": vals[1],
                    "Memetic": vals[2],
                    "SoftFocus": vals[3],
                    "Destructive": vals[4],
                    "Intensified": vals[5],
                    "winner": winner,
                }
            )

    # --- Checkpoint table
    cp_section = re.search(
        r"CHECKPOINT COMPARISON TABLE\n=+\n(.+?)\n\n",
        text,
        re.DOTALL,
    )
    checkpoints = {}
    if cp_section:
        lines = cp_section.group(1).strip().split("\n")
        for line in lines:
            if "LLH" in line or line.startswith("---"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            name = parts[0]
            try:
                checkpoints[name] = {
                    "hard_5": int(parts[1]),
                    "soft_5": int(parts[2]),
                    "hard_25": int(parts[3]),
                    "soft_25": int(parts[4]),
                    "hard_50": int(parts[5]),
                    "soft_50": int(parts[6]),
                    "total_time": float(parts[7]),
                    "s_per_gen": float(parts[8]),
                }
            except ValueError:
                continue

    # --- Best-ever per LLH
    best_section = re.search(
        r"BEST-EVER HARD PENALTY PER LLH\n=+\n(.+?)\n\n",
        text,
        re.DOTALL,
    )
    best_ever = {}
    if best_section:
        lines = best_section.group(1).strip().split("\n")
        for line in lines:
            if line.startswith("LLH") or line.startswith("---"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                name = parts[0]
                best_ever[name] = {
                    "best_hard": int(parts[1]),
                    "at_gen": int(parts[2]),
                    "best_soft": int(parts[3]),
                    "soft_at_best_hard": int(parts[4]),
                }

    # --- Soft trajectory (sampled gens)
    soft_section = re.search(
        r"PER-GENERATION TRAJECTORY \(best_soft\)\n=+\n(.+?)\n\n",
        text,
        re.DOTALL,
    )
    soft_trajectory = []
    if soft_section:
        lines = soft_section.group(1).strip().split("\n")
        for line in lines:
            if "Gen" in line or line.startswith("---"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                gen = int(parts[0])
            except ValueError:
                continue
            vals = [int(x) for x in parts[1:7]]
            soft_trajectory.append(
                {
                    "gen": gen,
                    "Conservative": vals[0],
                    "Aggressive": vals[1],
                    "Memetic": vals[2],
                    "SoftFocus": vals[3],
                    "Destructive": vals[4],
                    "Intensified": vals[5],
                }
            )

    # --- Verdicts
    verdicts = {}
    for q_num in [1, 2, 3]:
        m = re.search(rf"Q{q_num}.*?>>> ANSWER:\s+(\w+)", text, re.DOTALL)
        if m:
            verdicts[f"Q{q_num}"] = m.group(1)

    final_m = re.search(r"VERDICT:\s+(.+?)$", text, re.MULTILINE)
    verdict_text = final_m.group(1).strip() if final_m else ""

    return {
        "config": {"pop_size": 120, "max_gen": 50, "seed": 42},
        "total_time_s": total_time,
        "actions": actions,
        "trajectory_hard": trajectory,
        "trajectory_soft": soft_trajectory,
        "checkpoints": checkpoints,
        "best_ever": best_ever,
        "verdicts": verdicts,
        "verdict_text": verdict_text,
    }


def main():
    if not LOG_PATH.exists():
        print(f"ERROR: {LOG_PATH} not found")
        sys.exit(1)

    text = LOG_PATH.read_text(encoding="utf-8")
    data = parse_log(text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- CSV: per-generation trajectory
    csv_path = OUT_DIR / "llh_differentiation_trajectory.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "gen",
                "Conservative",
                "Aggressive",
                "Memetic",
                "SoftFocus",
                "Destructive",
                "Intensified",
                "winner",
            ]
        )
        for row in data["trajectory_hard"]:
            w.writerow(
                [
                    row["gen"],
                    row["Conservative"],
                    row["Aggressive"],
                    row["Memetic"],
                    row["SoftFocus"],
                    row["Destructive"],
                    row["Intensified"],
                    row["winner"],
                ]
            )
    print(f"  Wrote {csv_path}  ({len(data['trajectory_hard'])} rows)")

    # --- CSV: soft constraint trajectory
    csv_soft_path = OUT_DIR / "llh_differentiation_soft_trajectory.csv"
    with open(csv_soft_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "gen",
                "Conservative",
                "Aggressive",
                "Memetic",
                "SoftFocus",
                "Destructive",
                "Intensified",
            ]
        )
        for row in data["trajectory_soft"]:
            w.writerow(
                [
                    row["gen"],
                    row["Conservative"],
                    row["Aggressive"],
                    row["Memetic"],
                    row["SoftFocus"],
                    row["Destructive"],
                    row["Intensified"],
                ]
            )
    print(f"  Wrote {csv_soft_path}  ({len(data['trajectory_soft'])} rows)")

    # --- JSON: full results
    json_path = OUT_DIR / "llh_differentiation_results.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {json_path}")

    # --- Copy log into output dir
    log_dest = OUT_DIR / "run.log"
    log_dest.write_text(text, encoding="utf-8")
    print(f"  Copied log → {log_dest}")

    # --- Summary
    print(f"\n  Output directory: {OUT_DIR}")
    print(
        f"  Total run time: {data['total_time_s']:.0f}s ({data['total_time_s']/3600:.1f}h)"
    )
    print("\n  Per-action results:")
    print(f"  {'Action':<16} {'BestHard':>8} {'BestSoft':>8} {'Time':>8}")
    print(f"  {'-'*44}")
    for aid in sorted(data["actions"]):
        a = data["actions"][aid]
        print(
            f"  {a['name']:<16} {a['best_hard']:>8} {a['best_soft']:>8} {a['time_s']:>7.0f}s"
        )
    print(f"\n  Verdicts: {data['verdicts']}")
    print(f"  {data['verdict_text']}")


if __name__ == "__main__":
    main()
