import pandas as pd
from pathlib import Path

# Find latest output
latest = max(Path("output").glob("evaluation_*"), key=lambda p: p.stat().st_mtime)
metrics_file = latest / "data" / "metrics.csv"

if metrics_file.exists():
    df = pd.read_csv(metrics_file)
    last_row = df.iloc[-1]
    print(
        f'Gen {int(last_row["generation"])}: Hard={last_row["best_hard_violations"]:.0f}'
    )

    # Show constraint breakdown
    hard_cols = [c for c in df.columns if c.startswith("best_hard_")]
    for col in hard_cols:
        val = last_row[col]
        if val > 0:
            constraint_name = col.replace("best_hard_", "")
            print(f"  {constraint_name}: {val:.0f}")
else:
    print(f"No metrics file found in {latest}")
