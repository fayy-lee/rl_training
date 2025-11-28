import pandas as pd
from pathlib import Path

eval_file = Path("evaluation_results.csv")
if not eval_file.exists():
    raise FileNotFoundError("Run evaluate_sweep.py first.")

df = pd.read_csv(eval_file)

# Pivot table for report
table = df.pivot_table(
    index=["env","buffer","batch"],
    values=["mean_reward","std_reward"],
    aggfunc=["mean","std"]
)

# Save
table_file = "evaluation_table.csv"
table.to_csv(table_file)
print("Saved result table:", table_file)
print(table)
