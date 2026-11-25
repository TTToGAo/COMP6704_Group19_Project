"""Check D's archived Apple M6 aggregate tables. No raw training logs are in this archive."""
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent

def rows(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

pairs = rows("timing_pairs.csv")
summary = {row["algorithm"]: row for row in rows("timing_summary.csv")}
audit = json.loads((ROOT / "timing_audit.json").read_text())
assert len(pairs) == 5 and set(summary) == {"second_order_smo", "csmo"}
assert [row["first_mode"] for row in pairs] == ["second_order_smo", "csmo", "second_order_smo", "csmo", "second_order_smo"]
for name in summary:
    times = [float(row[f"{name}_train_seconds"]) for row in pairs]
    iterations = [int(row[f"{name}_iterations"]) for row in pairs]
    assert abs(median(times) - float(summary[name]["train_seconds_median"])) < 1e-12
    assert median(iterations) == int(summary[name]["iterations_median"])
    assert all(row[f"{name}_stop_reason"] == "converged" for row in pairs)
    assert all(float(row[f"{name}_kkt_violation"]) <= 0.001 for row in pairs)
    key = "full_a0" if name == "second_order_smo" else "full_a1"
    assert audit[key]["float64_recomputation"]["kkt_violation"] <= 0.001
ratios = []
for row in pairs:
    ratio = float(row["second_order_smo_train_seconds"]) / float(row["csmo_train_seconds"])
    assert abs(ratio - float(row["smo_over_csmo"])) < 1e-12
    ratios.append(ratio)
print(f"Verified five archived pairs; median SMO/CSMO ratio = {median(ratios):.9f}")
print("This validates the aggregate tables, not the missing original executable or raw training logs.")
