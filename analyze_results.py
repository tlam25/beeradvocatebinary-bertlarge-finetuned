import os
import json
import glob

import numpy as np
import pandas as pd

RESULTS_DIR = "./results"
ASPECTS     = ["appearance", "aroma", "palate", "taste"]
METRICS     = ["accuracy", "precision", "recall", "f1"]   # precision/recall/f1 are macro


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "seed_*_*.json")))
    if not files:
        print(f"No result files found in {RESULTS_DIR}/")
        return

    records = []
    for p in files:
        with open(p) as f:
            d = json.load(f)
        rec = {"seed": d["seed"], "aspect": d["aspect"],
               **{m: d["test"][m] for m in METRICS}}
        records.append(rec)
    df = pd.DataFrame(records).sort_values(["aspect", "seed"]).reset_index(drop=True)

    print(f"Loaded {len(df)} runs from {RESULTS_DIR}/\n")
    print("Per-run results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # Build summary
    rows = []
    for asp in ASPECTS:
        sub = df[df["aspect"] == asp]
        if len(sub) == 0:
            continue
        row = {"aspect": asp, "n_seeds": len(sub)}
        for m in METRICS:
            vals = sub[m].values
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)
    print("Summary (raw, 0-1 scale):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print("Formatted (mean ± std, % scale):")
    for _, r in summary.iterrows():
        print(f"\n[{r['aspect'].upper()}]  ({int(r['n_seeds'])} seeds)")
        for m in METRICS:
            mean = r[f"{m}_mean"] * 100
            std  = r[f"{m}_std"]  * 100
            print(f"  {m:10s}: {mean:6.2f} ± {std:.2f}")

    out_csv = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}")

    out_csv_runs = os.path.join(RESULTS_DIR, "all_runs.csv")
    df.to_csv(out_csv_runs, index=False)
    print(f"  wrote {out_csv_runs}")


if __name__ == "__main__":
    main()