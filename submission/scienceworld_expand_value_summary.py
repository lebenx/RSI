"""Aggregate the fixed-grid algorithmic grouped-value ScienceWorld null check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_rows(flat_dirs, value_dirs):
    rows = []
    for directory in flat_dirs:
        rows.extend(x for x in json.loads((Path(directory) / "summaries.json").read_text())
                    if x.get("method") == "flat" and x.get("error") is None)
    for directory in value_dirs:
        rows.extend(x for x in json.loads((Path(directory) / "summaries.json").read_text())
                    if x.get("method") == "provenance_value" and x.get("error") is None)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_submission/report")
    parser.add_argument("--flat", nargs="*", default=["results_submission/scienceworld_expand_m1", "results_submission/scienceworld_expand_m4"])
    parser.add_argument("--value", nargs="*", default=["results_submission/scienceworld_expand_value_m1", "results_submission/scienceworld_expand_value_m4"])
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    frame = load_rows(args.flat, args.value)
    frame.to_csv(out / "scienceworld_expand_value_rows.csv", index=False)
    summary = frame.groupby(["duplication", "method"], as_index=False).agg(
        n=("task", "size"), success=("success", "mean"), final_score=("final_score", "mean"),
        reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"))
    summary.to_csv(out / "scienceworld_expand_value_summary.csv", index=False)
    pairs = []
    rng = np.random.default_rng(20261030)
    for duplication, group in frame.groupby("duplication"):
        wide = group.pivot(index=["task", "variation"], columns="method",
                           values=["success", "final_score", "reward", "steps"])
        if not {"flat", "provenance_value"}.issubset(wide.columns.get_level_values(1)):
            continue
        for metric in ("success", "final_score", "reward", "steps"):
            values = (wide[(metric, "provenance_value")] - wide[(metric, "flat")]).dropna().to_numpy(float)
            draws = rng.integers(0, len(values), size=(20000, len(values)))
            means = values[draws].mean(axis=1)
            pairs.append({"duplication": int(duplication), "metric": metric, "n": len(values),
                          "provenance_value_minus_flat": float(values.mean()),
                          "ci95_low": float(np.quantile(means, .025)),
                          "ci95_high": float(np.quantile(means, .975)), "bootstrap_draws": 20000})
    pd.DataFrame(pairs).to_csv(out / "scienceworld_expand_value_paired.csv", index=False)
    meta = {"episodes": int(len(frame)), "methods": sorted(frame.method.unique().tolist()),
            "duplications": sorted(frame.duplication.unique().astype(int).tolist()),
            "tasks": sorted(frame.task.unique().tolist()), "variations": sorted(frame.variation.unique().astype(int).tolist()),
            "readout": "algorithmic unique-sample conditional-value mixture"}
    (out / "scienceworld_expand_value_metadata.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(index=False)); print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    main()
