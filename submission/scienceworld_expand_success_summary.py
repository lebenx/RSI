"""Summarize the fixed-grid success-aware provenance ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_dirs(dirs, methods):
    rows = []
    for directory in dirs:
        rows.extend(x for x in json.loads((Path(directory) / "summaries.json").read_text())
                    if x.get("method") in methods and x.get("error") is None)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_submission/report")
    parser.add_argument("--flat", nargs="*", default=["results_submission/scienceworld_expand_m1", "results_submission/scienceworld_expand_m4"])
    parser.add_argument("--success", nargs="*", default=["results_submission/scienceworld_expand_success_m1", "results_submission/scienceworld_expand_success_m4"])
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([read_dirs(args.flat, {"flat"}), read_dirs(args.success, {"provenance_success"})], ignore_index=True)
    frame.to_csv(out / "scienceworld_expand_success_rows.csv", index=False)
    summary = frame.groupby(["duplication", "method"], as_index=False).agg(
        n=("task", "size"), success=("success", "mean"), final_score=("final_score", "mean"),
        reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"))
    summary.to_csv(out / "scienceworld_expand_success_summary.csv", index=False)
    pairs = []
    rng = np.random.default_rng(20261031)
    for duplication, group in frame.groupby("duplication"):
        wide = group.pivot(index=["task", "variation"], columns="method",
                           values=["success", "final_score", "reward", "steps"])
        if not {"flat", "provenance_success"}.issubset(wide.columns.get_level_values(1)):
            continue
        for metric in ("success", "final_score", "reward", "steps"):
            values = (wide[(metric, "provenance_success")] - wide[(metric, "flat")]).dropna().to_numpy(float)
            draws = rng.integers(0, len(values), size=(20000, len(values)))
            boot = values[draws].mean(axis=1)
            pairs.append({"duplication": int(duplication), "metric": metric, "n": len(values),
                          "provenance_success_minus_flat": float(values.mean()),
                          "ci95_low": float(np.quantile(boot, .025)),
                          "ci95_high": float(np.quantile(boot, .975)), "bootstrap_draws": 20000})
    pd.DataFrame(pairs).to_csv(out / "scienceworld_expand_success_paired.csv", index=False)
    meta = {"valid_rows": int(len(frame)), "methods": sorted(frame.method.unique().tolist()),
            "duplications": sorted(frame.duplication.unique().astype(int).tolist()),
            "matched_rows_per_duplication": int(sum(1 for _ in [])),
            "tasks": sorted(frame.task.unique().tolist()), "variations": sorted(frame.variation.unique().astype(int).tolist()),
            "selection_rule": "all fixed-grid valid rows; paired contrasts use the method intersection",
            "objective": "unique-sample conditional success probability"}
    pairs_frame = pd.DataFrame(pairs)
    if not pairs_frame.empty:
        meta["matched_rows_per_duplication"] = int(pairs_frame[pairs_frame.metric == "success"].n.iloc[0])
    (out / "scienceworld_expand_success_metadata.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(index=False)); print(pairs_frame.to_string(index=False))


if __name__ == "__main__":
    main()
