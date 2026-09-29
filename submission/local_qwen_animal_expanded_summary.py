"""Summarize the expanded CUDA Qwen find-animal control.

Only variation-method-budget cells with valid completions are retained in the
row ledger.  Paired estimates use the intersection of variations that have all
four flat/provenance-value cells; schema failures remain in raw_errors.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results_submission")
VARIATIONS = [150, 151, 152, 153, 154, 155]


def load(directory: Path, duplication: int) -> tuple[list[dict], list[dict]]:
    valid, errors = [], []
    for row in json.loads((directory / "summaries.json").read_text()):
        row = dict(row)
        row["duplication"] = duplication
        if row.get("error") is None:
            valid.append(row)
        else:
            errors.append(row)
    return valid, errors


def bootstrap(values, seed=20260929, draws=20000):
    values = np.asarray(values, float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def main() -> None:
    out = Path("results_submission/report/local_qwen_animal_expanded_pilot")
    out.mkdir(parents=True, exist_ok=True)
    valid1, errors1 = load(ROOT / "scienceworld_local_qwen_animal150_155_v9_m1", 1)
    valid4, errors4 = load(ROOT / "scienceworld_local_qwen_animal150_155_v9_m4", 4)
    valid = pd.DataFrame(valid1 + valid4)
    if len(valid):
        valid.to_csv(out / "rows.csv", index=False)
    errors = pd.DataFrame(errors1 + errors4)
    errors.to_csv(out / "raw_errors.csv", index=False)

    # Keep only complete four-cell episode blocks in paired analysis.
    key = ["task", "variation"]
    cells = valid.groupby(key).apply(lambda g: set(zip(g.method, g.duplication)), include_groups=False)
    required = {("flat", 1), ("flat", 4), ("provenance_value", 1), ("provenance_value", 4)}
    paired_variations = [k for k, present in cells.items() if required <= present]
    paired_rows = valid[valid.variation.isin([v[1] for v in paired_variations])].copy()
    wide = paired_rows.pivot_table(index=key, columns=["method", "duplication"], values=["success", "final_score", "reward", "steps", "charged_tokens"], aggfunc="first")
    paired = []
    for (task, variation), row in wide.iterrows():
        rec = {"task": task, "variation": int(variation)}
        for metric in ("success", "final_score", "reward", "steps", "charged_tokens"):
            for dup in (1, 4):
                rec[f"flat_m{dup}_{metric}"] = row.get((metric, "flat", dup))
                rec[f"provenance_value_m{dup}_{metric}"] = row.get((metric, "provenance_value", dup))
                rec[f"provenance_value_minus_flat_m{dup}_{metric}"] = rec[f"provenance_value_m{dup}_{metric}"] - rec[f"flat_m{dup}_{metric}"]
        paired.append(rec)
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)

    contrasts = []
    for metric in ("success", "reward", "steps"):
        for dup in (1, 4):
            values = paired[f"provenance_value_minus_flat_m{dup}_{metric}"].dropna().to_numpy(float)
            est, lo, hi = bootstrap(values)
            contrasts.append({"duplication": dup, "metric": metric, "n_episodes": len(values), "estimate": est, "ci95_low": lo, "ci95_high": hi, "bootstrap_draws": 20000})
    pd.DataFrame(contrasts).to_csv(out / "bootstrap.csv", index=False)

    # Summary is restricted to the paired rows so n is directly interpretable.
    summary = paired_rows.groupby(["duplication", "method"], as_index=False).agg(n=("variation", "size"), success=("success", "mean"), final_score=("final_score", "mean"), reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"))
    summary.to_csv(out / "summary.csv", index=False)
    meta = {
        "protocol": "local-qwen-find-animal-expanded-v1", "model": "Qwen2.5-Coder-3B-Instruct",
        "task": "find-animal", "requested_variations": VARIATIONS,
        "paired_variations": [int(v[1]) for v in paired_variations], "duplications": [1, 4],
        "methods": ["flat", "provenance_value"], "valid_rows": int(len(valid)),
        "error_rows": int(len(errors)), "paired_episodes": int(len(paired)),
        "planner_claim": False, "cuda_run_policy": "LOCAL_QWEN_DEVICE=cuda; RTX 4090 FP16",
        "interpretation": "second-task-family exploratory control; incomplete variation blocks are retained but excluded from paired estimates",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False)); print(summary.to_string(index=False)); print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
