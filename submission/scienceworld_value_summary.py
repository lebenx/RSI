"""Aggregate the cached ScienceWorld grouped-value exploratory runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUTS = (
    "results_submission/scienceworld_value_m1",
    "results_submission/scienceworld_value_retry_m1",
    "results_submission/scienceworld_value_retry_161162_m1",
    "results_submission/scienceworld_value_plants_m4",
)


def load_rows(inputs, tasks=("find-plant",), variations=(152, 158, 161, 162)):
    rows = []
    for directory in inputs:
        path = Path(directory) / "summaries.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            if row.get("error") is not None:
                continue
            if row.get("method") not in {"flat", "provenance_value"}:
                continue
            if row.get("task") not in tasks or int(row.get("variation")) not in set(variations):
                continue
            rows.append({**row, "source_directory": directory})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no successful grouped-value rows")
    # Retry outputs are preferred over the initial exploratory directory.
    priority = frame.source_directory.map({
        "results_submission/scienceworld_value_m1": 0,
        "results_submission/scienceworld_value_retry_m1": 2,
        "results_submission/scienceworld_value_retry_161162_m1": 2,
        "results_submission/scienceworld_value_plants_m4": 3,
    }).fillna(1)
    frame = frame.assign(_priority=priority).sort_values("_priority")
    frame = frame.drop_duplicates(["task", "variation", "duplication", "method"], keep="last")
    return frame.drop(columns=["_priority"])


def summarize(frame, output, prefix="scienceworld_value"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{prefix}_rows.csv", index=False)
    summary = frame.groupby(["duplication", "method"]).agg(
        n=("task", "size"), success=("success", "mean"), final_score=("final_score", "mean"),
        reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"),
    ).reset_index()
    summary.to_csv(output / f"{prefix}_summary.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
    colors = {"flat": "#c43c39", "provenance_value": "#276fbf"}
    for method in ("flat", "provenance_value"):
        part = summary[summary.method == method].sort_values("duplication")
        axs[0].plot(part.duplication, part.reward, "o-", label=method, color=colors[method])
        axs[1].plot(part.duplication, part.success, "o-", label=method, color=colors[method])
    axs[0].set(xlabel="duplication", ylabel="episode reward")
    axs[1].set(xlabel="duplication", ylabel="success rate")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_curve.png", dpi=180)
    plt.close(fig)
    pairs = []
    rng = np.random.default_rng(20260928)
    for duplication, group in frame.groupby("duplication"):
        pivot = group.pivot(index=["task", "variation"], columns="method", values=["success", "reward", "final_score", "steps"])
        methods = set(pivot.columns.get_level_values(1))
        if not {"flat", "provenance_value"}.issubset(methods):
            continue
        for metric in ("success", "reward", "final_score", "steps"):
            diff = pivot[(metric, "provenance_value")] - pivot[(metric, "flat")]
            values = diff.dropna().to_numpy(dtype=float)
            draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
            pairs.append({"duplication": duplication, "metric": metric, "n": int(len(values)),
                          "provenance_minus_flat": float(values.mean()),
                          "ci95_low": float(np.quantile(draws, .025)),
                          "ci95_high": float(np.quantile(draws, .975))})
    pd.DataFrame(pairs).to_csv(output / f"{prefix}_paired.csv", index=False)
    return summary, pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_submission/report")
    parser.add_argument("--inputs", nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--task", default="find-plant", choices=("find-plant", "find-animal"))
    parser.add_argument("--prefix", default="scienceworld_value")
    args = parser.parse_args()
    if args.task == "find-animal":
        frame = load_rows(args.inputs, tasks=("find-animal",), variations=(153,))
    else:
        frame = load_rows(args.inputs)
    summary, pairs = summarize(frame, args.output, args.prefix)
    print(summary.to_string(index=False))
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
