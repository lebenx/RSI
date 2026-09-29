"""Build a task-family-stratified local-Qwen interactive main table.

The table deliberately uses the core algorithmic provenance-value method.  It
combines the five complete plant pairs with the three complete animal pairs;
unpaired animal cells and schema failures stay in their source audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results_submission/report")


def bootstrap(values, seed=20260929, draws=20000):
    values = np.asarray(values, float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def main() -> None:
    out = ROOT / "local_qwen_interactive_main_table"
    out.mkdir(parents=True, exist_ok=True)
    plant = pd.read_csv(ROOT / "local_qwen_value_replay/rows.csv")
    plant = plant[plant.method.isin(["flat", "provenance_value"])].copy()
    animal = pd.read_csv(ROOT / "local_qwen_animal_expanded_pilot/rows.csv")
    animal = animal[animal.variation.astype(int).isin([150, 153, 155]) & animal.method.isin(["flat", "provenance_value"])].copy()
    frame = pd.concat([plant, animal], ignore_index=True, sort=False)
    frame["task_family"] = frame.task
    frame["unnecessary_actions"] = frame["repeated_no_visible_change"].astype(float)
    frame["calibration_brier"] = (frame["initial_success_probability"] - frame["success"]) ** 2
    frame.to_csv(out / "rows.csv", index=False)

    summary = frame.groupby(["task_family", "duplication", "method"], as_index=False).agg(
        n=("variation", "size"), success=("success", "mean"), reward=("reward", "mean"),
        final_score=("final_score", "mean"), regret_proxy=("final_score", lambda x: float(100 - np.mean(x))),
        steps=("steps", "mean"), unnecessary_actions=("unnecessary_actions", "mean"),
        calibration_brier=("calibration_brier", "mean"), tokens=("charged_tokens", "mean"))
    summary.to_csv(out / "summary.csv", index=False)

    # Complete paired blocks only.
    paired_rows = []
    for (family, variation), group in frame.groupby(["task_family", "variation"]):
        required = {("flat", 1), ("flat", 4), ("provenance_value", 1), ("provenance_value", 4)}
        present = set(zip(group.method, group.duplication))
        if not required <= present:
            continue
        wide = group.set_index(["method", "duplication"])
        rec = {"task_family": family, "variation": int(variation)}
        for metric in ("success", "reward", "final_score", "steps", "unnecessary_actions", "calibration_brier", "charged_tokens"):
            for dup in (1, 4):
                rec[f"flat_m{dup}_{metric}"] = float(wide.loc[("flat", dup), metric])
                rec[f"provenance_value_m{dup}_{metric}"] = float(wide.loc[("provenance_value", dup), metric])
                rec[f"provenance_value_minus_flat_m{dup}_{metric}"] = rec[f"provenance_value_m{dup}_{metric}"] - rec[f"flat_m{dup}_{metric}"]
        paired_rows.append(rec)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired.csv", index=False)

    contrasts = []
    for family in sorted(paired.task_family.unique()):
        subset = paired[paired.task_family == family]
        for dup in (1, 4):
            for metric in ("success", "reward", "final_score", "steps", "unnecessary_actions", "calibration_brier", "charged_tokens"):
                values = subset[f"provenance_value_minus_flat_m{dup}_{metric}"].to_numpy(float)
                est, low, high = bootstrap(values)
                contrasts.append({"task_family": family, "duplication": dup, "metric": metric,
                                  "n_episodes": len(values), "estimate": est, "ci95_low": low,
                                  "ci95_high": high, "bootstrap_draws": 20000})
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(out / "contrasts.csv", index=False)

    # Compact paper-facing figure.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ("success", "reward"), ("Episode success", "Mean reward")):
        pivot = summary.pivot_table(index=["task_family", "duplication"], columns="method", values=metric)
        labels = [f"{f.replace('find-', '')}\nm={d}" for f, d in pivot.index]
        x = np.arange(len(labels)); width = .36
        for i, method in enumerate(["flat", "provenance_value"]):
            ax.bar(x + (i - .5) * width, pivot[method].to_numpy(), width, label=method)
        ax.set_xticks(x, labels); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    axes[1].legend(fontsize=8)
    fig.savefig(out / "curve.png", dpi=160); plt.close(fig)

    metadata = {
        "protocol": "local-qwen-interactive-main-table-v1", "model": "Qwen2.5-Coder-3B-Instruct",
        "task_families": sorted(frame.task_family.unique().tolist()), "methods": ["flat", "provenance_value"],
        "duplications": [1, 4], "valid_rows": int(len(frame)), "paired_episodes": int(len(paired)),
        "plant_paired_episodes": int((paired.task_family == "find-plant").sum()),
        "animal_paired_episodes": int((paired.task_family == "find-animal").sum()),
        "metrics": ["success", "reward", "final_score", "regret_proxy", "steps", "unnecessary_actions", "calibration_brier", "tokens"],
        "planner_claim": False, "interpretation": "stratified exploratory second-model main table; incomplete animal blocks excluded",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False)); print(summary.to_string(index=False)); print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
