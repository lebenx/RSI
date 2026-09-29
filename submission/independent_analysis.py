"""Summaries for the independent-rollout value-information axis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def summarize(raw_path: str | Path, draws: int = 20_000, seed: int = 20260928):
    raw = pd.read_csv(raw_path)
    independent = raw[raw.condition == "independent_rollout"].copy()
    summary = independent.groupby(["budget", "method"], as_index=False).agg(
        n=("task_id", "size"),
        reward=("reward", "mean"),
        regret=("regret", "mean"),
        action_correct=("action_correct", "mean"),
        root_confidence=("root_confidence", "mean"),
    )

    methods = ["independent_trajectory", "mean_pooling", "flat_rollout", "provenance_preserving"]
    summary = summary[summary.method.isin(methods)].copy()
    rng = np.random.default_rng(seed)
    pairs = []
    for budget, group in independent.groupby("budget", sort=True):
        wide = group.pivot_table(index="task_id", columns="method", values=["reward", "regret", "action_correct"], aggfunc="first")
        if "independent_trajectory" not in wide.columns.get_level_values("method"):
            continue
        for baseline in ("flat_rollout", "provenance_preserving"):
            if baseline not in wide.columns.get_level_values("method"):
                continue
            for metric in ("reward", "regret", "action_correct"):
                a = wide[(metric, "independent_trajectory")]
                b = wide[(metric, baseline)]
                delta = (a - b).dropna().to_numpy(dtype=float)
                if not len(delta):
                    continue
                indices = rng.integers(0, len(delta), size=(draws, len(delta)))
                means = delta[indices].mean(axis=1)
                pairs.append({
                    "budget": int(budget),
                    "contrast": f"independent_trajectory-{baseline}",
                    "metric": metric,
                    "n_tasks": len(delta),
                    "estimate": float(delta.mean()),
                    "ci95_low": float(np.quantile(means, .025)),
                    "ci95_high": float(np.quantile(means, .975)),
                    "bootstrap_draws": int(draws),
                    "seed": int(seed),
                })
    return summary, pd.DataFrame(pairs)


def main() -> None:
    root = Path("results_submission")
    summary, pairs = summarize(root / "scaling/baseline_raw.csv")
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "independent_scaling.csv", index=False)
    pairs.to_csv(out / "independent_paired.csv", index=False)
    print(f"wrote {len(summary)} summary rows and {len(pairs)} paired rows")


if __name__ == "__main__":
    main()
