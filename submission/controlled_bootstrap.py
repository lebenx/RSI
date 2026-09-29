"""Task-cluster bootstrap for the frozen controlled duplication bank."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "root_confidence": "root_confidence",
    "reward": "reward",
    "regret": "regret",
    "action_flip_rate": "flip_from_m1",
}


def bootstrap(raw_path: str | Path, draws: int = 20_000, seed: int = 20260928) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    base = raw[raw["condition"] == "pure_duplication"].copy()
    reference = (
        base[base["budget"] == 1]
        .set_index(["task_id", "method"])["action"]
        .rename("action_m1")
    )
    base = base.join(reference, on=["task_id", "method"])
    base["flip_from_m1"] = (base["action"] != base["action_m1"]).astype(float)

    rng = np.random.default_rng(seed)
    rows = []
    for (budget, method), group in base.groupby(["budget", "method"], sort=True):
        # The frozen bank has one row per task for each budget/method. Assert
        # that cluster resampling cannot silently weight a task twice.
        task_rows = group.groupby("task_id", as_index=False)[list(METRICS.values())].mean()
        values = task_rows[list(METRICS.values())].to_numpy(dtype=float)
        n = len(values)
        if n == 0:
            continue
        indices = rng.integers(0, n, size=(draws, n))
        sampled = values[indices].mean(axis=1)
        for column, metric in METRICS.items():
            j = list(METRICS.values()).index(metric)
            rows.append(
                {
                    "condition": "pure_duplication",
                    "budget": int(budget),
                    "method": method,
                    "metric": column,
                    "n_tasks": n,
                    "estimate": float(values[:, j].mean()),
                    "ci95_low": float(np.quantile(sampled[:, j], 0.025)),
                    "ci95_high": float(np.quantile(sampled[:, j], 0.975)),
                    "bootstrap_draws": int(draws),
                    "seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results_submission/scaling/baseline_raw.csv")
    parser.add_argument("--output", default="results_submission/report/controlled_bootstrap.csv")
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    result = bootstrap(args.raw, args.draws, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
