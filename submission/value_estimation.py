"""Empirical conditional-value MSE for independent samples vs pure copies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BUDGETS = (1, 2, 4, 8, 16, 32, 64)


def load_groups(path: str | Path):
    groups = []
    with Path(path).open() as f:
        for line in f:
            task = json.loads(line)
            grouped = {}
            for row in task["rollout_lineage"]:
                key = (row["action"], bool(row["hypothesis_value"]))
                grouped.setdefault(key, []).append((int(row["draw_id"]), float(row["value"])))
            for key, rows in grouped.items():
                rows = sorted(rows)
                if len(rows) != 64 or [x[0] for x in rows] != list(range(64)):
                    raise ValueError(f"expected draws 0..63 for {task['task_id']} {key}")
                groups.append(np.asarray([x[1] for x in rows], dtype=float))
    return groups


def summarize(path: str | Path, repeats: int = 2000, seed: int = 20260928) -> pd.DataFrame:
    groups = load_groups(path)
    rng = np.random.default_rng(seed)
    rows = []
    for budget in BUDGETS:
        independent_errors = []
        duplicate_errors = []
        for values in groups:
            target = float(values.mean())
            indices = rng.integers(0, len(values), size=(repeats, budget))
            independent_errors.extend((values[indices].mean(axis=1) - target) ** 2)
            duplicate_errors.append(float((values[0] - target) ** 2))
        rows.append({
            "condition": "independent_rollout",
            "budget": budget,
            "groups": len(groups),
            "mse": float(np.mean(independent_errors)),
            "rmse": float(np.sqrt(np.mean(independent_errors))),
            "repeats": repeats,
            "seed": seed,
        })
        rows.append({
            "condition": "pure_duplication",
            "budget": budget,
            "groups": len(groups),
            "mse": float(np.mean(duplicate_errors)),
            "rmse": float(np.sqrt(np.mean(duplicate_errors))),
            "repeats": 1,
            "seed": seed,
        })
    return pd.DataFrame(rows)


def main() -> None:
    root = Path("results_submission")
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    result = summarize(root / "scaling_benchmark/test.jsonl")
    result.to_csv(out / "value_estimation_mse.csv", index=False)
    print(f"wrote {out / 'value_estimation_mse.csv'} ({len(result)} rows)")


if __name__ == "__main__":
    main()
