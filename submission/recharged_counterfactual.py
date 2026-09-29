"""Run the no-new-API ScienceWorld state-replay intervention on fresh runs.

For each public state, the archived flat m=1 action, flat m=4 action, and
unique-lineage grouped action are replayed with the same archived
no-imagination suffix.  The intervention changes only the first-step readout;
it is a causal episode-level diagnostic rather than a closed-loop success-rate
estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from submission.scienceworld_expand_counterfactual import run as replay_run


CASES = {
    "find-plant": (
        "results_submission/scienceworld_recharged_findplant_m1",
        "results_submission/scienceworld_recharged_findplant_m4",
    ),
    "find-animal": (
        "results_submission/scienceworld_recharged_v9_findanimal_m1",
        "results_submission/scienceworld_recharged_v9_findanimal_m4",
    ),
}


def bootstrap(values: np.ndarray, seed: int, draws: int = 20000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sample.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def make_paired(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot_table(
        index=["task", "variation"], columns=["method", "duplication"],
        values=["success", "reward", "final_score", "action"], aggfunc="first",
    )
    rows = []
    for (task, variation), row in wide.iterrows():
        def get(metric, method, duplication):
            key = (metric, method, duplication)
            return row[key] if key in row.index else np.nan
        record = {"task": task, "variation": int(variation)}
        for method, duplication, prefix in (
            ("flat", 1, "flat_m1"), ("flat", 4, "flat_m4"),
            ("provenance_grouped", 1, "provenance_grouped"),
        ):
            for metric in ("success", "reward", "final_score", "action"):
                record[f"{prefix}_{metric}"] = get(metric, method, duplication)
        record["grouped_minus_flat_m4_success"] = record["provenance_grouped_success"] - record["flat_m4_success"]
        record["grouped_minus_flat_m4_reward"] = record["provenance_grouped_reward"] - record["flat_m4_reward"]
        record["flat_m4_minus_flat_m1_success"] = record["flat_m4_success"] - record["flat_m1_success"]
        record["flat_m4_minus_flat_m1_reward"] = record["flat_m4_reward"] - record["flat_m1_reward"]
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["task", "variation"])


def make_bootstrap(paired: pd.DataFrame, draws: int = 20000) -> pd.DataFrame:
    rows = []
    for task, group in paired.groupby("task"):
        for metric in ("success", "reward"):
            for contrast, column in (
                ("provenance_grouped-flat_m4", f"grouped_minus_flat_m4_{metric}"),
                ("flat_m4-flat_m1", f"flat_m4_minus_flat_m1_{metric}"),
            ):
                est, lo, hi = bootstrap(group[column].dropna().to_numpy(float), 20260928 + len(rows), draws)
                rows.append({"task": task, "contrast": contrast, "metric": metric,
                             "n_states": int(group[column].notna().sum()),
                             "estimate": est, "ci95_low": lo, "ci95_high": hi,
                             "bootstrap_draws": draws})
    return pd.DataFrame(rows)


def run_all(output: str | Path, steps: int = 8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task, (m1, m4) in CASES.items():
        task_rows = replay_run(m1, m4, output / task, steps=steps)
        all_rows.extend(task_rows)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "summary.csv", index=False)
    frame.to_json(output / "summary.json", orient="records", indent=2)
    paired = make_paired(frame)
    paired.to_csv(output / "paired.csv", index=False)
    bootstrap_frame = make_bootstrap(paired)
    bootstrap_frame.to_csv(output / "bootstrap.csv", index=False)
    metadata = {
        "provider": "DeepSeek archived traces",
        "tasks": sorted(CASES),
        "variations": [150, 151, 152, 153, 154, 155],
        "states": int(len(paired)),
        "rows": int(len(frame)),
        "methods": ["flat_m1", "flat_m4", "provenance_grouped"],
        "step_limit": steps,
        "common_suffix": "no_imagination archived action suffix",
        "new_api_calls": 0,
        "gold_path": False,
        "hidden_state": False,
        "interpretation": "paired causal state-replay diagnostic; not a broad closed-loop success estimate",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return frame, paired, bootstrap_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_submission/report/scienceworld_recharged_counterfactual_all")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    run_all(args.output, args.steps)


if __name__ == "__main__":
    main()
