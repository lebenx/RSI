"""Summarize the pre-registered expanded ScienceWorld paired run.

The two input directories contain the same 12 task/variation pairs at
duplication 1 and 4.  Failed episodes stay in the raw rows and are excluded
only from metric means, with valid/error counts reported explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_rows(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads((Path(path) / "summaries.json").read_text()))
    return pd.DataFrame(rows)


def bootstrap(values, seed=20261001, n_boot=20000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return (float(values.mean()), float(np.quantile(means, .025)),
            float(np.quantile(means, .975)))


def summarize_metrics(raw):
    valid = raw[raw.error.isna()].copy()
    valid["regret_proxy"] = 100.0 - valid["final_score"]
    valid["initial_brier"] = (valid["initial_success_probability"].fillna(.5) - valid["success"]) ** 2
    grouped = valid.groupby(["task", "duplication", "method"], as_index=False).agg(
        n_valid=("task", "size"),
        success=("success", "mean"),
        final_score=("final_score", "mean"),
        reward=("reward", "mean"),
        regret_proxy=("regret_proxy", "mean"),
        steps=("steps", "mean"),
        tokens=("charged_tokens", "mean"),
        calibration_brier=("initial_brier", "mean"),
    )
    counts = raw.groupby(["task", "duplication", "method"], as_index=False).agg(
        n_total=("task", "size"), n_errors=("error", lambda x: int(x.notna().sum())))
    return grouped.merge(counts, on=["task", "duplication", "method"], how="outer")


def paired_contrasts(raw):
    valid = raw[raw.error.isna()].copy()
    valid["regret_proxy"] = 100.0 - valid["final_score"]
    rows = []
    for duplication, group in valid.groupby("duplication"):
        wide = group.pivot_table(index=["task", "variation"], columns="method",
                                 values=["success", "final_score", "reward", "regret_proxy"],
                                 aggfunc="first")
        for metric in ("success", "final_score", "reward", "regret_proxy"):
            key_a = (metric, "provenance")
            key_b = (metric, "flat")
            if key_a not in wide or key_b not in wide:
                continue
            diff = (wide[key_a] - wide[key_b]).dropna().to_numpy(dtype=float)
            est, lo, hi = bootstrap(diff, seed=20261010 + int(duplication) + len(metric))
            rows.append({"duplication": int(duplication), "contrast": "provenance-flat",
                         "metric": metric, "n_units": len(diff), "estimate": est,
                         "ci95_low": lo, "ci95_high": hi, "bootstrap_draws": 20000})
    return pd.DataFrame(rows)


def action_flips(paths, out):
    records = []
    for method in ("no_imagination", "flat", "provenance"):
        by_unit = {}
        for duplication, root in ((1, Path(paths[0])), (4, Path(paths[1]))):
            for trace in (root / "episodes").glob(f"*-{method}-m{duplication}-trace.jsonl"):
                stem = trace.name[: -len(f"-{method}-m{duplication}-trace.jsonl")]
                actions = {}
                for line in trace.read_text().splitlines():
                    row = json.loads(line)
                    actions[int(row["outcome"]["step"])] = row["action"]
                by_unit.setdefault(stem, {})[duplication] = actions
        for unit, both in by_unit.items():
            if 1 not in both or 4 not in both:
                continue
            keys = sorted(set(both[1]) & set(both[4]))
            records.append({"method": method, "episode": unit,
                            "flips": sum(both[1][k] != both[4][k] for k in keys),
                            "steps_aligned": len(keys)})
    frame = pd.DataFrame(records)
    frame.to_csv(out / "scienceworld_expand_action_units.csv", index=False)
    rows = []
    if not frame.empty:
        for method, group in frame.groupby("method"):
            num = group.flips.to_numpy(dtype=float)
            den = group.steps_aligned.to_numpy(dtype=float)
            rng = np.random.default_rng(20261020 + len(method))
            draws = rng.integers(0, len(group), size=(20000, len(group)))
            rates = num[draws].sum(axis=1) / den[draws].sum(axis=1)
            rows.append({"method": method, "n_episodes": len(group),
                         "aligned_steps": int(den.sum()),
                         "action_flip_rate": float(num.sum() / den.sum()),
                         "ci95_low": float(np.quantile(rates, .025)),
                         "ci95_high": float(np.quantile(rates, .975)),
                         "bootstrap_draws": 20000})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", default="results_submission/scienceworld_expand_m1")
    parser.add_argument("--m4", default="results_submission/scienceworld_expand_m4")
    parser.add_argument("--output", default="results_submission/report")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_rows([args.m1, args.m4])
    raw.to_csv(out / "scienceworld_expand_rows.csv", index=False)
    summary = summarize_metrics(raw)
    summary.to_csv(out / "scienceworld_expand_summary.csv", index=False)
    paired = paired_contrasts(raw)
    paired.to_csv(out / "scienceworld_expand_method_bootstrap.csv", index=False)
    action = action_flips([args.m1, args.m4], out)
    action.to_csv(out / "scienceworld_expand_action_bootstrap.csv", index=False)
    meta = {
        "inputs": [str(args.m1), str(args.m4)],
        "tasks": sorted(raw.task.unique().tolist()),
        "variations": sorted(raw.variation.unique().tolist()),
        "methods": sorted(raw.method.unique().tolist()),
        "duplications": sorted(raw.duplication.unique().tolist()),
        "episodes": int(len(raw)),
        "valid_episodes": int(raw.error.isna().sum()),
        "error_episodes": int(raw.error.notna().sum()),
        "selection_rule": "fixed task/variation grid: find-plant and find-animal IDs 150--155",
        "gold_path": False,
        "hidden_state": False,
    }
    (out / "scienceworld_expand_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
