"""Summarize the fair algorithmic flat-value ScienceWorld intervention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEFT = "flat_value"
RIGHT = "provenance_value"


def load_run(path: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(path.glob("episodes/*-summary.json")):
        row = json.loads(p.read_text())
        row["source_directory"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def boot(values: np.ndarray, seed: int = 20260929, draws: int = 20000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(sample, .025)), float(np.quantile(sample, .975))


def actions(path: str) -> list[str]:
    return [str(json.loads(line).get("action", "")) for line in Path(path).open()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", type=Path, required=True)
    ap.add_argument("--m4", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    frame = pd.concat([load_run(args.m1), load_run(args.m4)], ignore_index=True)
    if frame.empty:
        raise RuntimeError("no episode summaries")
    errors = frame[frame.error.notna()].copy()
    valid = frame[frame.error.isna()].copy()
    valid["task_family"] = valid.task.map(lambda x: "find-plant" if "plant" in str(x) else "find-animal")
    args.output.mkdir(parents=True, exist_ok=True)
    valid.to_csv(args.output / "rows.csv", index=False)
    errors.to_csv(args.output / "raw_errors.csv", index=False)
    summary = valid.groupby(["task_family", "duplication", "method"], as_index=False).agg(
        n=("variation", "size"), success=("success", "mean"), final_score=("final_score", "mean"),
        reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"),
        unnecessary_actions=("repeated_no_visible_change", "mean"),
        initial_success_probability=("initial_success_probability", "mean"))
    summary.to_csv(args.output / "summary.csv", index=False)
    paired = []
    for (family, variation), group in valid.groupby(["task_family", "variation"]):
        w = group.set_index(["method", "duplication"])
        required = {(LEFT, 1), (LEFT, 4), (RIGHT, 1), (RIGHT, 4)}
        if not required.issubset(set(w.index)):
            continue
        rec = {"task_family": family, "variation": int(variation)}
        for metric in ("success", "final_score", "reward", "steps", "charged_tokens", "repeated_no_visible_change"):
            for method in (LEFT, RIGHT):
                for dup in (1, 4):
                    rec[f"{method}_m{dup}_{metric}"] = float(w.loc[(method, dup), metric])
            rec[f"{RIGHT}_minus_{LEFT}_m1_{metric}"] = rec[f"{RIGHT}_m1_{metric}"] - rec[f"{LEFT}_m1_{metric}"]
            rec[f"{RIGHT}_minus_{LEFT}_m4_{metric}"] = rec[f"{RIGHT}_m4_{metric}"] - rec[f"{LEFT}_m4_{metric}"]
        paired.append(rec)
    paired = pd.DataFrame(paired)
    paired.to_csv(args.output / "paired.csv", index=False)
    contrasts = []
    for family, group in paired.groupby("task_family"):
        for metric in ("success", "final_score", "reward", "steps", "charged_tokens", "repeated_no_visible_change"):
            for dup in (1, 4):
                col = f"{RIGHT}_minus_{LEFT}_m{dup}_{metric}"
                estimate, lo, hi = boot(group[col].dropna().to_numpy(), 20260929 + dup)
                contrasts.append({"task_family": family, "duplication": dup, "metric": metric,
                                  "n_episodes": int(group[col].notna().sum()), "estimate": estimate,
                                  "ci95_low": lo, "ci95_high": hi, "bootstrap_draws": 20000})
    pd.DataFrame(contrasts).to_csv(args.output / "bootstrap.csv", index=False)
    consistency = []
    for (family, variation, method), group in valid.groupby(["task_family", "variation", "method"]):
        by_dup = {int(row.duplication): row for _, row in group.iterrows()}
        if 1 not in by_dup or 4 not in by_dup:
            continue
        a, b = actions(by_dup[1].trace_file), actions(by_dup[4].trace_file)
        n = min(len(a), len(b)); flips = sum(x != y for x, y in zip(a[:n], b[:n]))
        consistency.append({"task_family": family, "variation": int(variation), "method": method,
                            "steps_m1": len(a), "steps_m4": len(b), "aligned_steps": n,
                            "action_flips": flips, "action_flip_rate": flips / n if n else 0.0})
    consistency = pd.DataFrame(consistency)
    consistency.to_csv(args.output / "action_consistency.csv", index=False)
    action_rows = []
    for (family, method), group in consistency.groupby(["task_family", "method"]):
        estimate, lo, hi = boot(group.action_flip_rate.to_numpy(), 20261001)
        action_rows.append({"task_family": family, "method": method, "n_episodes": len(group),
                            "estimate": estimate, "ci95_low": lo, "ci95_high": hi, "bootstrap_draws": 20000})
    pd.DataFrame(action_rows).to_csv(args.output / "action_bootstrap.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for ax, family in zip(axes, sorted(valid.task_family.unique())):
        z = summary[summary.task_family == family]
        for method, group in z.groupby("method"):
            group = group.sort_values("duplication")
            ax.plot(group.duplication, group.success, "o-", label=method)
        ax.set_title(family); ax.set_xticks([1, 4]); ax.set_xlabel("duplication"); ax.set_ylim(-.05, 1.05); ax.set_ylabel("success")
    axes[0].legend(fontsize=7); fig.savefig(args.output / "curve.png", dpi=180); plt.close(fig)
    metadata = {"protocol": "scienceworld-local-qwen-fair-flat-value-v1", "model": "Qwen2.5-Coder-3B-Instruct",
                "valid_rows": int(len(valid)), "raw_error_rows": int(len(errors)), "paired_episodes": int(len(paired)),
                "task_families": sorted(valid.task_family.unique().tolist()), "duplications": [1, 4],
                "methods": [LEFT, RIGHT], "step_limit": int(max(valid.step_limit)),
                "gpu_probe": "results_submission/report/local_qwen_gpu_probe.json", "episode_success_claim": False,
                "flat_value_definition": "same conditional-value estimator with q shifted by (duplication-1)*log(1.5)",
                "interpretation": "fair algorithmic readout intervention; task-family strata remain separate"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False)); print(summary.to_string(index=False)); print(pd.DataFrame(action_rows).to_string(index=False))


if __name__ == "__main__":
    main()
