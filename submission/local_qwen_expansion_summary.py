"""Summarize a fresh CUDA-Qwen ScienceWorld paired expansion.

The runner stores one summary per method/variation/budget.  This report keeps
plant and animal task families separate, excludes malformed episodes, computes
paired episode bootstrap intervals, and measures duplication-1/4 action flips
from the serialized traces.  It is deliberately independent of the older
five-variation pilot so the new evidence has an explicit provenance boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run(path: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(path.glob("episodes/*-summary.json")):
        row = json.loads(p.read_text())
        row["source_directory"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(values: np.ndarray, seed: int = 20260929, draws: int = 20000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(sample, .025)), float(np.quantile(sample, .975))


def trace_actions(path: str) -> list[str]:
    out = []
    with Path(path).open() as f:
        for line in f:
            out.append(str(json.loads(line).get("action", "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", type=Path, required=True)
    ap.add_argument("--m4", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    frame = pd.concat([load_run(args.m1), load_run(args.m4)], ignore_index=True)
    if frame.empty:
        raise RuntimeError("no episode summaries")
    errors = frame[frame.error.notna()]
    valid = frame[frame.error.isna()].copy()
    valid["task_family"] = valid["task"].map(lambda x: "find-plant" if "plant" in str(x) else "find-animal")
    # The fair algorithmic control uses ``flat_value`` while older interactive
    # runs use ``flat``.  Normalize the baseline name only for the paired
    # columns so the same summarizer can audit either protocol.
    flat_method = "flat" if "flat" in set(valid.method) else "flat_value"
    args.output.mkdir(parents=True, exist_ok=True)
    valid.to_csv(args.output / "rows.csv", index=False)
    errors.to_csv(args.output / "raw_errors.csv", index=False)
    summary = valid.groupby(["task_family", "duplication", "method"], as_index=False).agg(
        n=("variation", "size"), success=("success", "mean"), final_score=("final_score", "mean"),
        reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"),
        repeated_no_visible_change=("repeated_no_visible_change", "mean"),
        initial_success_probability=("initial_success_probability", "mean"))
    summary.to_csv(args.output / "summary.csv", index=False)
    paired = []
    for (family, variation), g in valid.groupby(["task_family", "variation"]):
        wide = g.set_index(["method", "duplication"])
        if not all(k in wide.index for k in ((flat_method, 1), (flat_method, 4), ("provenance_value", 1), ("provenance_value", 4))):
            continue
        rec = {"task_family": family, "variation": int(variation)}
        for metric in ("success", "final_score", "reward", "steps", "charged_tokens", "repeated_no_visible_change"):
            for method in (flat_method, "provenance_value"):
                for dup in (1, 4):
                    label = "flat" if method == flat_method else method
                    rec[f"{label}_m{dup}_{metric}"] = float(wide.loc[(method, dup), metric])
            rec[f"provenance_value_minus_flat_m1_{metric}"] = rec[f"provenance_value_m1_{metric}"] - rec[f"flat_m1_{metric}"]
            rec[f"provenance_value_minus_flat_m4_{metric}"] = rec[f"provenance_value_m4_{metric}"] - rec[f"flat_m4_{metric}"]
        paired.append(rec)
    paired = pd.DataFrame(paired)
    paired.to_csv(args.output / "paired.csv", index=False)
    contrasts = []
    for family, g in paired.groupby("task_family"):
        for metric in ("success", "final_score", "reward", "steps", "charged_tokens", "repeated_no_visible_change"):
            for dup in (1, 4):
                col = f"provenance_value_minus_flat_m{dup}_{metric}"
                est, lo, hi = bootstrap(g[col].dropna().to_numpy(), seed=20260929 + dup)
                contrasts.append({"task_family": family, "duplication": dup, "metric": metric,
                                  "n_episodes": int(g[col].notna().sum()), "estimate": est,
                                  "ci95_low": lo, "ci95_high": hi, "bootstrap_draws": 20000})
    pd.DataFrame(contrasts).to_csv(args.output / "bootstrap.csv", index=False)
    consistency = []
    for (family, variation, method), g in valid.groupby(["task_family", "variation", "method"]):
        by_dup = {int(r.duplication): r for _, r in g.iterrows()}
        if 1 not in by_dup or 4 not in by_dup:
            continue
        a, b = trace_actions(by_dup[1].trace_file), trace_actions(by_dup[4].trace_file)
        n = min(len(a), len(b)); flips = sum(x != y for x, y in zip(a[:n], b[:n]))
        consistency.append({"task_family": family, "variation": int(variation), "method": method,
                            "steps_m1": len(a), "steps_m4": len(b), "aligned_steps": n,
                            "action_flips": flips, "action_flip_rate": flips / n if n else 0.0})
    consistency = pd.DataFrame(consistency)
    consistency.to_csv(args.output / "action_consistency.csv", index=False)
    action_bootstrap = []
    for (family, method), g in consistency.groupby(["task_family", "method"]):
        est, lo, hi = bootstrap(g.action_flip_rate.to_numpy(), seed=20261001)
        action_bootstrap.append({"task_family": family, "method": method, "n_episodes": len(g),
                                 "estimate": est, "ci95_low": lo, "ci95_high": hi, "bootstrap_draws": 20000})
    pd.DataFrame(action_bootstrap).to_csv(args.output / "action_bootstrap.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for ax, family in zip(axes, sorted(valid.task_family.unique())):
        z = summary[summary.task_family == family]
        for method, h in z.groupby("method"):
            h = h.sort_values("duplication")
            ax.plot(h.duplication, h.success, "o-", label=method)
        ax.set_title(family); ax.set_xticks([1, 4]); ax.set_xlabel("duplication"); ax.set_ylim(-.05, 1.05); ax.set_ylabel("success")
    axes[0].legend(fontsize=7)
    fig.savefig(args.output / "curve.png", dpi=180); plt.close(fig)
    metadata = {
        "protocol": "scienceworld-local-qwen-cuda-expansion-v1", "model": "Qwen2.5-Coder-3B-Instruct",
        "valid_rows": int(len(valid)), "raw_error_rows": int(len(errors)), "paired_episodes": int(len(paired)),
        "task_families": sorted(valid.task_family.unique().tolist()), "duplications": [1, 4],
        "methods": sorted(valid.method.unique().tolist()), "baseline_method": flat_method,
        "step_limit": int(max(valid.step_limit)),
        "gpu_probe": "results_submission/report/local_qwen_gpu_probe.json",
        "cuda_policy": "CUDA required; CPU fallback disabled", "episode_success_claim": False,
        "candidate_schema_retry": "one explicit local-CUDA retry after preserving the first malformed completion; no API repair",
        "interpretation": "fresh paired CUDA-Qwen exploratory expansion; task-family strata remain separate",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False)); print(summary.to_string(index=False)); print(pd.DataFrame(action_bootstrap).to_string(index=False))


if __name__ == "__main__":
    main()
