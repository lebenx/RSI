"""Summarize the expanded CUDA local-Qwen ScienceWorld pilot.

This is a separate, balanced six-variation pilot.  It deliberately keeps
the success-aware provenance objective labelled as an exploratory readout and
does not pool these rows with the DeepSeek experiments or the earlier three-
variation pilot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# 150/151 are retained as interface-failure controls in the raw run.  The
# balanced pilot below uses the five variations with complete paired rows;
# 158/161 come from the same v9 runner archives used by the original pilot.
VARIATIONS = [152, 153, 154, 158, 161]


def load_run(path: Path, duplication: int) -> pd.DataFrame:
    rows = []
    for row in json.loads((path / "summaries.json").read_text()):
        if row.get("error") is not None:
            continue
        row = dict(row)
        row["duplication"] = duplication
        row["source_directory"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(values: np.ndarray, seed: int = 20260929, draws: int = 20000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=Path, default=Path("results_submission/scienceworld_local_qwen_grid150_155_v9_m1"))
    parser.add_argument("--m4", type=Path, default=Path("results_submission/scienceworld_local_qwen_grid150_155_v9_m4"))
    parser.add_argument("--output", type=Path, default=Path("results_submission/report/local_qwen_expanded_pilot"))
    args = parser.parse_args()

    extra_m1 = load_run(Path("results_submission/scienceworld_local_qwen_value_grid_v9_m1"), 1)
    extra_m4_grid = load_run(Path("results_submission/scienceworld_local_qwen_value_grid_m4"), 4)
    extra_m4_161 = load_run(Path("results_submission/scienceworld_local_qwen_value161_v9_m4"), 4)
    frame = pd.concat([load_run(args.m1, 1), load_run(args.m4, 4),
                       extra_m1[extra_m1["variation"].astype(int).isin([158, 161])],
                       extra_m4_grid[extra_m4_grid["variation"].astype(int).isin([158])],
                       extra_m4_161], ignore_index=True)
    frame = frame[frame["variation"].astype(int).isin(VARIATIONS)].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "rows.csv", index=False)
    raw_error_rows = []
    for directory, duplication in ((args.m1, 1), (args.m4, 4)):
        for row in json.loads((directory / "summaries.json").read_text()):
            if row.get("error") is not None:
                raw_error_rows.append({"task": row.get("task"), "variation": row.get("variation"),
                                       "method": row.get("method"), "duplication": duplication,
                                       "error": json.dumps(row.get("error"), ensure_ascii=False),
                                       "source_directory": str(directory)})
    pd.DataFrame(raw_error_rows).to_csv(args.output / "raw_errors.csv", index=False)
    frame["initial_brier"] = (frame["initial_success_probability"] - frame["success"]) ** 2
    summary = frame.groupby(["duplication", "method"], as_index=False).agg(
        n=("variation", "size"), success=("success", "mean"),
        final_score=("final_score", "mean"), reward=("reward", "mean"),
        steps=("steps", "mean"), tokens=("charged_tokens", "mean"),
        initial_brier=("initial_brier", "mean"),
        repeated_no_visible_change=("repeated_no_visible_change", "mean"),
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    # Compact paper-facing figure; the underlying CSV remains authoritative.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    labels = ["flat m1", "prov m1", "flat m4", "prov m4"]
    for ax, metric, title in zip(axes, ("success", "reward"), ("Episode success", "Mean reward")):
        vals = []
        for dup, method in ((1, "flat"), (1, "provenance_success"), (4, "flat"), (4, "provenance_success")):
            row = summary[(summary.duplication == dup) & (summary.method == method)]
            vals.append(float(row[metric].iloc[0]) if len(row) else float("nan"))
        ax.bar(labels, vals, color=["#8884d8", "#82ca9d", "#8884d8", "#82ca9d"])
        ax.set_title(title); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
    fig.savefig(args.output / "curve.png", dpi=160); plt.close(fig)

    wide = frame.pivot_table(index=["task", "variation"], columns=["method", "duplication"],
                              values=["success", "final_score", "reward", "steps", "charged_tokens"],
                              aggfunc="first")
    paired = []
    for (task, variation), row in wide.iterrows():
        if ("success", "flat", 1) not in row.index or ("success", "flat", 4) not in row.index:
            continue
        rec = {"task": task, "variation": int(variation)}
        for metric in ("success", "final_score", "reward", "steps", "charged_tokens"):
            for method in ("flat", "provenance_success"):
                for dup in (1, 4):
                    rec[f"{method}_m{dup}_{metric}"] = row.get((metric, method, dup))
            rec[f"provenance_success_minus_flat_m1_{metric}"] = rec[f"provenance_success_m1_{metric}"] - rec[f"flat_m1_{metric}"]
            rec[f"provenance_success_minus_flat_m4_{metric}"] = rec[f"provenance_success_m4_{metric}"] - rec[f"flat_m4_{metric}"]
            rec[f"flat_m4_minus_flat_m1_{metric}"] = rec[f"flat_m4_{metric}"] - rec[f"flat_m1_{metric}"]
        paired.append(rec)
    paired = pd.DataFrame(paired)
    paired.to_csv(args.output / "paired.csv", index=False)

    contrasts = []
    for metric in ("success", "reward", "steps"):
        for dup in (1, 4):
            col = f"provenance_success_minus_flat_m{dup}_{metric}"
            estimate, low, high = bootstrap(paired[col].dropna().to_numpy(float))
            contrasts.append({"duplication": dup, "metric": metric, "n_episodes": int(paired[col].notna().sum()),
                              "estimate": estimate, "ci95_low": low, "ci95_high": high, "bootstrap_draws": 20000})
    pd.DataFrame(contrasts).to_csv(args.output / "bootstrap.csv", index=False)

    probe_path = Path("results_submission/report/local_qwen_gpu_probe.json")
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    metadata = {
        "protocol": "local-qwen-expanded-full-episode-pilot-v1",
        "model": "Qwen2.5-Coder-3B-Instruct",
        "task": "find-plant",
        "variations": VARIATIONS,
        "duplications": [1, 4],
        "methods": ["flat", "provenance_success"],
        "valid_rows": int(len(frame)),
        "error_rows": 0,
        "paired_episodes": int(len(paired)),
        "step_limit": 8,
        "episode_success_claim": False,
        "planner_claim": False,
        "calibration_metric": "initial Brier=(initial_success_probability-success)^2",
        "gpu_probe": probe,
        "cuda_run_policy": "LOCAL_QWEN_DEVICE=cuda; CUDA is required and CPU fallback is disabled",
        "runner_version": "scienceworld_runner v9",
        "raw_extension_variations": [150, 151, 155],
        "raw_extension_error_rows": len(raw_error_rows),
        "interpretation": "small second-model exploratory pilot; complete paired rows only; raw interface failures are retained separately; not pooled with DeepSeek and not a confirmatory success estimate",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
