"""Generate stratified effect estimates for the consolidated CUDA-Qwen study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def bootstrap(x: np.ndarray, seed: int, draws: int = 20000) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def run(output: str = "results_submission/report/local_qwen_effect_heterogeneity") -> None:
    expansions = [
        ROOT / "results_submission/report/local_qwen_expansion_162_164",
        ROOT / "results_submission/report/local_qwen_expansion_165_167",
        ROOT / "results_submission/report/local_qwen_expansion_170_171_retry",
        ROOT / "results_submission/report/local_qwen_expansion_172_173",
    ]
    rows = []
    for path in expansions:
        frame = pd.read_csv(path / "paired.csv")
        label = path.name.replace("local_qwen_expansion_", "")
        for family, group in frame.groupby("task_family"):
            for duplication in (1, 4):
                for metric in ("reward", "success", "repeated_no_visible_change"):
                    col = f"provenance_value_minus_flat_m{duplication}_{metric}"
                    values = group[col].dropna().to_numpy(dtype=float)
                    estimate, low, high = bootstrap(values, 20261000 + duplication)
                    rows.append({"expansion": label, "task_family": family,
                                 "duplication": duplication, "metric": metric,
                                 "n_episodes": int(len(values)), "estimate": estimate,
                                 "ci95_low": low, "ci95_high": high,
                                 "bootstrap_draws": 20000})
    result = pd.DataFrame(rows)
    out = ROOT / output
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "heterogeneity.csv", index=False)
    reward = result[result.metric == "reward"].copy()
    reward["label"] = reward.apply(lambda r: f"{r.expansion}\n{r.task_family.replace('find-','')}\nm={int(r.duplication)}", axis=1)
    reward = reward.sort_values(["duplication", "task_family", "expansion"])
    fig, ax = plt.subplots(figsize=(10, max(4, .27 * len(reward) + 1.5)), constrained_layout=True)
    y = np.arange(len(reward))
    ax.errorbar(reward.estimate, y,
                xerr=[reward.estimate - reward.ci95_low, reward.ci95_high - reward.estimate],
                fmt="o", capsize=2, color="#245b8a")
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(y, reward.label)
    ax.set_xlabel("Provenance-value minus flat reward")
    ax.set_title("CUDA-Qwen paired effect heterogeneity")
    ax.grid(axis="x", alpha=.25)
    fig.savefig(out / "forest.png", dpi=180); plt.close(fig)
    metadata = {"protocol": "local-qwen-effect-heterogeneity-v1", "source_expansions": [str(p) for p in expansions],
                "rows": int(len(result)), "reward_rows": int(len(reward)), "bootstrap_draws": 20000,
                "paired_unit": "episode variation", "pooled": False,
                "interpretation": "effect display only; task-family and expansion strata remain separate"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_submission/report/local_qwen_effect_heterogeneity")
    run(parser.parse_args().output)
