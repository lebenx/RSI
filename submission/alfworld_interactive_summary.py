"""Merge validated ALFWorld TextWorld interactive runs into one report.

The runner writes one row per episode/method/duplication. This module combines
all validated protocol-v4 strata (the initial DeepSeek/Qwen smokes, the
six-task extensions, and the cached success-objective ablation); earlier
protocol versions are intentionally excluded because their task-persistence
contract was invalid or their rollout bank was incomplete.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_submission" / "report" / "alfworld_interactive"
EXCLUDED_RUNS = {
    "alfworld_interactive_v1", "alfworld_interactive_v2", "alfworld_interactive_v3",
    "alfworld_interactive_v5_qwen",
}


def discover_runs() -> list[Path]:
    """Return validated protocol-v4 run directories in deterministic order."""
    runs = []
    for path in sorted(ROOT.glob("results_submission/alfworld_interactive_v*")):
        if not path.is_dir() or path.name in EXCLUDED_RUNS:
            continue
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("protocol") == "alfworld-textworld-provenance-v4-task-persistent":
            runs.append(path)
    required = {"alfworld_interactive_v4_deepseek", "alfworld_interactive_v6_qwen"}
    missing = required - {p.name for p in runs}
    if missing:
        raise FileNotFoundError(f"required validated ALFWorld runs missing: {sorted(missing)}")
    return runs


def _load_valid(run: Path) -> tuple[pd.DataFrame, dict]:
    meta = json.loads((run / "metadata.json").read_text())
    rows = pd.read_csv(run / "rows.csv")
    if "errors" in rows:
        bad = rows[rows.errors.fillna("[]").astype(str).ne("[]")]
        if len(bad):
            raise ValueError(f"{run}: rows with errors are not valid: {len(bad)}")
    if bool(meta.get("uses_hidden_state_in_prompt")) or bool(meta.get("uses_gold_plan")):
        raise ValueError(f"{run}: hidden state/gold plan flag is set")
    rows = rows.copy()
    rows["model"] = "DeepSeek" if meta.get("backend") == "deepseek" else "Qwen2.5-Coder-3B"
    rows["backend"] = str(meta.get("backend"))
    rows["source_run"] = run.name
    rows["root_confidence_delta"] = rows["root_confidence_last"] - rows["root_confidence_first"]
    rows["trace_file"] = rows.trace_file.map(lambda p: str(Path(p)))
    return rows, meta


def _trace_actions(path: Path) -> list[str]:
    actions = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            actions.append(str(row.get("action", "")))
    return actions


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, metas = [], {}
    runs = discover_runs()
    for run in runs:
        frame, meta = _load_valid(run)
        frames.append(frame)
        metas[run.name] = meta
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(OUT / "rows.csv", index=False)

    metric_cols = [
        "success", "reward", "final_score", "steps", "unnecessary_actions",
        "charged_tokens", "mean_belief_drift", "initial_success_probability",
        "calibration_brier", "root_confidence_first", "root_confidence_last",
        "root_confidence_delta",
    ]
    grouped = rows.groupby(["model", "task_type", "method", "duplication"], as_index=False)
    summary = grouped[metric_cols].agg(["count", "mean"]).reset_index()
    # Flatten the two-level metric names while retaining n for each metric.
    flat = summary.copy()
    flat.columns = [
        "_".join(str(x) for x in col if str(x) not in ("", "None"))
        if isinstance(col, tuple) else str(col) for col in flat.columns
    ]
    # A compact table is easier to cite: all metrics have the same row count.
    compact = rows.groupby(["model", "task_type", "method", "duplication"], as_index=False)[metric_cols].mean()
    compact.insert(4, "n", rows.groupby(["model", "task_type", "method", "duplication"]).size().to_numpy())
    compact.to_csv(OUT / "summary.csv", index=False)

    paired_rows = []
    objective_rows = []
    keys = ["model", "source_run", "task_type", "game_id", "duplication"]
    for key, g in rows[rows.method.isin(["flat", "provenance_value", "provenance_success"])].groupby(keys):
        wide = g.set_index("method")
        if "flat" not in wide.index:
            continue
        rec = dict(zip(keys, key))
        if "provenance_value" in wide.index:
            rec["provenance_value_minus_flat_success"] = float(wide.loc["provenance_value", "success"] - wide.loc["flat", "success"])
            rec["provenance_value_minus_flat_reward"] = float(wide.loc["provenance_value", "reward"] - wide.loc["flat", "reward"])
            rec["provenance_value_minus_flat_steps"] = float(wide.loc["provenance_value", "steps"] - wide.loc["flat", "steps"])
            rec["provenance_value_minus_flat_unnecessary_actions"] = float(wide.loc["provenance_value", "unnecessary_actions"] - wide.loc["flat", "unnecessary_actions"])
            rec["provenance_value_minus_flat_tokens"] = float(wide.loc["provenance_value", "charged_tokens"] - wide.loc["flat", "charged_tokens"])
            rec["flat_mean_belief_drift"] = float(wide.loc["flat", "mean_belief_drift"])
            rec["provenance_value_mean_belief_drift"] = float(wide.loc["provenance_value", "mean_belief_drift"])
            rec["duplication_invariant_provenance_row"] = bool(wide.loc["provenance_value", "mean_belief_drift"] == 0.0)
            paired_rows.append(rec)
        for method in ("provenance_value", "provenance_success"):
            if method not in wide.index:
                continue
            o = dict(zip(keys, key)); o["method"] = method
            o["contrast"] = f"{method}_minus_flat"
            for metric in ("success", "reward", "steps", "unnecessary_actions", "charged_tokens", "mean_belief_drift", "root_confidence_delta", "calibration_brier"):
                o[f"{method}_minus_flat_{metric}"] = float(wide.loc[method, metric] - wide.loc["flat", metric])
            objective_rows.append(o)
    pd.DataFrame(paired_rows).to_csv(OUT / "paired.csv", index=False)
    # Fair algorithmic control: compare provenance_value with flat_value on
    # the same conditional bank.  Unlike the legacy ``flat`` reader, this
    # isolates the duplicate-count pseudo-likelihood from model readout.
    algorithmic_rows = []
    algo_keys = ["model", "source_run", "task_type", "game_id", "duplication"]
    for key, g in rows[rows.method.isin(["flat_value", "provenance_value"])].groupby(algo_keys):
        wide = g.set_index("method")
        if not {"flat_value", "provenance_value"}.issubset(wide.index):
            continue
        rec = dict(zip(algo_keys, key))
        for metric in ("success", "reward", "final_score", "steps", "unnecessary_actions",
                       "charged_tokens", "mean_belief_drift", "calibration_brier"):
            rec[f"provenance_value_minus_flat_value_{metric}"] = float(
                wide.loc["provenance_value", metric] - wide.loc["flat_value", metric])
        algorithmic_rows.append(rec)
    pd.DataFrame(algorithmic_rows).to_csv(OUT / "paired_algorithmic.csv", index=False)
    # The success-objective run is intentionally a separate cache-backed
    # protocol. Pair it against the matching flat rows from the six-task
    # DeepSeek base run by task/game/duplication, while retaining both source
    # run names in the provenance columns.
    success_run = rows[rows.source_run.str.contains("v9_deepseek_success", regex=False) & (rows.method == "provenance_success")]
    base_flat = rows[(rows.source_run == "alfworld_interactive_v7_deepseek") & (rows.method == "flat")]
    if len(success_run) and len(base_flat):
        merge_keys = ["model", "task_type", "game_id", "duplication"]
        merged = success_run.merge(base_flat, on=merge_keys, suffixes=("_success", "_flat"))
        for _, row in merged.iterrows():
            o = {k: row[k] for k in merge_keys}
            o.update({"source_run": "v9_success_vs_v7_flat", "method": "provenance_success",
                      "contrast": "provenance_success_minus_flat"})
            for metric in ("success", "reward", "steps", "unnecessary_actions", "charged_tokens", "mean_belief_drift", "root_confidence_delta"):
                o[f"provenance_success_minus_flat_{metric}"] = float(row[f"{metric}_success"] - row[f"{metric}_flat"])
            objective_rows.append(o)
    pd.DataFrame(objective_rows).to_csv(OUT / "paired_objectives.csv", index=False)

    # Episode-cluster bootstrap for the provenance-value minus flat contrast.
    # Models and task types remain strata; no pooled superiority estimate is
    # implied by the aggregate ``task_type=all`` rows.
    bootstrap_rows = []
    rng = np.random.default_rng(20260929)
    contrast_metrics = ["success", "reward", "steps", "unnecessary_actions",
                        "charged_tokens", "root_confidence_delta", "mean_belief_drift",
                        "calibration_brier"]
    contrast_methods = ["provenance_value", "provenance_success"]
    for model, dup in sorted(rows[["model", "duplication"]].drop_duplicates().itertuples(index=False, name=None)):
        strata = [("all", rows[(rows.model == model) & (rows.duplication == dup)])]
        strata.extend((str(int(task)), rows[(rows.model == model) & (rows.duplication == dup) & (rows.task_type == task)])
                      for task in sorted(rows.loc[rows.model == model, "task_type"].unique()))
        for task_label, subset in strata:
            wide = subset[subset.method.isin(["flat", *contrast_methods])].pivot_table(
                index=["source_run", "task_type", "game_id"], columns="method", values=contrast_metrics, aggfunc="first")
            for metric in contrast_metrics:
                for method in contrast_methods:
                    if (metric, "flat") not in wide.columns or (metric, method) not in wide.columns:
                        continue
                    diff = (wide[(metric, method)] - wide[(metric, "flat")]).dropna().to_numpy(float)
                    if not len(diff):
                        continue
                    draws = rng.integers(0, len(diff), size=(20000, len(diff)))
                    means = diff[draws].mean(axis=1)
                    bootstrap_rows.append({
                        "model": model, "task_type": task_label, "duplication": int(dup),
                        "contrast": f"{method}_minus_flat", "metric": metric,
                        "n_episodes": int(len(diff)), "estimate": float(diff.mean()),
                        "ci95_low": float(np.quantile(means, 0.025)),
                        "ci95_high": float(np.quantile(means, 0.975)), "bootstrap_draws": 20000,
                    })
            algo_wide = subset[subset.method.isin(["flat_value", "provenance_value"])].pivot_table(
                index=["source_run", "task_type", "game_id"], columns="method", values=contrast_metrics, aggfunc="first")
            for metric in contrast_metrics:
                if (metric, "flat_value") not in algo_wide.columns or (metric, "provenance_value") not in algo_wide.columns:
                    continue
                diff = (algo_wide[(metric, "provenance_value")] - algo_wide[(metric, "flat_value")]).dropna().to_numpy(float)
                if not len(diff):
                    continue
                draws = rng.integers(0, len(diff), size=(20000, len(diff)))
                means = diff[draws].mean(axis=1)
                bootstrap_rows.append({
                    "model": model, "task_type": task_label, "duplication": int(dup),
                    "contrast": "provenance_value_minus_flat_value", "metric": metric,
                    "n_episodes": int(len(diff)), "estimate": float(diff.mean()),
                    "ci95_low": float(np.quantile(means, 0.025)),
                    "ci95_high": float(np.quantile(means, 0.975)), "bootstrap_draws": 20000,
                })
    if len(success_run) and len(base_flat):
        merged = success_run.merge(base_flat, on=["model", "task_type", "game_id", "duplication"], suffixes=("_success", "_flat"))
        for dup in sorted(merged.duplication.unique()):
            z = merged[merged.duplication == dup]
            for metric in contrast_metrics:
                diff = (z[f"{metric}_success"] - z[f"{metric}_flat"]).dropna().to_numpy(float)
                if not len(diff):
                    continue
                draws = rng.integers(0, len(diff), size=(20000, len(diff)))
                means = diff[draws].mean(axis=1)
                bootstrap_rows.append({"model": "DeepSeek", "task_type": "success_ablation",
                    "duplication": int(dup), "contrast": "provenance_success_minus_flat",
                    "metric": metric, "n_episodes": int(len(diff)), "estimate": float(diff.mean()),
                    "ci95_low": float(np.quantile(means, 0.025)), "ci95_high": float(np.quantile(means, 0.975)),
                    "bootstrap_draws": 20000})

    consistency = []
    for (model, source_run, task_type, game_id, method), g in rows[rows.method.isin(["flat", "flat_value", "provenance_value", "provenance_success"])].groupby(["model", "source_run", "task_type", "game_id", "method"]):
        by_dup = {int(r.duplication): r for _, r in g.iterrows()}
        if 1 not in by_dup or 4 not in by_dup:
            continue
        a = _trace_actions(ROOT / by_dup[1].trace_file)
        b = _trace_actions(ROOT / by_dup[4].trace_file)
        n = min(len(a), len(b))
        flips = sum(x != y for x, y in zip(a[:n], b[:n]))
        consistency.append({
            "model": model, "source_run": source_run, "task_type": task_type, "game_id": game_id, "method": method,
            "steps_m1": len(a), "steps_m4": len(b), "aligned_steps": n,
            "action_flips": flips, "action_flip_rate": (flips / n if n else 0.0),
            "traces_aligned_by_step": True,
        })
    pd.DataFrame(consistency).to_csv(OUT / "action_consistency.csv", index=False)
    action_bootstrap = []
    for (model, method), g in pd.DataFrame(consistency).groupby(["model", "method"]):
        vals = g.action_flip_rate.to_numpy(float)
        if not len(vals):
            continue
        draws = rng.integers(0, len(vals), size=(20000, len(vals)))
        means = vals[draws].mean(axis=1)
        action_bootstrap.append({
            "model": model, "method": method, "n_episodes": int(len(vals)),
            "estimate": float(vals.mean()), "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)), "bootstrap_draws": 20000,
        })
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "bootstrap.csv", index=False)
    pd.DataFrame(action_bootstrap).to_csv(OUT / "action_bootstrap.csv", index=False)

    # Compact paper/appendix figure; each model is a separate panel so the
    # exploratory strata are never pooled into one effect curve.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    colors = {"flat": "#c43c39", "flat_value": "#d98c00", "provenance_value": "#276fbf",
              "provenance_success": "#7a5aa6", "no_imagination": "#4f9d69"}
    for model, panel in zip(sorted(rows.model.unique()), axes[:2]):
        z = compact[compact.model == model]
        for method, g in z.groupby("method"):
            g = g.sort_values("duplication")
            panel.plot(g.duplication, g.success, "o-", label=method,
                       color=colors.get(method, "#333333"))
        panel.set_xscale("log", base=2); panel.set_ylim(-0.05, 1.05)
        panel.set_xticks([1, 4]); panel.set_xticklabels(["1", "4"])
        panel.set_xlabel("duplication"); panel.set_ylabel("success")
        panel.set_title(model)
    if len(action_bootstrap):
        for method, g in pd.DataFrame(action_bootstrap).groupby("method"):
            # One point per model; use model-coded labels in the x axis.
            vals = g.sort_values("model")
            axes[2].plot(vals.model, vals.estimate, "o", label=method,
                         color=colors.get(method, "#333333"))
        axes[2].set_ylim(-0.05, 1.05); axes[2].set_ylabel("aligned action-flip rate")
        axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_title("duplication 1→4 action consistency")
    axes[0].legend(fontsize=7)
    fig.savefig(OUT / "curve.png", dpi=180)
    plt.close(fig)

    meta = {
        "protocol": "alfworld-textworld-provenance-v4-task-persistent",
        "valid_rows": int(len(rows)),
        "models": sorted(rows.model.unique().tolist()),
        "runs": {name: {k: v for k, v in m.items() if k not in {"model"}} for name, m in metas.items()},
        "deepseek_valid_rows": int((rows.model == "DeepSeek").sum()),
        "deepseek_base_valid_rows": int(((rows.model == "DeepSeek") & ~rows.source_run.str.contains("v9_deepseek_success", regex=False)).sum()),
        "deepseek_success_ablation_rows": int(((rows.model == "DeepSeek") & rows.source_run.str.contains("v9_deepseek_success", regex=False)).sum()),
        "flat_value_valid_rows": int((rows.method == "flat_value").sum()),
        "algorithmic_control": "v12 compares flat_value and provenance_value on identical conditional banks; duplicate-sensitive flat posterior uses logit increment (m-1)log(1.5)",
        "deepseek_episode_count": int(rows.loc[rows.model == "DeepSeek", ["source_run", "task_type", "game_id"]].drop_duplicates().shape[0]),
        "qwen_episode_count": int(rows.loc[rows.model == "Qwen2.5-Coder-3B", ["source_run", "task_type", "game_id"]].drop_duplicates().shape[0]),
        "qwen_cuda_valid_rows": int((rows.model == "Qwen2.5-Coder-3B").sum()),
        "qwen_cuda_inference": metas.get("alfworld_interactive_v6_qwen", {}).get("inference"),
        "qwen_cuda_inference_by_run": {name: m.get("inference") for name, m in metas.items() if m.get("backend") == "cuda_worker"},
        "bootstrap_draws": 20000,
        "figure": "curve.png",
        "uses_hidden_state_in_prompt": False,
        "uses_gold_plan": False,
        "included_runs": [p.name for p in runs],
        "invalid_runs_excluded": sorted(EXCLUDED_RUNS),
        "interpretation": "exploratory public-observation smoke; models and task families are not pooled for a broad interactive-success claim",
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT} ({len(rows)} valid rows)")


if __name__ == "__main__":
    main()
