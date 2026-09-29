"""Summarize the refreshed DeepSeek ScienceWorld comparison.

The run uses the public-action shortlist and the same runner protocol at
duplication 1 and 4.  This module only reads archived summaries/traces and
therefore cannot silently add API calls or change the evaluation sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("results_submission")
VARIATIONS = [150, 151, 152, 153, 154, 155]
RUNS = {
    ("flat", 1): "scienceworld_recharged_findplant_m1",
    ("flat", 4): "scienceworld_recharged_findplant_m4",
    ("provenance", 1): "scienceworld_recharged_findplant_m1",
    ("provenance", 4): "scienceworld_recharged_findplant_m4",
    ("provenance_success", 1): "scienceworld_recharged_findplant_success_m1_api",
    ("provenance_success", 4): "scienceworld_recharged_findplant_success_m4_api",
    ("provenance_value", 1): "scienceworld_recharged_findplant_value_m1",
    ("provenance_value", 4): "scienceworld_recharged_findplant_value_m4",
    ("no_imagination", 1): "scienceworld_recharged_findplant_m1",
    ("no_imagination", 4): "scienceworld_recharged_findplant_m4",
}
RUNS_BY_TASK = {
    "find-plant": RUNS,
    "find-animal": {
        ("flat", 1): "scienceworld_recharged_v9_findanimal_m1",
        ("flat", 4): "scienceworld_recharged_v9_findanimal_m4",
        ("provenance", 1): "scienceworld_recharged_v9_findanimal_m1",
        ("provenance", 4): "scienceworld_recharged_v9_findanimal_m4",
        ("no_imagination", 1): "scienceworld_recharged_v9_findanimal_m1",
        ("no_imagination", 4): "scienceworld_recharged_v9_findanimal_m4",
    },
}


def load_rows(task: str = "find-plant") -> pd.DataFrame:
    runs = RUNS_BY_TASK[task]
    rows = []
    for (method, duplication), run_name in runs.items():
        path = ROOT / run_name / "summaries.json"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in json.loads(path.read_text()):
            if row.get("method") != method:
                continue
            if int(row.get("duplication", -1)) != duplication:
                continue
            if row.get("error") is not None:
                continue
            rows.append({**row, "task_family": task, "experiment_method": method,
                         "run_name": run_name})
    # One v9 animal provenance row was retried after a transient SSL failure;
    # prefer the successful retry over the retained failed primary row.
    retry = ROOT / "scienceworld_recharged_v9_findanimal_m1_retry2" / "summaries.json"
    if task == "find-animal" and retry.exists():
        for row in json.loads(retry.read_text()):
            if row.get("method") == "provenance" and row.get("error") is None:
                rows = [x for x in rows if not (x.get("task") == "find-animal" and int(x.get("variation", -1)) == int(row.get("variation", -2)) and x.get("experiment_method") == "provenance" and int(x.get("duplication", -1)) == 1)]
                rows.append({**row, "task_family": task, "experiment_method": "provenance",
                             "run_name": "scienceworld_recharged_v9_findanimal_m1_retry2"})
    frame = pd.DataFrame(rows)
    required = {"task", "variation", "experiment_method", "duplication",
                "success", "final_score", "reward", "steps",
                "charged_tokens", "repeated_no_visible_change"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing summary fields: {sorted(missing)}")
    frame["variation"] = frame["variation"].astype(int)
    frame["duplication"] = frame["duplication"].astype(int)
    return frame.sort_values(["experiment_method", "duplication", "variation"])


def bootstrap_mean(values: np.ndarray, draws: int = 20000, seed: int = 20260928):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sample.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def paired_table(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for duplication, group in frame.groupby("duplication"):
        wide = group.pivot_table(index=["task", "variation"], columns="experiment_method",
                                 values=["success", "final_score", "reward", "steps", "repeated_no_visible_change"],
                                 aggfunc="first")
        methods = ["provenance", "provenance_success", "provenance_value"]
        for method in methods:
            if ("reward", method) not in wide.columns or ("reward", "flat") not in wide.columns:
                continue
            for metric in ("success", "final_score", "reward", "steps", "repeated_no_visible_change"):
                if (metric, method) not in wide.columns or (metric, "flat") not in wide.columns:
                    continue
                values = (wide[(metric, method)] - wide[(metric, "flat")]).dropna().to_numpy(float)
                estimate, low, high = bootstrap_mean(values, seed=20260928 + int(duplication))
                records.append({"duplication": int(duplication), "contrast": f"{method}-flat",
                                "metric": metric, "n": int(len(values)), "estimate": estimate,
                                "ci95_low": low, "ci95_high": high, "bootstrap_draws": 20000})
    return pd.DataFrame(records)


def action_consistency(frame: pd.DataFrame, task: str = "find-plant") -> pd.DataFrame:
    records = []
    methods = ("flat", "provenance", "provenance_success", "provenance_value") if task == "find-plant" else ("flat", "provenance")
    for method in methods:
        for variation in VARIATIONS:
            traces = {}
            for duplication in (1, 4):
                # Use the trace recorded in the filtered frame.  This matters
                # for the one retried v9 request: the replacement row points
                # to its retry directory rather than the failed primary run.
                selected = frame[(frame.experiment_method == method) &
                                 (frame.variation == variation) &
                                 (frame.duplication == duplication)]
                if selected.empty:
                    continue
                path = Path(str(selected.iloc[0]["trace_file"]))
                if path.exists():
                    traces[duplication] = [json.loads(line)["action"] for line in path.read_text().splitlines() if line.strip()]
            if 1 not in traces or 4 not in traces:
                continue
            n = min(len(traces[1]), len(traces[4]))
            flips = sum(a != b for a, b in zip(traces[1][:n], traces[4][:n]))
            records.append({"method": method, "variation": variation,
                            "aligned_steps": n, "action_flips": flips,
                            "action_flip_rate": flips / n if n else float("nan"),
                            "m1_steps": len(traces[1]), "m4_steps": len(traces[4])})
    return pd.DataFrame(records)


def confidence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the root candidate confidence in archived traces.

    ``candidate.p_true`` is the runner's explicit root-hypothesis readout.
    We report its first and last value for each episode and the paired change;
    no confidence is inferred from reward or success.
    """
    records = []
    for _, row in frame.iterrows():
        path = Path(str(row["trace_file"]))
        if not path.exists():
            continue
        values = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            candidate = json.loads(line).get("candidate", {})
            if isinstance(candidate, dict) and "p_true" in candidate:
                values.append(float(candidate["p_true"]))
        if values:
            records.append({
                "task_family": row["task_family"],
                "experiment_method": row["experiment_method"],
                "duplication": int(row["duplication"]),
                "variation": int(row["variation"]),
                "root_confidence_first": values[0],
                "root_confidence_last": values[-1],
                "root_confidence_delta": values[-1] - values[0],
                "confidence_steps": len(values),
            })
    episodes = pd.DataFrame(records)
    if episodes.empty:
        return episodes
    return episodes.groupby(["task_family", "experiment_method", "duplication"], as_index=False).agg(
        n=("variation", "size"),
        root_confidence_first=("root_confidence_first", "mean"),
        root_confidence_last=("root_confidence_last", "mean"),
        root_confidence_delta=("root_confidence_delta", "mean"),
        confidence_steps=("confidence_steps", "mean"),
    )


def comparison_table(summary: pd.DataFrame, confidence: pd.DataFrame,
                     consistency: pd.DataFrame, action_ci: pd.DataFrame) -> pd.DataFrame:
    """Make one compact table covering confidence, flips, and reward."""
    table = summary.copy()
    if not confidence.empty:
        conf = confidence.rename(columns={"n": "confidence_episodes"})
        table = table.merge(conf, on=["task_family", "experiment_method", "duplication"], how="left")
    else:
        for col in ("confidence_episodes", "root_confidence_first", "root_confidence_last", "root_confidence_delta", "confidence_steps"):
            table[col] = np.nan
    if action_ci.empty:
        table["action_flip_rate_m1_vs_m4"] = np.nan
        table["action_flip_ci95_low"] = np.nan
        table["action_flip_ci95_high"] = np.nan
    else:
        flips = action_ci.rename(columns={
            "estimate": "action_flip_rate_m1_vs_m4",
            "ci95_low": "action_flip_ci95_low",
            "ci95_high": "action_flip_ci95_high",
        })[["method", "action_flip_rate_m1_vs_m4", "action_flip_ci95_low", "action_flip_ci95_high"]]
        table = table.merge(flips, left_on="experiment_method", right_on="method", how="left").drop(columns=["method"])
    return table.sort_values(["task_family", "experiment_method", "duplication"])


def combine_comparison_tables(output: str | Path = ROOT / "report") -> pd.DataFrame:
    """Combine the plant and animal tables when both archived outputs exist."""
    output = Path(output)
    paths = [output / "scienceworld_recharged_comparison.csv",
             output / "scienceworld_recharged_animal" / "scienceworld_recharged_comparison.csv"]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / "scienceworld_recharged_comparison_all.csv", index=False)
    return combined


def action_bootstrap(consistency: pd.DataFrame) -> pd.DataFrame:
    records = []
    if consistency.empty:
        return pd.DataFrame(records)
    for method, group in consistency.groupby("method"):
        values = group["action_flip_rate"].dropna().to_numpy(float)
        estimate, low, high = bootstrap_mean(values, seed=20260929)
        records.append({"method": method, "n_variations": int(len(values)),
                        "estimate": estimate, "ci95_low": low,
                        "ci95_high": high, "bootstrap_draws": 20000})
    return pd.DataFrame(records)


def run(output: str | Path = ROOT / "report", task: str = "find-plant"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_rows(task)
    summary = frame.groupby(["task_family", "duplication", "experiment_method"], as_index=False).agg(
        n=("task", "size"), success=("success", "mean"),
        final_score=("final_score", "mean"), reward=("reward", "mean"),
        steps=("steps", "mean"), tokens=("charged_tokens", "mean"),
        repeated_no_visible_change=("repeated_no_visible_change", "mean"))
    frame.to_csv(output / "scienceworld_recharged_rows.csv", index=False)
    summary.to_csv(output / "scienceworld_recharged_summary.csv", index=False)
    paired = paired_table(frame)
    paired.to_csv(output / "scienceworld_recharged_paired.csv", index=False)
    consistency = action_consistency(frame, task)
    consistency.to_csv(output / "scienceworld_recharged_action_consistency.csv", index=False)
    action_ci = action_bootstrap(consistency)
    action_ci.to_csv(output / "scienceworld_recharged_action_bootstrap.csv", index=False)
    confidence = confidence_summary(frame)
    confidence.to_csv(output / "scienceworld_recharged_confidence.csv", index=False)
    comparison = comparison_table(summary, confidence, consistency, action_ci)
    comparison.to_csv(output / "scienceworld_recharged_comparison.csv", index=False)
    plot_methods = [m for m in ("no_imagination", "flat", "provenance", "provenance_success", "provenance_value") if m in set(summary.experiment_method)]
    colors = {"no_imagination": "#777777", "flat": "#c43c39", "provenance": "#276fbf", "provenance_success": "#4f9d69", "provenance_value": "#7a5aa6"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for method in plot_methods:
        z = summary[summary.experiment_method == method].sort_values("duplication")
        axes[0].plot(z.duplication, z.success, "o-", label=method, color=colors[method])
        axes[1].plot(z.duplication, z.reward, "o-", label=method, color=colors[method])
    for ax, ylabel in zip(axes, ("success rate", "reward")):
        ax.set_xlabel("duplication")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 4])
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "scienceworld_recharged_curves.png", dpi=180)
    plt.close(fig)
    metadata = {
        "provider": "DeepSeek",
        "model": "deepseek-chat",
        "api_probe_status": 200,
        "task": task,
        "variations": VARIATIONS,
        "duplications": [1, 4],
        "methods": sorted(frame.experiment_method.unique().tolist()),
        "valid_rows": int(len(frame)),
        "errors_excluded": True,
        "protocol": "sw-llm-v9-public-discovery-fair" if task == "find-animal" else "sw-llm-v8-public-discovery-fair",
        "uses_gold_path": False,
        "uses_hidden_state": False,
        "bootstrap_draws": 20000,
        "interpretation": "fresh real-model diagnostic; broad success improvement is not assumed",
    }
    (output / "scienceworld_recharged_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    if not consistency.empty:
        print(consistency.groupby("method", as_index=False).agg(
            n=("variation", "size"), action_flip_rate=("action_flip_rate", "mean")).to_string(index=False))
    return summary, paired, consistency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_submission/report")
    parser.add_argument("--task", choices=sorted(RUNS_BY_TASK), default="find-plant")
    args = parser.parse_args()
    run(args.output, args.task)


if __name__ == "__main__":
    main()
