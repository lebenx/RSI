"""Summarize the fresh-key LLM-readout ScienceWorld extension.

The flat rows are the already archived public-action extension.  Only the
provenance readout rows are newly generated; candidate and rollout-bank
requests were reused from the archived cache.  No calls are made here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK_INPUTS = {
    "find-plant": {
        "flat": {1: ("results_submission/scienceworld_recharged_extension_v9_m1_156_161",),
                 4: ("results_submission/scienceworld_recharged_extension_flat_m4_156",
                     "results_submission/scienceworld_recharged_extension_flat_m4_157_161")},
        "provenance": {1: ("results_submission/scienceworld_recharged_extension_llm_prov_m1_156_161",),
                        4: ("results_submission/scienceworld_recharged_extension_llm_prov_m4_156_161",)},
    },
    "find-animal": {
        "flat": {1: ("results_submission/scienceworld_recharged_animal_extension_v9_m1_156_probe",
                      "results_submission/scienceworld_recharged_animal_extension_v9_m1_157_161"),
                 4: ("results_submission/scienceworld_recharged_animal_extension_v9_m4_156_161",)},
        "provenance": {1: ("results_submission/scienceworld_recharged_animal_extension_llm_prov_m1_156_161",),
                        4: ("results_submission/scienceworld_recharged_animal_extension_llm_prov_m4_156_161",)},
    },
}
VARIATIONS = [156, 157, 158, 159, 160, 161]


def load_rows(task="find-plant"):
    inputs = TASK_INPUTS[task]
    rows = []
    for duplication, directories in inputs["flat"].items():
        for directory in directories:
            data = json.loads((Path(directory) / "summaries.json").read_text())
            rows.extend({**row, "experiment_method": "flat", "source_directory": directory}
                        for row in data if row.get("method") == "flat" and int(row.get("duplication", -1)) == duplication)
    for duplication, directories in inputs["provenance"].items():
        for directory in directories:
            data = json.loads((Path(directory) / "summaries.json").read_text())
            rows.extend({**row, "experiment_method": "provenance", "source_directory": directory}
                        for row in data if row.get("method") == "provenance" and int(row.get("duplication", -1)) == duplication)
    frame = pd.DataFrame(rows)
    frame["variation"] = frame["variation"].astype(int)
    frame["duplication"] = frame["duplication"].astype(int)
    return frame.sort_values(["experiment_method", "duplication", "variation"])


def bootstrap(values, seed, draws=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sample.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def confidence(frame):
    rows = []
    for _, row in frame[frame.error.isna()].iterrows():
        path = Path(str(row.trace_file))
        values = []
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    candidate = json.loads(line).get("candidate", {})
                    if isinstance(candidate, dict) and "p_true" in candidate:
                        values.append(float(candidate["p_true"]))
        if values:
            rows.append({"experiment_method": row.experiment_method, "duplication": int(row.duplication),
                         "variation": int(row.variation), "root_confidence_first": values[0],
                         "root_confidence_last": values[-1], "root_confidence_delta": values[-1] - values[0],
                         "confidence_steps": len(values)})
    return pd.DataFrame(rows)


def action_consistency(frame):
    rows = []
    for method in ("flat", "provenance"):
        for variation in VARIATIONS:
            traces = {}
            for duplication in (1, 4):
                selected = frame[(frame.experiment_method == method) & (frame.variation == variation) & (frame.duplication == duplication)]
                if selected.empty or not pd.isna(selected.iloc[0].error):
                    continue
                path = Path(str(selected.iloc[0].trace_file))
                if path.exists():
                    traces[duplication] = [json.loads(x)["action"] for x in path.read_text().splitlines() if x.strip()]
            if 1 in traces and 4 in traces:
                n = min(len(traces[1]), len(traces[4]))
                flips = sum(a != b for a, b in zip(traces[1][:n], traces[4][:n]))
                rows.append({"method": method, "variation": variation, "aligned_steps": n,
                             "action_flips": flips, "action_flip_rate": flips / n if n else np.nan,
                             "m1_steps": len(traces[1]), "m4_steps": len(traces[4])})
    return pd.DataFrame(rows)


def run(output="results_submission/report/scienceworld_recharged_llm_extension", task="find-plant"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frame = load_rows(task)
    frame.to_csv(output / "rows.csv", index=False)
    valid = frame[frame.error.isna()].copy()
    summary = valid.groupby(["duplication", "experiment_method"], as_index=False).agg(
        n=("variation", "size"), success=("success", "mean"), final_score=("final_score", "mean"),
        reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"),
        repeated_no_visible_change=("repeated_no_visible_change", "mean"))
    conf = confidence(frame)
    conf.to_csv(output / "confidence.csv", index=False)
    if not conf.empty:
        c = conf.groupby(["duplication", "experiment_method"], as_index=False).agg(
            confidence_episodes=("variation", "size"), root_confidence_first=("root_confidence_first", "mean"),
            root_confidence_last=("root_confidence_last", "mean"), root_confidence_delta=("root_confidence_delta", "mean"))
        summary = summary.merge(c, on=["duplication", "experiment_method"], how="left")
    summary.to_csv(output / "summary.csv", index=False)

    wide = valid.pivot_table(index="variation", columns="experiment_method",
                             values=["success", "final_score", "reward", "steps", "repeated_no_visible_change"], aggfunc="first")
    paired_rows = []
    boot_rows = []
    for variation, row in wide.iterrows():
        record = {"task": task, "variation": int(variation)}
        for metric in ("success", "final_score", "reward", "steps", "repeated_no_visible_change"):
            record[f"flat_{metric}"] = row.get((metric, "flat"), np.nan)
            record[f"provenance_{metric}"] = row.get((metric, "provenance"), np.nan)
            record[f"provenance_minus_flat_{metric}"] = record[f"provenance_{metric}"] - record[f"flat_{metric}"]
        paired_rows.append(record)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "paired.csv", index=False)
    for duplication in (1, 4):
        z = valid[valid.duplication == duplication].pivot_table(index="variation", columns="experiment_method",
            values=["success", "final_score", "reward", "steps", "repeated_no_visible_change"], aggfunc="first")
        for metric in ("success", "final_score", "reward", "steps", "repeated_no_visible_change"):
            values = (z[(metric, "provenance")] - z[(metric, "flat")]).dropna().to_numpy(float)
            estimate, low, high = bootstrap(values, 20260928 + duplication + len(metric))
            boot_rows.append({"duplication": duplication, "metric": metric, "n_units": len(values),
                              "provenance_minus_flat": estimate, "ci95_low": low, "ci95_high": high,
                              "bootstrap_draws": 20000})
    pd.DataFrame(boot_rows).to_csv(output / "paired_bootstrap.csv", index=False)
    actions = action_consistency(frame)
    actions.to_csv(output / "action_consistency.csv", index=False)
    action_boot = []
    for method, group in actions.groupby("method"):
        estimate, low, high = bootstrap(group.action_flip_rate.to_numpy(float), 20360928 + len(action_boot))
        action_boot.append({"method": method, "n_variations": len(group), "estimate": estimate,
                            "ci95_low": low, "ci95_high": high, "bootstrap_draws": 20000})
    pd.DataFrame(action_boot).to_csv(output / "action_bootstrap.csv", index=False)
    inputs = TASK_INPUTS[task]
    request_files = sum(len(list((Path(d) / "requests").glob("*.json"))) for method_inputs in inputs.values() for dirs in method_inputs.values() for d in dirs)
    metadata = {"provider": "DeepSeek", "model": "deepseek-chat", "api_probe_status": 200,
                "protocol": "sw-llm-v9-public-discovery-fair", "task": task,
                "variations": VARIATIONS, "duplications": [1, 4], "methods": ["flat", "provenance"],
                "valid_rows": int(len(valid)), "error_rows": int(frame.error.notna().sum()),
                "new_provenance_episode_rows": 12, "new_api_readout_rows": 12,
                "request_files_in_reused_and_new_caches": request_files,
                "interpretation": "fresh-key fair LLM-readout extension; separate from success-aware algorithmic extension",
                "broad_success_claim": False}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--task", choices=tuple(TASK_INPUTS), default="find-plant")
    args = parser.parse_args()
    output = args.output or ("results_submission/report/scienceworld_recharged_llm_extension" if args.task == "find-plant" else "results_submission/report/scienceworld_recharged_animal_llm_extension")
    run(output, args.task)


if __name__ == "__main__":
    main()
