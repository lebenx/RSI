"""Summarize fresh-key v9 ScienceWorld extensions without making API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUTS = {
    ("flat", 1): ("results_submission/scienceworld_recharged_extension_v9_m1_156_161",),
    ("provenance_success", 1): ("results_submission/scienceworld_recharged_extension_v9_m1_156_161",),
    ("flat", 4): ("results_submission/scienceworld_recharged_extension_flat_m4_156",
                   "results_submission/scienceworld_recharged_extension_flat_m4_157_161"),
    ("provenance_success", 4): ("results_submission/scienceworld_recharged_extension_v9_m4_156_161",),
}
ANIMAL_INPUTS = {
    ("flat", 1): ("results_submission/scienceworld_recharged_animal_extension_v9_m1_156_probe",
                   "results_submission/scienceworld_recharged_animal_extension_v9_m1_157_161"),
    ("provenance_success", 1): ("results_submission/scienceworld_recharged_animal_extension_v9_m1_156_probe",
                                 "results_submission/scienceworld_recharged_animal_extension_v9_m1_157_161"),
    ("flat", 4): ("results_submission/scienceworld_recharged_animal_extension_v9_m4_156_161",),
    ("provenance_success", 4): ("results_submission/scienceworld_recharged_animal_extension_v9_m4_156_161",),
}
VARIATIONS = [156, 157, 158, 159, 160, 161]


def load_rows(inputs=INPUTS):
    rows = []
    for (method, duplication), directories in inputs.items():
        for directory in directories:
            path = Path(directory) / "summaries.json"
            for row in json.loads(path.read_text()):
                if row.get("method") != method or int(row.get("duplication", -1)) != duplication:
                    continue
                rows.append({**row, "experiment_method": method,
                             "source_directory": directory})
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


def summarize(frame, output, task_family="find-plant", inputs=INPUTS):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    valid = frame[frame.error.isna()].copy()
    frame.to_csv(output / "rows.csv", index=False)
    summary = valid.groupby(["duplication", "experiment_method"], as_index=False).agg(
        n=("task", "size"), success=("success", "mean"), final_score=("final_score", "mean"),
        reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"),
    )
    summary.to_csv(output / "summary.csv", index=False)
    wide = valid.pivot_table(index=["task", "variation"], columns="experiment_method",
                             values=["success", "final_score", "reward", "steps"], aggfunc="first")
    paired = []
    for (task, variation), row in wide.iterrows():
        record = {"task": task, "variation": int(variation)}
        for metric in ("success", "final_score", "reward", "steps"):
            for method in ("flat", "provenance_success"):
                record[f"{method}_{metric}"] = row.get((metric, method), np.nan)
            record[f"provenance_success_minus_flat_{metric}"] = record[f"provenance_success_{metric}"] - record[f"flat_{metric}"]
        paired.append(record)
    paired = pd.DataFrame(paired).sort_values("variation")
    paired.to_csv(output / "paired.csv", index=False)
    consistency = []
    for method in ("flat", "provenance_success"):
        for variation in VARIATIONS:
            traces = {}
            for duplication in (1, 4):
                selected = frame[(frame.experiment_method == method) &
                                 (frame.variation == variation) &
                                 (frame.duplication == duplication)]
                if selected.empty or selected.iloc[0].get("error") is not None:
                    continue
                path = Path(str(selected.iloc[0]["trace_file"]))
                if not path.exists():
                    continue
                traces[duplication] = [json.loads(line)["action"] for line in path.read_text().splitlines() if line.strip()]
            if 1 in traces and 4 in traces:
                n = min(len(traces[1]), len(traces[4]))
                flips = sum(a != b for a, b in zip(traces[1][:n], traces[4][:n]))
                consistency.append({"method": method, "variation": variation,
                                    "aligned_steps": n, "action_flips": flips,
                                    "action_flip_rate": flips / n if n else np.nan})
    consistency = pd.DataFrame(consistency)
    consistency.to_csv(output / "action_consistency.csv", index=False)
    action_bootstrap = []
    for method, group in consistency.groupby("method"):
        values = group.action_flip_rate.dropna().to_numpy(float)
        estimate, low, high = bootstrap(values, 20360928 + len(action_bootstrap))
        action_bootstrap.append({"method": method, "n_variations": len(values),
                                 "estimate": estimate, "ci95_low": low, "ci95_high": high,
                                 "bootstrap_draws": 20000})
    pd.DataFrame(action_bootstrap).to_csv(output / "action_bootstrap.csv", index=False)
    contrasts = []
    for duplication in (1, 4):
        z = valid[valid.duplication == duplication].pivot_table(index=["task", "variation"], columns="experiment_method", values=["success", "final_score", "reward", "steps"], aggfunc="first")
        for metric in ("success", "final_score", "reward", "steps"):
            diff = (z[(metric, "provenance_success")] - z[(metric, "flat")]).dropna().to_numpy(float)
            estimate, low, high = bootstrap(diff, 20260928 + duplication + len(metric))
            contrasts.append({"duplication": duplication, "metric": metric, "n_units": len(diff),
                              "provenance_success_minus_flat": estimate, "ci95_low": low,
                              "ci95_high": high, "bootstrap_draws": 20000})
    pd.DataFrame(contrasts).to_csv(output / "paired_bootstrap.csv", index=False)
    request_files = 0
    for directory in inputs.values():
        for root in directory:
            request_files += len(list((Path(root) / "requests").glob("*.json")))
    metadata = {
        "provider": "DeepSeek", "model": "deepseek-chat", "api_probe_status": 200,
        "protocol": "sw-llm-v9-public-discovery-fair", "task": task_family,
        "variations": VARIATIONS, "duplications": [1, 4], "methods": ["flat", "provenance_success"],
        "valid_rows": int(len(valid)), "error_rows": int(frame.error.notna().sum()),
        "new_episode_rows": int(len(valid)), "network_request_files": request_files,
        "interpretation": "fresh-key extension; null/negative boundary, not a broad planner claim",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return frame, summary, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_submission/report/scienceworld_recharged_extension")
    parser.add_argument("--task", choices=("find-plant", "find-animal"), default="find-plant")
    args = parser.parse_args()
    inputs = INPUTS if args.task == "find-plant" else ANIMAL_INPUTS
    summarize(load_rows(inputs), args.output, args.task, inputs)


if __name__ == "__main__":
    main()
