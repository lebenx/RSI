"""Generate machine-readable main tables for all ScienceWorld strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results_submission/report")


def _summary(path, protocol, task):
    frame = pd.read_csv(ROOT / path)
    frame["protocol"] = protocol
    frame["task_family"] = task
    return frame


def _action(path, protocol, task):
    path = ROOT / path
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["protocol"] = protocol
    frame["task_family"] = task
    frame = frame.rename(columns={"method": "experiment_method", "estimate": "action_flip_rate"})
    return frame[["protocol", "task_family", "experiment_method", "action_flip_rate", "ci95_low", "ci95_high"]]


def _contrast(path, protocol, task, method, column):
    frame = pd.read_csv(ROOT / path)
    expected = f"{method}-flat"
    if "contrast" in frame.columns:
        frame = frame[frame.contrast == expected].copy()
    n_column = "n_units" if "n_units" in frame.columns else "n"
    rows = []
    for _, row in frame.iterrows():
        rows.append({"protocol": protocol, "task_family": task,
                     "duplication": int(row["duplication"]), "contrast": f"{method}-flat",
                     "metric": row["metric"], "n_units": int(row[n_column]),
                     "estimate": float(row[column]), "ci95_low": float(row["ci95_low"]),
                     "ci95_high": float(row["ci95_high"]), "bootstrap_draws": int(row["bootstrap_draws"])})
    return pd.DataFrame(rows)


def run(output="results_submission/report/interactive_main_table"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    summaries = [
        _summary("scienceworld_recharged_summary.csv", "refreshed_llm", "find-plant"),
        _summary("scienceworld_recharged_animal/scienceworld_recharged_summary.csv", "refreshed_llm", "find-animal"),
        _summary("scienceworld_recharged_extension/summary.csv", "fresh_key_success", "find-plant"),
        _summary("scienceworld_recharged_animal_extension/summary.csv", "fresh_key_success", "find-animal"),
        _summary("scienceworld_recharged_llm_extension/summary.csv", "fresh_key_llm", "find-plant"),
        _summary("scienceworld_recharged_animal_llm_extension/summary.csv", "fresh_key_llm", "find-animal"),
    ]
    summary = pd.concat(summaries, ignore_index=True)
    refreshed_confidence = pd.concat([
        pd.read_csv(ROOT / "scienceworld_recharged_comparison.csv"),
        pd.read_csv(ROOT / "scienceworld_recharged_animal/scienceworld_recharged_comparison.csv"),
    ], ignore_index=True)
    confidence_columns = ["task_family", "duplication", "experiment_method",
                          "confidence_episodes", "root_confidence_first",
                          "root_confidence_last", "root_confidence_delta", "confidence_steps"]
    summary = summary.drop(columns=[c for c in confidence_columns if c in summary.columns and c not in {"task_family", "duplication", "experiment_method"}], errors="ignore")
    summary = summary.merge(refreshed_confidence[confidence_columns],
                            on=["task_family", "duplication", "experiment_method"], how="left")
    action = pd.concat([
        _action("scienceworld_recharged_action_bootstrap.csv", "refreshed_llm", "find-plant"),
        _action("scienceworld_recharged_animal/scienceworld_recharged_action_bootstrap.csv", "refreshed_llm", "find-animal"),
        _action("scienceworld_recharged_extension/action_bootstrap.csv", "fresh_key_success", "find-plant"),
        _action("scienceworld_recharged_animal_extension/action_bootstrap.csv", "fresh_key_success", "find-animal"),
        _action("scienceworld_recharged_llm_extension/action_bootstrap.csv", "fresh_key_llm", "find-plant"),
        _action("scienceworld_recharged_animal_llm_extension/action_bootstrap.csv", "fresh_key_llm", "find-animal"),
    ], ignore_index=True)
    if not action.empty:
        summary = summary.merge(action, on=["protocol", "task_family", "experiment_method"], how="left")
    summary.to_csv(output / "summary.csv", index=False)

    contrasts = pd.concat([
        _contrast("scienceworld_recharged_paired.csv", "refreshed_llm", "find-plant", "provenance", "estimate"),
        _contrast("scienceworld_recharged_animal/scienceworld_recharged_paired.csv", "refreshed_llm", "find-animal", "provenance", "estimate"),
        _contrast("scienceworld_recharged_extension/paired_bootstrap.csv", "fresh_key_success", "find-plant", "provenance_success", "provenance_success_minus_flat"),
        _contrast("scienceworld_recharged_animal_extension/paired_bootstrap.csv", "fresh_key_success", "find-animal", "provenance_success", "provenance_success_minus_flat"),
        _contrast("scienceworld_recharged_llm_extension/paired_bootstrap.csv", "fresh_key_llm", "find-plant", "provenance", "provenance_minus_flat"),
        _contrast("scienceworld_recharged_animal_llm_extension/paired_bootstrap.csv", "fresh_key_llm", "find-animal", "provenance", "provenance_minus_flat"),
    ], ignore_index=True)
    contrasts.to_csv(output / "contrasts.csv", index=False)
    metadata = {
        "generated_by": "submission.interactive_main_table",
        "protocols": ["refreshed_llm", "fresh_key_success", "fresh_key_llm"],
        "task_families": ["find-plant", "find-animal"],
        "summary_rows": int(len(summary)), "contrast_rows": int(len(contrasts)),
        "methods": sorted(summary.experiment_method.unique().tolist()),
        "interpretation": "machine-generated real-environment main table; planner readouts remain stratified and broad success gain is not asserted",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return summary, contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_submission/report/interactive_main_table")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
