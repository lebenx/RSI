"""Build one auditable ledger from the archived ScienceWorld episodes.

This module makes no model or environment calls.  It deliberately keeps the
fresh primary LLM-readout comparison separate from the fresh-key algorithmic
success-readout extension, since those are different planner readouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results_submission/report")
SOURCES = [
    # Refreshed six-variation per-family comparison: flat vs LLM provenance.
    ("refreshed_llm", "find-plant", "scienceworld_recharged_rows.csv", ("flat", "provenance")),
    ("refreshed_llm", "find-animal", "scienceworld_recharged_animal/scienceworld_recharged_rows.csv", ("flat", "provenance")),
    # Fresh-key contiguous extension: flat vs algorithmic success-aware readout.
    ("fresh_key_success", "find-plant", "scienceworld_recharged_extension/rows.csv", ("flat", "provenance_success")),
    ("fresh_key_success", "find-animal", "scienceworld_recharged_animal_extension/rows.csv", ("flat", "provenance_success")),
]


def _read_sources() -> pd.DataFrame:
    frames = []
    for protocol, task_family, relative, methods in SOURCES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame = frame[frame.experiment_method.isin(methods)].copy()
        frame["protocol"] = protocol
        frame["task_family"] = task_family
        frame["source_file"] = str(path)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["variation"] = out["variation"].astype(int)
    out["duplication"] = out["duplication"].astype(int)
    out["valid"] = out["error"].isna()
    return out.sort_values(["protocol", "task_family", "variation", "experiment_method", "duplication"])


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sample.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _confidence(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in frame[frame.valid].iterrows():
        path = Path(str(row.get("trace_file", "")))
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
                "protocol": row.protocol,
                "task_family": row.task_family,
                "variation": int(row.variation),
                "experiment_method": row.experiment_method,
                "duplication": int(row.duplication),
                "root_confidence_first": values[0],
                "root_confidence_last": values[-1],
                "root_confidence_delta": values[-1] - values[0],
                "confidence_steps": len(values),
            })
    return pd.DataFrame(records)


def _actions(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in frame[frame.valid].groupby(["protocol", "task_family", "experiment_method", "variation"]):
        traces = {}
        for duplication in (1, 4):
            selected = group[group.duplication == duplication]
            if selected.empty:
                continue
            path = Path(str(selected.iloc[0].trace_file))
            if not path.exists():
                continue
            actions = []
            for line in path.read_text().splitlines():
                if line.strip():
                    actions.append(json.loads(line)["action"])
            traces[duplication] = actions
        if 1 not in traces or 4 not in traces:
            continue
        n = min(len(traces[1]), len(traces[4]))
        flips = sum(a != b for a, b in zip(traces[1][:n], traces[4][:n]))
        protocol, task_family, method, variation = keys
        records.append({
            "protocol": protocol,
            "task_family": task_family,
            "experiment_method": method,
            "variation": int(variation),
            "aligned_steps": n,
            "action_flips": flips,
            "action_flip_rate": flips / n if n else float("nan"),
            "m1_steps": len(traces[1]),
            "m4_steps": len(traces[4]),
        })
    return pd.DataFrame(records)


def _paired(valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_records = []
    bootstrap_records = []
    for (protocol, task_family, duplication), group in valid.groupby(["protocol", "task_family", "duplication"]):
        wide = group.pivot_table(index=["variation"], columns="experiment_method",
                                  values=["success", "final_score", "reward", "steps", "repeated_no_visible_change"],
                                  aggfunc="first")
        if ("reward", "flat") not in wide.columns:
            continue
        method = "provenance" if ("reward", "provenance") in wide.columns else "provenance_success"
        if ("reward", method) not in wide.columns:
            continue
        for variation, row in wide.iterrows():
            record = {"protocol": protocol, "task_family": task_family,
                      "duplication": int(duplication), "variation": int(variation),
                      "contrast": f"{method}-flat"}
            for metric in ("success", "final_score", "reward", "steps", "repeated_no_visible_change"):
                left = row.get((metric, "flat"), np.nan)
                right = row.get((metric, method), np.nan)
                record[f"flat_{metric}"] = left
                record[f"{method}_{metric}"] = right
                record[f"{method}_minus_flat_{metric}"] = right - left
            paired_records.append(record)
        for metric in ("success", "final_score", "reward", "steps", "repeated_no_visible_change"):
            values = (wide[(metric, method)] - wide[(metric, "flat")]).dropna().to_numpy(float)
            estimate, low, high = _bootstrap(values, 20260928 + int(duplication) + len(metric))
            bootstrap_records.append({
                "protocol": protocol, "task_family": task_family,
                "duplication": int(duplication), "contrast": f"{method}-flat",
                "metric": metric, "n_units": int(len(values)), "estimate": estimate,
                "ci95_low": low, "ci95_high": high, "bootstrap_draws": 20000,
            })
    return pd.DataFrame(paired_records), pd.DataFrame(bootstrap_records)


def run(output: str | Path = ROOT / "scienceworld_interactive_aggregate"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = _read_sources()
    all_rows.to_csv(output / "episodes.csv", index=False)
    valid = all_rows[all_rows.valid].copy()
    confidence = _confidence(valid)
    confidence.to_csv(output / "confidence.csv", index=False)
    actions = _actions(valid)
    actions.to_csv(output / "action_consistency.csv", index=False)
    paired, bootstrap = _paired(valid)
    paired.to_csv(output / "paired.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)

    summary = valid.groupby(["protocol", "task_family", "duplication", "experiment_method"], as_index=False).agg(
        n=("variation", "size"), success=("success", "mean"), final_score=("final_score", "mean"),
        reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"),
        repeated_no_visible_change=("repeated_no_visible_change", "mean"),
    )
    if not confidence.empty:
        conf_summary = confidence.groupby(["protocol", "task_family", "duplication", "experiment_method"], as_index=False).agg(
            confidence_episodes=("variation", "size"), root_confidence_first=("root_confidence_first", "mean"),
            root_confidence_last=("root_confidence_last", "mean"), root_confidence_delta=("root_confidence_delta", "mean"),
        )
        summary = summary.merge(conf_summary, on=["protocol", "task_family", "duplication", "experiment_method"], how="left")
    summary.to_csv(output / "summary.csv", index=False)

    metadata = {
        "generated_by": "submission.scienceworld_interactive_aggregate",
        "api_calls": 0,
        "protocols": ["refreshed_llm", "fresh_key_success"],
        "task_families": ["find-plant", "find-animal"],
        "raw_rows": int(len(all_rows)), "valid_rows": int(len(valid)),
        "error_rows": int((~all_rows.valid).sum()),
        "valid_episode_units": int(valid.groupby(["protocol", "task_family", "variation", "experiment_method", "duplication"]).ngroups),
        "interpretation": "consolidated audit; separate planner readouts remain stratified and broad success gain is not asserted",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return summary, paired, bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "scienceworld_interactive_aggregate"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
