"""Create a non-pooled cross-model interactive effect table.

DeepSeek and local Qwen rows are kept in separate strata.  The table is a
heterogeneity display, not a random-effects meta-analysis: protocols, task
families, objectives, and model-specific readouts are never pooled.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("results_submission/report")


def main() -> None:
    out = ROOT / "cross_model_interactive_table"
    out.mkdir(parents=True, exist_ok=True)
    deep = pd.read_csv(ROOT / "interactive_main_table/contrasts.csv")
    deep = deep[deep.metric.isin(["success", "reward", "steps", "repeated_no_visible_change"])].copy()
    deep["metric"] = deep["metric"].replace({"repeated_no_visible_change": "unnecessary_actions"})
    deep["model"] = "DeepSeek"
    deep["objective"] = deep["contrast"].map({"provenance-flat": "provenance", "provenance_success-flat": "provenance_success"})
    deep = deep.rename(columns={"n_units": "n_episodes"})
    deep = deep[["model", "protocol", "task_family", "duplication", "objective", "metric", "n_episodes", "estimate", "ci95_low", "ci95_high", "bootstrap_draws"]]

    # Use the expanded CUDA-Qwen table when present; it contains the original
    # eight paired episodes plus six fresh plant/animal pairs.  Keeping all
    # reported metrics is intentional: calibration and token-cost rows are
    # part of the model-specific protocol appendix.
    qwen_path = ROOT / "local_qwen_interactive_expanded_table_v4/contrasts.csv"
    if not qwen_path.exists():
        qwen_path = ROOT / "local_qwen_interactive_expanded_table/contrasts.csv"
    if not qwen_path.exists():
        qwen_path = ROOT / "local_qwen_interactive_main_table/contrasts.csv"
    qwen = pd.read_csv(qwen_path)
    qwen_meta_path = qwen_path.parent / "metadata.json"
    qwen_meta = json.loads(qwen_meta_path.read_text()) if qwen_meta_path.exists() else {}
    qwen["model"] = "Qwen2.5-Coder-3B"
    qwen["protocol"] = "qwen_core_cuda"
    qwen["objective"] = "provenance_value"
    qwen = qwen.rename(columns={"n_episodes": "n_episodes"})
    qwen = qwen[["model", "protocol", "task_family", "duplication", "objective", "metric", "n_episodes", "estimate", "ci95_low", "ci95_high", "bootstrap_draws"]]

    # ALFWorld is an independent public-observation protocol. Keep its
    # episode-cluster contrasts as separate environment strata and never pool
    # them with ScienceWorld or across models.
    alf = pd.read_csv(ROOT / "alfworld_interactive/bootstrap.csv")
    alf = alf[alf.task_type == "all"].copy()
    alf["protocol"] = "alfworld_public"
    alf["task_family"] = "ALFWorld-all"
    alf["objective"] = alf["contrast"].map({
        "provenance_value_minus_flat": "provenance_value",
        "provenance_success_minus_flat": "provenance_success",
    })
    alf = alf[["model", "protocol", "task_family", "duplication", "objective", "metric", "n_episodes", "estimate", "ci95_low", "ci95_high", "bootstrap_draws"]]

    frame = pd.concat([deep, qwen, alf], ignore_index=True)
    frame.to_csv(out / "strata.csv", index=False)
    # A compact range table is useful in the paper, but remains stratified.
    ranges = frame.groupby(["model", "objective", "metric"], as_index=False).agg(
        strata=("protocol", "nunique"), n_rows=("estimate", "size"),
        estimate_min=("estimate", "min"), estimate_max=("estimate", "max"),
        ci_low_min=("ci95_low", "min"), ci_high_max=("ci95_high", "max"))
    ranges.to_csv(out / "ranges.csv", index=False)
    metadata = {
        "protocol": "cross-model-interactive-stratified-v3", "models": ["DeepSeek", "Qwen2.5-Coder-3B"],
        "pooled": False, "deepseek_rows": int(len(deep) + len(alf[alf.model == "DeepSeek"])),
        "qwen_rows": int(len(qwen) + len(alf[alf.model == "Qwen2.5-Coder-3B"])),
        "alfworld_rows": int(len(alf)),
        "rows": int(len(frame)), "metrics": sorted(frame.metric.unique().tolist()),
        "qwen_source_table": str(qwen_path),
        "qwen_paired_episodes": int(qwen_meta.get("paired_episodes", 14 if "expanded" in str(qwen_path.parent) else 8)),
        "interpretation": "model/protocol/task-family strata are displayed without pooled superiority claim",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False)); print(ranges.to_string(index=False))


if __name__ == "__main__":
    main()
