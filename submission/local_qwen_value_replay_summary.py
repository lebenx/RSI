"""Summarize the cached-request local-Qwen provenance-value replay.

Candidate and conditional-bank requests are copied byte-for-byte from the
expanded flat/provenance-success run.  The replay changes only the deterministic
aggregation objective, so it adds no new LLM calls and isolates the core value
aggregator on the same public ScienceWorld states.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIATIONS = [152, 153, 154, 158, 161]
ROOT = Path("results_submission")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cached_summaries(directory: Path, duplication: int) -> pd.DataFrame:
    rows = []
    for row in json.loads((directory / "summaries.json").read_text()):
        if row.get("error") is not None:
            continue
        row = dict(row)
        row["duplication"] = duplication
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(values: np.ndarray, seed: int = 20260929, draws: int = 20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def cache_audit(source: Path, destination: Path) -> dict:
    src = {p.name: digest(p) for p in (source / "requests").glob("*.json")}
    dst = {p.name: digest(p) for p in (destination / "requests").glob("*.json")}
    return {"source": str(source), "destination": str(destination),
            "source_requests": len(src), "destination_requests": len(dst),
            "byte_identical": src == dst, "new_request_files": sorted(set(dst) - set(src)),
            "changed_request_files": sorted(k for k in set(src) & set(dst) if src[k] != dst[k])}


def main() -> None:
    out = Path("results_submission/report/local_qwen_value_replay")
    out.mkdir(parents=True, exist_ok=True)
    expanded = pd.read_csv(Path("results_submission/report/local_qwen_expanded_pilot/rows.csv"))
    flat = expanded[(expanded.method == "flat") & expanded.variation.astype(int).isin(VARIATIONS)].copy()
    flat["method"] = "flat"
    m1_paths = [ROOT / "scienceworld_local_qwen_provenance_value_current_m1", ROOT / "scienceworld_local_qwen_provenance_value_158_m1", ROOT / "scienceworld_local_qwen_provenance_value_161_m1"]
    m4_paths = [ROOT / "scienceworld_local_qwen_provenance_value_current_m4", ROOT / "scienceworld_local_qwen_provenance_value_158_m4", ROOT / "scienceworld_local_qwen_provenance_value_161_m4"]
    m1 = pd.concat([load_cached_summaries(p, 1) for p in m1_paths], ignore_index=True)
    m4 = pd.concat([load_cached_summaries(p, 4) for p in m4_paths], ignore_index=True)
    value = pd.concat([m1, m4], ignore_index=True)
    value = value[value.variation.astype(int).isin(VARIATIONS)].copy()
    value["method"] = "provenance_value"
    frame = pd.concat([flat, value], ignore_index=True, sort=False)
    frame.to_csv(out / "rows.csv", index=False)
    summary = frame.groupby(["duplication", "method"], as_index=False).agg(
        n=("variation", "size"), success=("success", "mean"),
        final_score=("final_score", "mean"), reward=("reward", "mean"),
        steps=("steps", "mean"), tokens=("charged_tokens", "mean"))
    summary.to_csv(out / "summary.csv", index=False)
    wide = frame.pivot_table(index=["task", "variation"], columns=["method", "duplication"],
                              values=["success", "final_score", "reward", "steps", "charged_tokens"], aggfunc="first")
    paired = []
    for (task, variation), row in wide.iterrows():
        rec = {"task": task, "variation": int(variation)}
        if ("reward", "flat", 1) not in row.index or ("reward", "provenance_value", 4) not in row.index:
            continue
        for metric in ("success", "final_score", "reward", "steps", "charged_tokens"):
            for dup in (1, 4):
                rec[f"flat_m{dup}_{metric}"] = row.get((metric, "flat", dup))
                rec[f"provenance_value_m{dup}_{metric}"] = row.get((metric, "provenance_value", dup))
                rec[f"provenance_value_minus_flat_m{dup}_{metric}"] = rec[f"provenance_value_m{dup}_{metric}"] - rec[f"flat_m{dup}_{metric}"]
        paired.append(rec)
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)
    contrasts = []
    for metric in ("success", "reward", "steps"):
        for dup in (1, 4):
            values = paired[f"provenance_value_minus_flat_m{dup}_{metric}"].dropna().to_numpy(float)
            est, lo, hi = bootstrap(values)
            contrasts.append({"duplication": dup, "metric": metric, "n_episodes": len(values), "estimate": est,
                              "ci95_low": lo, "ci95_high": hi, "bootstrap_draws": 20000})
    pd.DataFrame(contrasts).to_csv(out / "bootstrap.csv", index=False)
    audits = [cache_audit(ROOT / "scienceworld_local_qwen_grid150_155_v9_m1", ROOT / "scienceworld_local_qwen_provenance_value_current_m1"),
              cache_audit(ROOT / "scienceworld_local_qwen_grid150_155_v9_m4", ROOT / "scienceworld_local_qwen_provenance_value_current_m4"),
              cache_audit(ROOT / "scienceworld_local_qwen_value_grid_v9_m1", ROOT / "scienceworld_local_qwen_provenance_value_158_m1"),
              cache_audit(ROOT / "scienceworld_local_qwen_value_grid_m4", ROOT / "scienceworld_local_qwen_provenance_value_158_m4"),
              cache_audit(ROOT / "scienceworld_local_qwen_value_grid_v9_m1", ROOT / "scienceworld_local_qwen_provenance_value_161_m1"),
              cache_audit(ROOT / "scienceworld_local_qwen_value161_v9_m4", ROOT / "scienceworld_local_qwen_provenance_value_161_m4")]
    (out / "cache_audit.json").write_text(json.dumps(audits, indent=2))
    metadata = {"protocol": "local-qwen-cached-provenance-value-replay-v1", "model": "Qwen2.5-Coder-3B-Instruct",
                "task": "find-plant", "variations": VARIATIONS, "duplications": [1, 4],
                "methods": ["flat", "provenance_value"], "valid_rows": int(len(frame)),
                "paired_episodes": int(len(paired)), "new_model_requests": 0,
                "cache_audit": "cache_audit.json", "planner_claim": False,
                "interpretation": "same cached candidate/bank, deterministic provenance-value objective; exploratory second-model replay",
                "cuda_run_policy": "LOCAL_QWEN_DEVICE=cuda; runner loaded on CUDA while all model requests were cache hits"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False)); print(summary.to_string(index=False)); print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
