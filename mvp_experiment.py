#!/usr/bin/env python3
"""Minimal shared-assumption duplication experiment.

The environment is intentionally small and controlled.  There are two independent
provenance groups (A and B), but both produce the same surface-level evidence:
"LEFT is safer".  Copies within a group are exact repeated rollouts, so they do
not contain new information.  The true state is RIGHT for all three seeds; this
lets us measure when confidence in the false root hypothesis (LEFT) rises enough
to flip the action.

DeepSeek is used once per seed to produce a short rollout narrative.  The numeric
likelihood ratios are fixed by the benchmark so that the causal comparison is
reproducible and does not depend on parsing a free-form model answer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


METHODS = [
    "ordinary_aggregation",
    "single_trajectory_average",
    "dedup_by_evidence",
    "provenance_preserving",
]
DUPLICATIONS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]
TRUE_STATE = "RIGHT"
HYPOTHESIS = "LEFT is safer"
EVIDENCE_SIGNATURE = "left_safer_shared_assumption_v1"


@dataclass(frozen=True)
class Rollout:
    seed: int
    source_id: str
    evidence_signature: str
    likelihood_ratio: float
    model_note: str


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def posterior_from_lrs(prior: float, lrs: list[float]) -> float:
    return sigmoid(logit(prior) + sum(math.log(lr) for lr in lrs))


def call_deepseek(seed: int, api_key: str | None, model: str) -> tuple[str, bool]:
    """Get narrative evidence from DeepSeek; return a safe fallback on failure."""
    fallback = (
        "A shared world-model assumption says the LEFT route is safer; "
        "this report is a repeated rollout and adds no new observation."
    )
    if not api_key:
        return fallback, False

    prompt = f"""
You are producing one short evidence note for a controlled planning benchmark.
The benchmark tests whether repeated rollouts from one shared world-model
assumption are mistakenly treated as independent evidence.  The root hypothesis
is: {HYPOTHESIS}.  Write one concise note (under 35 words) that supports this
hypothesis and explicitly says the rollout repeats the same assumption without a
new real observation.  Do not invent measurements, numbers, or a final action.
Return JSON only: {{"note": "..."}}.  Benchmark seed: {seed}.
""".strip()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=body,
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        note = str(parsed.get("note", "")).strip()
        if not note:
            raise ValueError("empty note")
        return note, True
    except Exception as exc:  # keep the benchmark runnable if API is unavailable
        return f"{fallback} (API fallback: {type(exc).__name__})", False


def make_rollouts(seed: int, api_key: str | None, model: str) -> tuple[float, list[Rollout], bool]:
    # Small deterministic jitter gives three independent seeds while preserving
    # the threshold crossing at duplication >= 2.
    rng = random.Random(1000 + seed)
    prior = [0.25, 0.255, 0.245][seed]
    lr_a = 1.25 + rng.uniform(-0.02, 0.02)
    lr_b = 1.80 + rng.uniform(-0.025, 0.025)
    note, api_ok = call_deepseek(seed, api_key, model)
    rollouts = [
        Rollout(seed, "source_A", EVIDENCE_SIGNATURE, lr_a, note),
        Rollout(seed, "source_B", EVIDENCE_SIGNATURE, lr_b, note),
    ]
    return prior, rollouts, api_ok


def aggregate(method: str, prior: float, copies: list[Rollout]) -> float:
    if method == "ordinary_aggregation":
        return posterior_from_lrs(prior, [r.likelihood_ratio for r in copies])

    if method == "single_trajectory_average":
        # Each rollout is scored once from the same prior; averaging these scores
        # does not count exact copies as independent observations.
        per_rollout = [posterior_from_lrs(prior, [r.likelihood_ratio]) for r in copies]
        return sum(per_rollout) / len(per_rollout)

    if method == "dedup_by_evidence":
        # Naive text/signature deduplication.  A and B have the same surface
        # evidence, so only the first representative is retained.
        seen: set[str] = set()
        unique: list[Rollout] = []
        for rollout in copies:
            if rollout.evidence_signature not in seen:
                seen.add(rollout.evidence_signature)
                unique.append(rollout)
        return posterior_from_lrs(prior, [r.likelihood_ratio for r in unique])

    if method == "provenance_preserving":
        # Group by source, average within each source, and combine each source
        # exactly once.  This keeps independent source groups while suppressing
        # duplicate rollouts emitted by the same group.
        grouped: dict[str, list[Rollout]] = {}
        for rollout in copies:
            grouped.setdefault(rollout.source_id, []).append(rollout)
        source_lrs = [
            sum(r.likelihood_ratio for r in source_rollouts) / len(source_rollouts)
            for source_rollouts in grouped.values()
        ]
        return posterior_from_lrs(prior, source_lrs)

    raise ValueError(f"unknown method: {method}")


def run(api_key: str | None, model: str, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "model": model,
        "api_requested": bool(api_key),
        "api_success_by_seed": {},
        "true_state": TRUE_STATE,
        "root_hypothesis": HYPOTHESIS,
        "prior_and_rollouts": {},
    }
    for seed in SEEDS:
        prior, base_rollouts, api_ok = make_rollouts(seed, api_key, model)
        metadata["api_success_by_seed"][str(seed)] = api_ok
        metadata["prior_and_rollouts"][str(seed)] = {
            "prior_h_left": prior,
            "rollouts": [asdict(r) for r in base_rollouts],
        }
        baseline_actions: dict[str, str] = {}
        for method in METHODS:
            copies = base_rollouts
            p = aggregate(method, prior, copies)
            baseline_actions[method] = "LEFT" if p >= 0.5 else "RIGHT"

        for duplication in DUPLICATIONS:
            copies = [r for r in base_rollouts for _ in range(duplication)]
            for method in METHODS:
                p = aggregate(method, prior, copies)
                action = "LEFT" if p >= 0.5 else "RIGHT"
                reward = int(action == TRUE_STATE)
                rows.append(
                    {
                        "seed": seed,
                        "duplication": duplication,
                        "method": method,
                        "prior_h_left": prior,
                        "root_confidence_h_left": p,
                        "delta_vs_duplication_1": None,
                        "action": action,
                        "baseline_action_duplication_1": baseline_actions[method],
                        "action_flip_vs_duplication_1": int(action != baseline_actions[method]),
                        "reward": reward,
                        "true_state": TRUE_STATE,
                        "api_ok": api_ok,
                    }
                )

    results = pd.DataFrame(rows)
    baseline = results[results["duplication"] == 1][["seed", "method", "root_confidence_h_left"]].rename(
        columns={"root_confidence_h_left": "baseline_confidence"}
    )
    results = results.merge(baseline, on=["seed", "method"], how="left")
    results["delta_vs_duplication_1"] = results["root_confidence_h_left"] - results["baseline_confidence"]
    results.drop(columns=["baseline_confidence"], inplace=True)

    summary = (
        results.groupby(["duplication", "method"], as_index=False)
        .agg(
            mean_root_confidence_h_left=("root_confidence_h_left", "mean"),
            std_root_confidence_h_left=("root_confidence_h_left", "std"),
            mean_confidence_delta=("delta_vs_duplication_1", "mean"),
            action_flip_rate=("action_flip_vs_duplication_1", "mean"),
            mean_reward=("reward", "mean"),
            wrong_action_rate=("reward", lambda s: 1 - float(s.mean())),
        )
        .sort_values(["duplication", "method"])
    )
    results.to_csv(output_dir / "mvp_results.csv", index=False)
    summary.to_csv(output_dir / "mvp_summary.csv", index=False)
    (output_dir / "mvp_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def markdown_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in frame.iterrows():
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    report_lines = [
        "# Shared-assumption duplication MVP",
        "",
        f"- Model: `{model}`; seeds: `{SEEDS}`; duplications: `{DUPLICATIONS}`",
        f"- True state: **{TRUE_STATE}**; false root hypothesis: **{HYPOTHESIS}**",
        "- Each seed has two source groups with the same surface evidence signature. Copies within a source are exact repeats and add no new observation.",
        "- Action is LEFT when P(LEFT is safer) >= 0.5; reward is 1 only for RIGHT.",
        "",
        "## Mean results",
        "",
        markdown_table(summary),
        "",
        "`action_flip_rate` is measured against that method's duplication=1 action. `mean_reward` is averaged over the three seeds.",
        "",
        "## Interpretation",
        "",
        "If ordinary aggregation is behaving as the failure mode predicts, its false-root confidence should rise with duplication, flip from RIGHT to LEFT, and reduce reward. Provenance-preserving aggregation should remain close to its duplication=1 confidence and reward.",
        "",
        "## API status",
        "",
        json.dumps(metadata["api_success_by_seed"], ensure_ascii=False),
    ]
    (output_dir / "mvp_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return results, summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results_mvp")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key", default=None, help="DeepSeek key; prefer DEEPSEEK_API_KEY")
    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    results, summary, metadata = run(api_key, args.model, Path(args.output_dir))
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {len(results)} rows to {args.output_dir}/mvp_results.csv")
    print("API success by seed:", metadata["api_success_by_seed"])


if __name__ == "__main__":
    main()
