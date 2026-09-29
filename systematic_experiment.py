#!/usr/bin/env python3
"""Systematic controlled experiment for shared-assumption calibration.

This extends the MVP without introducing a learned local model.  DeepSeek is
called once per seed for a traceable rollout note; all likelihoods and rewards
come from a binary environment whose factors are explicit and reproducible.

Two information conditions are kept separate:
  * pure_duplication: exact copies of existing rollouts, same source and text;
  * independent_rollout: new source/context samples with fresh evidence noise.

All controlled texts are padded to equal length and presentation order is
randomized independently for every seed/scenario/duplication.  The resulting
raw table has one row per seed/scenario/method/duplication, while summaries use
Student-t 95% confidence intervals over seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy import stats


METHODS = [
    "ordinary_flat_aggregation",
    "single_trajectory_score_average",
    "exact_deduplication",
    "semantic_group_deduplication",
    "source_group_average",
    "explicit_belief_mixing",
    "provenance_preserving",
]
CONDITIONS = ["pure_duplication", "independent_rollout"]
DUPLICATIONS = [1, 2, 4, 8, 16]
SEEDS = list(range(12))
PRIORS = [0.20, 0.50, 0.80]
STRENGTHS = {"weak": 1.20, "medium": 1.50, "strong": 2.00}
STRENGTH_ORDER = ["weak", "medium", "strong"]
TEXT_LENGTH = 96
TRUE_STATES = ["LEFT", "RIGHT"]
HYPOTHESIS_DIRECTIONS = ["LEFT", "RIGHT"]


@dataclass(frozen=True)
class Rollout:
    seed: int
    source_id: str
    exact_signature: str
    semantic_group: str
    likelihood_ratio: float
    controlled_text: str
    model_note: str
    sample_index: int


def sigmoid(x: float) -> float:
    # Stable enough for the bounded likelihood strengths used here.
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def posterior_from_lrs(prior: float, lrs: list[float]) -> float:
    return sigmoid(logit(prior) + sum(math.log(max(lr, 1e-9)) for lr in lrs))


def fixed_length_text(hypothesis: str, sample_label: str) -> str:
    base = (
        f"Shared-model rollout supports {hypothesis}; the evidence is a controlled "
        f"sample and contains no hidden reward or extra observation. {sample_label}."
    )
    if len(base) >= TEXT_LENGTH:
        return base[:TEXT_LENGTH]
    return base + " " * (TEXT_LENGTH - len(base))


def call_deepseek(seed: int, api_key: str | None, model: str) -> tuple[str, bool]:
    fallback = (
        "The rollout repeats the shared world-model assumption and adds no new "
        "real observation; treat it as correlated evidence."
    )
    if not api_key:
        return fallback, False
    prompt = f"""
Return JSON only with a single field note. Write a concise (under 35 words)
research note for a controlled planning experiment. It must say that a rollout
supports a root hypothesis but repeats a shared world-model assumption and adds
no new real observation. Do not give numbers or choose an action. Seed={seed}.
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
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        note = str(parsed.get("note", "")).strip()
        return (note or fallback), bool(note)
    except Exception as exc:
        return f"{fallback} (API fallback: {type(exc).__name__})", False


def _base_log_lr(strength: float, rng: random.Random) -> float:
    # Fresh independent samples vary mildly around the requested strength.
    return math.log(strength) + rng.gauss(0.0, 0.07)


def generate_rollouts(
    seed: int,
    hypothesis: str,
    strength: float,
    condition: str,
    duplication: int,
    model_note: str,
) -> list[Rollout]:
    """Generate a balanced, order-randomized rollout multiset."""
    rng = random.Random(10_000_000 + seed * 10_000 + int(strength * 100) + duplication)
    base_sources = ["source_A", "source_B"]
    rollouts: list[Rollout] = []

    if condition == "pure_duplication":
        # Two base source groups. Every copy is exact: same source, signature,
        # semantic group, likelihood ratio, and equal-length controlled text.
        base_lrs = [math.exp(_base_log_lr(strength, rng)), math.exp(_base_log_lr(strength, rng))]
        for source_index, source_id in enumerate(base_sources):
            lr = base_lrs[source_index]
            text = fixed_length_text(hypothesis, f"shared sample source={source_id}")
            for copy_index in range(duplication):
                rollouts.append(
                    Rollout(
                        seed=seed,
                        source_id=source_id,
                        exact_signature="exact_shared_report_v1",
                        semantic_group="shared_root_assumption",
                        likelihood_ratio=lr,
                        controlled_text=text,
                        model_note=model_note,
                        sample_index=copy_index,
                    )
                )
    elif condition == "independent_rollout":
        # Every sample is a new source/context. Exact signatures are unique;
        # semantic groups intentionally have two context families to test the
        # cost of semantic over-collapse on genuinely new evidence.
        sample_index = 0
        for source_index in range(2 * duplication):
            source_id = f"independent_source_{source_index:02d}"
            lr = math.exp(_base_log_lr(strength, rng))
            semantic_group = f"independent_context_{source_index % 2}"
            text = fixed_length_text(hypothesis, f"independent context={source_index % 2} sample={source_index}")
            rollouts.append(
                Rollout(
                    seed=seed,
                    source_id=source_id,
                    exact_signature=f"independent_exact_{seed}_{source_index}",
                    semantic_group=semantic_group,
                    likelihood_ratio=lr,
                    controlled_text=text,
                    model_note=model_note,
                    sample_index=sample_index,
                )
            )
            sample_index += 1
    else:
        raise ValueError(condition)

    rng.shuffle(rollouts)  # explicit order randomization control
    return rollouts


def aggregate(method: str, prior: float, rollouts: list[Rollout]) -> float:
    if not rollouts:
        return prior
    if method == "ordinary_flat_aggregation":
        return posterior_from_lrs(prior, [r.likelihood_ratio for r in rollouts])

    per_rollout = [posterior_from_lrs(prior, [r.likelihood_ratio]) for r in rollouts]
    if method == "single_trajectory_score_average":
        return float(np.mean(per_rollout))

    if method == "exact_deduplication":
        selected: dict[str, Rollout] = {}
        for rollout in rollouts:
            selected.setdefault(rollout.exact_signature, rollout)
        return posterior_from_lrs(prior, [r.likelihood_ratio for r in selected.values()])

    if method == "semantic_group_deduplication":
        selected = {}
        for rollout in rollouts:
            selected.setdefault(rollout.semantic_group, rollout)
        return posterior_from_lrs(prior, [r.likelihood_ratio for r in selected.values()])

    grouped: dict[str, list[Rollout]] = {}
    for rollout in rollouts:
        grouped.setdefault(rollout.source_id, []).append(rollout)

    if method == "source_group_average":
        group_scores = []
        for group in grouped.values():
            group_scores.append(float(np.mean([posterior_from_lrs(prior, [r.likelihood_ratio]) for r in group])))
        return float(np.mean(group_scores))

    if method == "explicit_belief_mixing":
        # A standard convex mixture: retain half of the prior and half of the
        # equally weighted rollout beliefs. It is explicit about uncertainty but
        # does not pretend repeated beliefs are independent observations.
        return 0.5 * prior + 0.5 * float(np.mean(per_rollout))

    if method == "provenance_preserving":
        # Average evidence within each source, then combine independent source
        # groups. Duplicate copies therefore have no effect; new sources do.
        source_lrs = [float(np.mean([r.likelihood_ratio for r in group])) for group in grouped.values()]
        return posterior_from_lrs(prior, source_lrs)

    raise ValueError(method)


def _ci95(values: pd.Series) -> tuple[float, float, float, float]:
    arr = values.dropna().astype(float).to_numpy()
    n = len(arr)
    mean = float(np.mean(arr)) if n else float("nan")
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        margin = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
    else:
        margin = float("nan")
    return mean, std, mean - margin, mean + margin


def summarize(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in rows.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        for metric in [
            "root_confidence",
            "confidence_delta_vs_duplication_1",
            "action_flip_vs_duplication_1",
            "reward",
            "wrong_action",
        ]:
            mean, std, ci_low, ci_high = _ci95(group[metric])
            if metric in {"root_confidence", "action_flip_vs_duplication_1", "reward", "wrong_action"}:
                ci_low = max(0.0, ci_low)
                ci_high = min(1.0, ci_high)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_ci95_low"] = ci_low
            record[f"{metric}_ci95_high"] = ci_high
        record["n"] = int(group.shape[0])
        records.append(record)
    return pd.DataFrame(records)


def run(api_key: str | None, model: str, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "model": model,
        "api_requested": bool(api_key),
        "api_success_by_seed": {},
        "seeds": SEEDS,
        "priors": PRIORS,
        "strengths": STRENGTHS,
        "duplications": DUPLICATIONS,
        "conditions": CONDITIONS,
        "methods": METHODS,
        "controls": {"order_randomized": True, "equal_text_length": TEXT_LENGTH},
    }

    # Call the single external model once per seed; all scenario factors below
    # use the same note so API phrasing cannot leak condition labels.
    notes: dict[int, str] = {}
    for seed in SEEDS:
        note, ok = call_deepseek(seed, api_key, model)
        notes[seed] = note
        metadata["api_success_by_seed"][str(seed)] = ok

    for seed in SEEDS:
        for hypothesis in HYPOTHESIS_DIRECTIONS:
            for true_state in TRUE_STATES:
                root_correct = hypothesis == true_state
                for prior in PRIORS:
                    for strength_name in STRENGTH_ORDER:
                        strength = STRENGTHS[strength_name]
                        for condition in CONDITIONS:
                            # Generate all duplication levels as nested sample
                            # sets. For pure duplication this is simply exact
                            # copies; for independent rollout it is a prefix of
                            # new source/context samples.
                            max_generated = generate_rollouts(
                                seed, hypothesis, strength, condition, max(DUPLICATIONS), notes[seed]
                            )
                            # Use nested prefixes of one generated experiment.
                            # This makes the duplication curve reflect added
                            # copies/samples rather than a fresh resampling at
                            # every x-axis value.
                            by_source: dict[str, list[Rollout]] = {}
                            for rollout in max_generated:
                                by_source.setdefault(rollout.source_id, []).append(rollout)
                            generated: dict[int, list[Rollout]] = {}
                            for duplication in DUPLICATIONS:
                                if condition == "pure_duplication":
                                    selected = []
                                    for source_rollouts in by_source.values():
                                        selected.extend(
                                            sorted(source_rollouts, key=lambda r: r.sample_index)[:duplication]
                                        )
                                else:
                                    # Independent samples each have a unique
                                    # source, so the nested prefix is selected
                                    # globally (two new samples per duplication
                                    # level, matching the two base source slots).
                                    selected = sorted(max_generated, key=lambda r: r.sample_index)[: 2 * duplication]
                                order_rng = random.Random(
                                    70_000_000
                                    + seed * 100_000
                                    + duplication * 1_000
                                    + int(strength * 100)
                                    + (0 if condition == "pure_duplication" else 1)
                                )
                                order_rng.shuffle(selected)
                                generated[duplication] = selected
                            baseline_actions: dict[str, str] = {}
                            baseline_confidence: dict[str, float] = {}
                            for method in METHODS:
                                p1 = aggregate(method, prior, generated[1])
                                baseline_confidence[method] = p1
                                baseline_actions[method] = hypothesis if p1 >= 0.5 else ("RIGHT" if hypothesis == "LEFT" else "LEFT")
                            for duplication in DUPLICATIONS:
                                rollouts = generated[duplication]
                                # These should be exact by construction; retain
                                # checks in metadata rows for auditability.
                                for method in METHODS:
                                    confidence = aggregate(method, prior, rollouts)
                                    action = hypothesis if confidence >= 0.5 else ("RIGHT" if hypothesis == "LEFT" else "LEFT")
                                    rows.append(
                                        {
                                            "seed": seed,
                                            "hypothesis_direction": hypothesis,
                                            "true_state": true_state,
                                            "root_correct": root_correct,
                                            "prior_hypothesis_confidence": prior,
                                            "strength_name": strength_name,
                                            "likelihood_strength": strength,
                                            "condition": condition,
                                            "duplication": duplication,
                                            "method": method,
                                            "root_confidence": confidence,
                                            "confidence_delta_vs_duplication_1": confidence - baseline_confidence[method],
                                            "action": action,
                                            "baseline_action_duplication_1": baseline_actions[method],
                                            "action_flip_vs_duplication_1": int(action != baseline_actions[method]),
                                            "reward": int(action == true_state),
                                            "wrong_action": int(action != true_state),
                                            "rollout_count": len(rollouts),
                                            "order_randomized": True,
                                            "equal_text_length": all(len(r.controlled_text) == TEXT_LENGTH for r in rollouts),
                                            "api_ok": metadata["api_success_by_seed"][str(seed)],
                                        }
                                    )

    raw = pd.DataFrame(rows)
    full_summary = summarize(
        raw,
        [
            "hypothesis_direction",
            "true_state",
            "root_correct",
            "prior_hypothesis_confidence",
            "strength_name",
            "likelihood_strength",
            "condition",
            "duplication",
            "method",
        ],
    )

    # Seed-level summaries average the factorial settings within each seed first,
    # preventing pseudo-replication when reporting the main claim.
    seed_level = (
        raw.groupby(["seed", "root_correct", "condition", "duplication", "method"], as_index=False)[
            ["root_confidence", "confidence_delta_vs_duplication_1", "action_flip_vs_duplication_1", "reward", "wrong_action"]
        ]
        .mean()
    )
    claim_summary = summarize(seed_level, ["root_correct", "condition", "duplication", "method"])

    raw.to_csv(output_dir / "systematic_raw.csv", index=False)
    full_summary.to_csv(output_dir / "systematic_summary.csv", index=False)
    claim_summary.to_csv(output_dir / "systematic_claims.csv", index=False)
    (output_dir / "systematic_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compact report with the claim-level table and an independent-vs-duplicate
    # comparison for the provenance method.
    def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in frame.iterrows():
            vals = []
            for v in row:
                vals.append(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    selected = claim_summary[
        (claim_summary["method"].isin(["ordinary_flat_aggregation", "provenance_preserving"]))
        & (claim_summary["duplication"].isin([1, 2, 4, 8, 16]))
    ][
        [
            "root_correct",
            "condition",
            "duplication",
            "method",
            "root_confidence_mean",
            "root_confidence_ci95_low",
            "root_confidence_ci95_high",
            "action_flip_vs_duplication_1_mean",
            "reward_mean",
        ]
    ]
    report = [
        "# Systematic shared-assumption experiment",
        "",
        f"Seeds={SEEDS}; priors={PRIORS}; strengths={STRENGTHS}; duplications={DUPLICATIONS}.",
        "The full factorial design includes both hypothesis directions, both true states, pure duplication and independent rollout conditions, seven aggregators, randomized order, and equal-length controlled texts.",
        "",
        "## Claim-level results (seed-level 95% t intervals)",
        "",
        md_table(selected),
        "",
        "`action_flip_vs_duplication_1_mean` is measured against each method's duplication=1 action. `reward_mean` is the probability of selecting the true state.",
        "",
        "## Method definitions",
        "",
        "- ordinary_flat_aggregation: multiply every rollout likelihood ratio as if independent.",
        "- single_trajectory_score_average: average per-rollout posteriors from the same prior.",
        "- exact_deduplication: retain one exact signature.",
        "- semantic_group_deduplication: retain one representative per semantic group.",
        "- source_group_average: average per-rollout beliefs within each source and then across sources.",
        "- explicit_belief_mixing: 0.5 prior + 0.5 mean rollout belief.",
        "- provenance_preserving: average likelihood evidence within each source, then combine source groups once.",
        "",
        "## API and controls",
        "",
        json.dumps(metadata["api_success_by_seed"], ensure_ascii=False),
        "",
        f"Every controlled rollout text has length {TEXT_LENGTH}; order_randomized=True in all rows.",
    ]
    (output_dir / "systematic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return raw, full_summary, claim_summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results_systematic")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    raw, full_summary, claim_summary, metadata = run(api_key, args.model, Path(args.output_dir))
    print("rows", len(raw), "full_summary_rows", len(full_summary), "claim_rows", len(claim_summary))
    print("api_success_by_seed", metadata["api_success_by_seed"])
    print("\nSelected claim summary:")
    print(
        claim_summary[
            [
                "root_correct",
                "condition",
                "duplication",
                "method",
                "root_confidence_mean",
                "action_flip_vs_duplication_1_mean",
                "reward_mean",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
