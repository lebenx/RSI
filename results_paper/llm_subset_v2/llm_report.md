# DeepSeek LLM subset

This is a real `deepseek-chat` run with 12 scenarios, 3 information conditions, m∈{1,2,4,8}, 7 derived methods, 300 archived API calls and no synthetic fallback. Every API response parsed successfully; controlled rollout strings had equal length and presentation order was randomized. The prompts never contained hidden physical state or reward labels.

The item scorer and evidence-only call are used to construct deduplication, source averaging, explicit mixing and provenance rows. The flat baseline has a separate whole-list API call. These derived rows are still an API subset, not a population estimate.

## Wrong root / pure duplication

| multiplicity | method | root_confidence_mean | root_confidence_ci95_low | root_confidence_ci95_high | action_flip_mean | expected_reward_mean | decision_regret_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | belief_mixing | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 1 | exact_dedup | 0.467 | 0.381 | 0.552 | 0.000 | 0.148 | 0.962 |
| 1 | flat | 0.467 | 0.197 | 0.738 | 0.000 | 0.148 | 0.962 |
| 1 | provenance | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 1 | semantic_dedup | 0.467 | 0.381 | 0.552 | 0.000 | 0.148 | 0.962 |
| 1 | source_average | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 1 | trajectory_average | 0.467 | 0.381 | 0.552 | 0.000 | 0.148 | 0.962 |
| 8 | belief_mixing | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 8 | exact_dedup | 0.433 | 0.325 | 0.542 | 0.000 | 0.148 | 0.962 |
| 8 | flat | 0.367 | 0.212 | 0.522 | 0.167 | 0.579 | 0.532 |
| 8 | provenance | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 8 | semantic_dedup | 0.433 | 0.325 | 0.542 | 0.000 | 0.148 | 0.962 |
| 8 | source_average | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 8 | trajectory_average | 0.433 | 0.325 | 0.542 | 0.000 | 0.148 | 0.962 |

## Correct root / independent rollout

| multiplicity | method | root_confidence_mean | root_confidence_ci95_low | root_confidence_ci95_high | action_flip_mean | expected_reward_mean | decision_regret_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | belief_mixing | 0.595 | 0.310 | 0.880 | 0.000 | 0.727 | 0.470 |
| 1 | exact_dedup | 0.634 | 0.416 | 0.853 | 0.000 | 0.727 | 0.470 |
| 1 | flat | 0.568 | 0.259 | 0.877 | 0.000 | 0.727 | 0.470 |
| 1 | provenance | 0.595 | 0.310 | 0.880 | 0.000 | 0.727 | 0.470 |
| 1 | semantic_dedup | 0.634 | 0.416 | 0.853 | 0.000 | 0.727 | 0.470 |
| 1 | source_average | 0.595 | 0.310 | 0.880 | 0.000 | 0.727 | 0.470 |
| 1 | trajectory_average | 0.634 | 0.416 | 0.853 | 0.000 | 0.727 | 0.470 |
| 8 | belief_mixing | 0.595 | 0.310 | 0.880 | 0.167 | 1.197 | 0.000 |
| 8 | exact_dedup | 0.506 | 0.281 | 0.731 | 0.000 | 0.727 | 0.470 |
| 8 | flat | 0.568 | 0.259 | 0.877 | 0.333 | 1.015 | 0.182 |
| 8 | provenance | 0.595 | 0.310 | 0.880 | 0.167 | 1.197 | 0.000 |
| 8 | semantic_dedup | 0.506 | 0.281 | 0.731 | 0.000 | 0.727 | 0.470 |
| 8 | source_average | 0.595 | 0.310 | 0.880 | 0.167 | 1.197 | 0.000 |
| 8 | trajectory_average | 0.506 | 0.281 | 0.731 | 0.000 | 0.727 | 0.470 |

## Wrong root / paraphrase duplication

| multiplicity | method | root_confidence_mean | root_confidence_ci95_low | root_confidence_ci95_high | action_flip_mean | expected_reward_mean | decision_regret_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | belief_mixing | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 1 | exact_dedup | 0.500 | 0.500 | 0.500 | 0.000 | 0.148 | 0.962 |
| 1 | flat | 0.467 | 0.197 | 0.737 | 0.000 | 0.579 | 0.532 |
| 1 | provenance | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 1 | semantic_dedup | 0.500 | 0.500 | 0.500 | 0.000 | 0.148 | 0.962 |
| 1 | source_average | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 1 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.148 | 0.962 |
| 8 | belief_mixing | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 8 | exact_dedup | 0.433 | 0.325 | 0.542 | 0.000 | 0.148 | 0.962 |
| 8 | flat | 0.400 | 0.240 | 0.560 | 0.000 | 0.579 | 0.532 |
| 8 | provenance | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 8 | semantic_dedup | 0.433 | 0.325 | 0.542 | 0.000 | 0.148 | 0.962 |
| 8 | source_average | 0.552 | 0.339 | 0.764 | 0.000 | 0.148 | 0.962 |
| 8 | trajectory_average | 0.433 | 0.325 | 0.542 | 0.000 | 0.148 | 0.962 |

## Interpretation

The analytical grid shows the specified flat-counting failure strongly. The DeepSeek subset does not show a stable monotone confidence increase for the flat whole-list call: its direction varies by scenario and its intervals are wide. This is a valid negative diagnostic, not evidence that every LLM implements the analytical flat estimator.
The stable invariant in the API subset is structural: evidence-only q and provenance/belief-mixing rows remain unchanged across pure duplicates, while conditional-return rows can change under independent samples. A larger API study or a frozen local/open model is required before claiming a model-level effect.

Archived prompts and responses: `api_archive.jsonl`; row-level data: `llm_raw.csv`; summary with 95% intervals: `llm_summary.csv`.
