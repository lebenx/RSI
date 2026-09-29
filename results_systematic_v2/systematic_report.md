# Systematic shared-assumption experiment

Seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]; priors=[0.2, 0.5, 0.8]; strengths={'weak': 1.2, 'medium': 1.5, 'strong': 2.0}; duplications=[1, 2, 4, 8, 16].
The full factorial design includes both hypothesis directions, both true states, pure duplication and independent rollout conditions, seven aggregators, randomized order, and equal-length controlled texts.

## Claim-level results (seed-level 95% t intervals)

| root_correct | condition | duplication | method | root_confidence_mean | root_confidence_ci95_low | root_confidence_ci95_high | action_flip_vs_duplication_1_mean | reward_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | independent_rollout | 1 | ordinary_flat_aggregation | 0.658 | 0.651 | 0.664 | 0.000 | 0.278 |
| False | independent_rollout | 1 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.278 |
| False | independent_rollout | 2 | ordinary_flat_aggregation | 0.775 | 0.768 | 0.783 | 0.167 | 0.111 |
| False | independent_rollout | 2 | provenance_preserving | 0.775 | 0.768 | 0.783 | 0.167 | 0.111 |
| False | independent_rollout | 4 | ordinary_flat_aggregation | 0.898 | 0.891 | 0.905 | 0.250 | 0.028 |
| False | independent_rollout | 4 | provenance_preserving | 0.898 | 0.891 | 0.905 | 0.250 | 0.028 |
| False | independent_rollout | 8 | ordinary_flat_aggregation | 0.972 | 0.967 | 0.977 | 0.278 | 0.000 |
| False | independent_rollout | 8 | provenance_preserving | 0.972 | 0.967 | 0.977 | 0.278 | 0.000 |
| False | independent_rollout | 16 | ordinary_flat_aggregation | 0.998 | 0.998 | 0.999 | 0.278 | 0.000 |
| False | independent_rollout | 16 | provenance_preserving | 0.998 | 0.998 | 0.999 | 0.278 | 0.000 |
| False | pure_duplication | 1 | ordinary_flat_aggregation | 0.658 | 0.651 | 0.664 | 0.000 | 0.278 |
| False | pure_duplication | 1 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.278 |
| False | pure_duplication | 2 | ordinary_flat_aggregation | 0.779 | 0.768 | 0.790 | 0.148 | 0.130 |
| False | pure_duplication | 2 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.278 |
| False | pure_duplication | 4 | ordinary_flat_aggregation | 0.901 | 0.886 | 0.916 | 0.250 | 0.028 |
| False | pure_duplication | 4 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.278 |
| False | pure_duplication | 8 | ordinary_flat_aggregation | 0.973 | 0.960 | 0.985 | 0.278 | 0.000 |
| False | pure_duplication | 8 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.278 |
| False | pure_duplication | 16 | ordinary_flat_aggregation | 0.997 | 0.992 | 1.001 | 0.278 | 0.000 |
| False | pure_duplication | 16 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.278 |
| True | independent_rollout | 1 | ordinary_flat_aggregation | 0.658 | 0.651 | 0.664 | 0.000 | 0.722 |
| True | independent_rollout | 1 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.722 |
| True | independent_rollout | 2 | ordinary_flat_aggregation | 0.775 | 0.768 | 0.783 | 0.167 | 0.889 |
| True | independent_rollout | 2 | provenance_preserving | 0.775 | 0.768 | 0.783 | 0.167 | 0.889 |
| True | independent_rollout | 4 | ordinary_flat_aggregation | 0.898 | 0.891 | 0.905 | 0.250 | 0.972 |
| True | independent_rollout | 4 | provenance_preserving | 0.898 | 0.891 | 0.905 | 0.250 | 0.972 |
| True | independent_rollout | 8 | ordinary_flat_aggregation | 0.972 | 0.967 | 0.977 | 0.278 | 1.000 |
| True | independent_rollout | 8 | provenance_preserving | 0.972 | 0.967 | 0.977 | 0.278 | 1.000 |
| True | independent_rollout | 16 | ordinary_flat_aggregation | 0.998 | 0.998 | 0.999 | 0.278 | 1.000 |
| True | independent_rollout | 16 | provenance_preserving | 0.998 | 0.998 | 0.999 | 0.278 | 1.000 |
| True | pure_duplication | 1 | ordinary_flat_aggregation | 0.658 | 0.651 | 0.664 | 0.000 | 0.722 |
| True | pure_duplication | 1 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.722 |
| True | pure_duplication | 2 | ordinary_flat_aggregation | 0.779 | 0.768 | 0.790 | 0.148 | 0.870 |
| True | pure_duplication | 2 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.722 |
| True | pure_duplication | 4 | ordinary_flat_aggregation | 0.901 | 0.886 | 0.916 | 0.250 | 0.972 |
| True | pure_duplication | 4 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.722 |
| True | pure_duplication | 8 | ordinary_flat_aggregation | 0.973 | 0.960 | 0.985 | 0.278 | 1.000 |
| True | pure_duplication | 8 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.722 |
| True | pure_duplication | 16 | ordinary_flat_aggregation | 0.997 | 0.992 | 1.001 | 0.278 | 1.000 |
| True | pure_duplication | 16 | provenance_preserving | 0.658 | 0.651 | 0.664 | 0.000 | 0.722 |

`action_flip_vs_duplication_1_mean` is measured against each method's duplication=1 action. `reward_mean` is the probability of selecting the true state.

## Method definitions

- ordinary_flat_aggregation: multiply every rollout likelihood ratio as if independent.
- single_trajectory_score_average: average per-rollout posteriors from the same prior.
- exact_deduplication: retain one exact signature.
- semantic_group_deduplication: retain one representative per semantic group.
- source_group_average: average per-rollout beliefs within each source and then across sources.
- explicit_belief_mixing: 0.5 prior + 0.5 mean rollout belief.
- provenance_preserving: average likelihood evidence within each source, then combine source groups once.

## API and controls

{"0": true, "1": true, "2": true, "3": true, "4": true, "5": true, "6": true, "7": true, "8": true, "9": true, "10": true, "11": true}

Every controlled rollout text has length 96; order_randomized=True in all rows.
