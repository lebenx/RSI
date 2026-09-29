# Shared-assumption duplication MVP

- Model: `deepseek-chat`; seeds: `[0, 1, 2]`; duplications: `[1, 2, 4, 8]`
- True state: **RIGHT**; false root hypothesis: **LEFT is safer**
- Each seed has two source groups with the same surface evidence signature. Copies within a source are exact repeats and add no new observation.
- Action is LEFT when P(LEFT is safer) >= 0.5; reward is 1 only for RIGHT.

## Mean results

| duplication | method | mean_root_confidence_h_left | std_root_confidence_h_left | mean_confidence_delta | action_flip_rate | mean_reward | wrong_action_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | dedup_by_evidence | 0.295 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 |
| 1 | ordinary_aggregation | 0.429 | 0.007 | 0.000 | 0.000 | 1.000 | 0.000 |
| 1 | provenance_preserving | 0.429 | 0.007 | 0.000 | 0.000 | 1.000 | 0.000 |
| 1 | single_trajectory_average | 0.335 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 |
| 2 | dedup_by_evidence | 0.295 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 |
| 2 | ordinary_aggregation | 0.629 | 0.007 | 0.200 | 1.000 | 0.000 | 1.000 |
| 2 | provenance_preserving | 0.429 | 0.007 | 0.000 | 0.000 | 1.000 | 0.000 |
| 2 | single_trajectory_average | 0.335 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4 | dedup_by_evidence | 0.295 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4 | ordinary_aggregation | 0.896 | 0.004 | 0.467 | 1.000 | 0.000 | 1.000 |
| 4 | provenance_preserving | 0.429 | 0.007 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4 | single_trajectory_average | 0.335 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 |
| 8 | dedup_by_evidence | 0.295 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 |
| 8 | ordinary_aggregation | 0.996 | 0.000 | 0.566 | 1.000 | 0.000 | 1.000 |
| 8 | provenance_preserving | 0.429 | 0.007 | 0.000 | 0.000 | 1.000 | 0.000 |
| 8 | single_trajectory_average | 0.335 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 |

`action_flip_rate` is measured against that method's duplication=1 action. `mean_reward` is averaged over the three seeds.

## Interpretation

If ordinary aggregation is behaving as the failure mode predicts, its false-root confidence should rise with duplication, flip from RIGHT to LEFT, and reduce reward. Provenance-preserving aggregation should remain close to its duplication=1 confidence and reward.

## API status

{"0": true, "1": true, "2": true}
