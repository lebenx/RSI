# Qwen conditional-bank diagnostic

Generated with `python -m submission.qwen_bank_diagnostic` from archived traces.

Main table: 751/768 decision rows have constant conditional scores; 396/403 unique snapshots do so.

56/56 provenance episode rows follow candidate ID 0 at every recorded step. This is a trajectory-level behavioral identity, not a newly executed no-imagination baseline.

| Cohort | Method | Steps | Constant scores | Template values | Belief-sensitive winner | Rank 0 match |
|---|---|---:|---:|---:|---:|---:|
| fair_168_169 | flat_value | 54 | 1.000 | 1.000 | 0.000 | 1.000 |
| fair_168_169 | provenance_value | 54 | 1.000 | 1.000 | 0.000 | 1.000 |
| heldout_animal | flat_value | 48 | 0.958 | 0.958 | 0.000 | 1.000 |
| heldout_animal | provenance_value | 48 | 0.958 | 0.958 | 0.000 | 1.000 |
| heldout_plant | flat_value | 48 | 0.875 | 0.875 | 0.042 | 0.958 |
| heldout_plant | provenance_value | 48 | 0.875 | 0.875 | 0.042 | 0.958 |
| main_27_pairs | flat | 406 | 0.978 | 0.978 | 0.000 | 0.409 |
| main_27_pairs | provenance_value | 362 | 0.978 | 0.978 | 0.000 | 1.000 |

The compact local prompt specifies a schema and example values but omits the API prompt's conditional value-estimation instructions. The observed matching values suggest an example-copying failure; the prompt has not yet been causally isolated. Constant banks force the deterministic aggregator to choose the lowest action ID independently of premise probability.

Historical positive reward differences remain descriptive, but are not evidence that provenance caused the improvement. The next experiment must compare identical candidate/bank inputs under an explicitly semantic local rollout prompt, include rank-zero and direct-policy baselines, and measure value discrimination and calibration before interpreting closed-loop rewards. Keep constant banks and failures; do not select only instances that show a positive effect.

Belief-sensitive winner tests q=0 and q=1 analytically: with linear mixtures and fixed ID tie-breaking, identical endpoint winners imply the same winner for every q in [0,1]. All per-step and per-episode observations are retained in the CSVs.
