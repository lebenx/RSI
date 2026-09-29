# Shared-premise MVP artifacts

Run from the repository root.

```bash
# regenerate the 5-family benchmark (5,000 tasks)
python -m submission.benchmark --output results_submission/benchmark --instances-per-family 1000 --rollouts-per-action 4

# 11 controlled baselines, duplication 1/2/4/8/16
python -m submission.benchmark_eval --benchmark results_submission/benchmark --output results_submission/baselines --budgets 1,2,4,8,16

# scaling 1..64 (the frozen run is in results_submission/scaling)
python -m submission.benchmark_eval --benchmark results_submission/scaling_benchmark --output results_submission/scaling --budgets 1,2,4,8,16,32,64

# ablations and report
python -m submission.ablation --benchmark results_submission/scaling_benchmark --output results_submission/ablations
python -m submission.controlled_bootstrap
python -m submission.scienceworld_value_summary
python -m submission.scienceworld_interactive_aggregate
python -m submission.report
```

Key outputs:

- `report/REPORT.md`, `report/controlled_curves.png`, `report/main_table.csv`
- `report/controlled_bootstrap.csv` (20,000-draw task-cluster 95% intervals for confidence, reward, regret, and action flips)
- `report/main_table_ci.csv` (wide main table with point estimates and 95% intervals at duplication 1/8/64)
- `report/controlled_curves_ci.png` (scaling curves with task-cluster 95% error bars)
- `report/independent_scaling.csv`, `independent_paired.csv`, `independent_scaling.png` (separate independent-rollout value-information axis)
- `report/value_estimation_mse.csv`, `value_estimation_mse.png` (direct conditional-value MSE test over 624 held-out action/premise groups)
- `report/ablation_endpoint.csv`, `report/qualitative_examples.md`
- `scaling/baseline_raw.csv` and `scaling/baseline_summary.csv`
- `provenance/test_edge_metrics.csv`, `provenance/noisy_downstream.csv`
- `provenance/test_source_edge_metrics.csv`, `provenance/test_source_downstream.csv`, `provenance/metadata.json`
- `report/provenance_edge_breakdown.csv`, `report/provenance_noise_curve.csv`, `report/provenance_noise_curve.png` (family/difficulty recovery and downstream noise audit)
- `provenance_semantic_hard_final/summary.json` (multi-root opaque/paraphrased recovery stress; edge-only diagnostic)
- `provenance_semantic_hard_final/identifiability_null.csv` (2,000-draw test-label permutation null for opaque premise/source recovery)
- `local_qwen/summary.csv` (negative local-model diagnostic)
- `local_qwen_controls/summary.csv` (local ordering/paraphrase/prompt audit)
- `scienceworld_dev_m1/`, `scienceworld_dev_m4/` (closed-loop smoke traces)
- `scienceworld_findplant_dev2_m1/`, `scienceworld_findplant_dev2_m4/` (paired guided public-action run)
- `scienceworld_animal_dev1_m1/`, `scienceworld_animal_dev1_m4/` (additional paired guided task)
- `scienceworld_dev8_m1/`, `scienceworld_dev8_m4/` (eight-variation follow-up; mixed/negative result)
- `scienceworld_v7_dev8_m1/`, `scienceworld_v7_dev8_m4_retry/` (fair same-readout eight-variation run with public discovery/state-order policy)
- `scienceworld_v8_extra_m1/`, `scienceworld_v8_extra_m4/` (four additional held-out variations; negative candidate-generation stress)
- `scienceworld_v4_living_m1/`, `scienceworld_v4_living_m4/` (six-variation difficult living-thing stress run)
- `scienceworld_v4_core_m1/` (boil/power-component public-discovery stress run)
- `scienceworld_v7_animal161_long_m1/` (20-step long-horizon failure case)
- `report/scienceworld_paired_summary.csv`, `report/scienceworld_guided.png` (interactive metrics and plot)
- `report/scienceworld_dev8_summary.csv`, `report/scienceworld_v3_summary.csv` (broader and fresh stress summaries)
- `report/scienceworld_snapshot_summary.csv` (fixed-bank causal snapshot audit)
- `scienceworld_snapshot_audit_v5/` (state-replayed corrected snapshot audit; v4 raw scores retained for provenance)
- `report/scienceworld_v4_fair_summary.csv`, `report/scienceworld_v7_dev8_summary.csv` (fair same-readout interactive summaries)
- `report/scienceworld_v7_method_bootstrap.csv` (paired provenance-minus-flat full-episode contrasts)
- `report/evidence_audit.json` (requirement-by-requirement reproducibility audit)
- `../paper/evidence_matrix.md` (reviewer-facing claim/evidence/limitation matrix)
- `../paper/submission_checklist.md` (numbered requirement-to-evidence mapping with allowed wording)
- `../paper/claim_ledger.md` (machine-generated supported/limited/forbidden claim wording for submission review)
- `report/scienceworld_v7_bootstrap.csv`, `report/scienceworld_snapshot_bootstrap.csv` (deterministic paired/cluster bootstrap intervals)
- `report/scienceworld_interactive_counterfactual.csv` (four-state real-environment duplication intervention with common continuation)
- `interactive_counterfactual_replay/summary.json` (archived replay traces and episode-level scores)
- `report/scienceworld_value_summary.csv`, `report/scienceworld_value_paired.csv`, `report/scienceworld_value_curve.png` (four-variation grouped-value exploratory full episodes)
- `report/scienceworld_value_animal_summary.csv`, `report/scienceworld_value_animal_paired.csv`, `report/scienceworld_value_animal_curve.png` (one find-animal null control)
- `report/scienceworld_v4_living_summary.csv` (difficult living-thing stress summary)
- `report/scienceworld_v8_extra_summary.csv`, `report/scienceworld_v8_extra_by_task.csv` (additional held-out stress summary)
- `report/scienceworld_v9_discovery_summary.csv`, `report/scienceworld_v9_discovery_rows.csv` (public-navigation repair; m4 transport failures retained)
- `report/scienceworld_v7_fair_curves.png` (fair ScienceWorld reward/success/action-consistency figure)
- `report/scienceworld_local_qwen_compact_summary.csv` (one valid compact local-Qwen flat/provenance pair; schema/feasibility diagnostic only)
- `report/scienceworld_local_qwen_compact_grid_summary.csv` (three-variation, six-row local-Qwen compact grid; flat success 0.333 vs provenance 0.000; diagnostic only)
- `report/scienceworld_recharged_summary.csv`, `scienceworld_recharged_paired.csv`, `scienceworld_recharged_action_consistency.csv` (fresh DeepSeek six-variation `find-plant` comparison; 60 valid rows, negative planning boundary)
- `report/scienceworld_recharged_curves.png` (success/reward plot for the refreshed `find-plant` comparison)
- `report/scienceworld_recharged_action_bootstrap.csv` (variation-level 20,000-draw action-flip intervals)
- `report/scienceworld_recharged_metadata.json` (protocol, model, variation, and API-status manifest)
- `report/scienceworld_recharged_counterfactual/summary.csv` (six-state common-continuation replay; grouped provenance 6/6 success vs flat m=4 5/6)
- `report/scienceworld_recharged_animal/scienceworld_recharged_summary.csv` (second-task-family `find-animal` control; 36 valid rows, one transient transport retry archived)
- `report/scienceworld_recharged_animal/scienceworld_recharged_curves.png` (second-task-family success/reward plot)
- `report/scienceworld_expand_summary.csv`, `scienceworld_expand_method_bootstrap.csv`, `scienceworld_expand_action_bootstrap.csv` (fixed-grid 12-variation expansion; errors retained)
- `report/scienceworld_expand_counterfactual/summary.csv` (offline common-continuation replay for the same fixed-grid initial states)
- `report/scienceworld_expand_value_summary.csv`, `scienceworld_expand_value_paired.csv` (12-state algorithmic grouped-value null check)
- `report/scienceworld_expand_success_summary.csv`, `scienceworld_expand_success_paired.csv` (success-aware provenance readout ablation; matched valid rows)
- `report/scienceworld_interactive_aggregate/` (read-only 96-row ledger joining the refreshed LLM-readout and fresh-key success-aware protocols, with confidence/action/reward and paired bootstrap tables)
- `report/scienceworld_recharged_llm_extension/` (24-row fair fresh-key flat-vs-LLM-provenance plant extension; candidate/bank cache reused, readout rows newly generated)
- `report/scienceworld_recharged_animal_llm_extension/` (matched 24-row fair fresh-key flat-vs-LLM-provenance animal extension; negative external-validity control)
- `report/interactive_main_table/` (machine-generated 32-row interactive summary and 56-row paired-contrast table across refreshed, success-aware, and fair LLM protocols)
- `report/local_qwen_expanded_pilot/` (20 valid CUDA Qwen plant rows across five complete paired variations; raw schema failures retained separately)
- `report/local_qwen_value_replay/` (20-row byte-identical cached plant replay for algorithmic provenance-value, zero new model calls)
- `report/local_qwen_animal_expanded_pilot/` (17 valid CUDA Qwen animal rows, three complete pairs, and seven retained schema failures)
- `report/local_qwen_interactive_main_table/` (32-row, eight-pair task-family-stratified Qwen table with success, reward, unnecessary actions, calibration, steps, and tokens)
- `report/cross_model_interactive_table/` (104 non-pooled contrast rows, including 32 ALFWorld public-observation rows alongside DeepSeek/Qwen; model/task/objective heterogeneity is explicit)
- `scienceworld_expand2_m1/` and `scienceworld_expand2_success_m4/` (contiguous 156--161 extension; plant cache completed, animal rows with HTTP 402 transport failures retained and excluded from claims)
- `report/api_status.json` (latest DeepSeek connectivity diagnostic; refreshed key HTTP 200, earlier HTTP 402 rows retained and excluded, no secret stored)
- `deepseek_controls/summary.csv` (ordering, position, paraphrase, and prompt audit)
- `scienceworld_local_qwen_smoke_v3_m1/` (local-Qwen interactive schema negative diagnostic; excluded from planner metrics)
- `../paper/draft.md` (full experimental manuscript draft)

The controlled flat curve is a stipulated pseudo-likelihood mechanism used to
isolate the information/provenance error. The DeepSeek and Qwen runs are
model-specific diagnostics. The fair v7 ScienceWorld protocol gives flat and
provenance the same LLM readout, with provenance receiving a sample-grouped
bank. Across eight official dev variations, flat flips 27/64 paired actions
between duplication 1 and 4, while provenance flips zero; provenance reward is
77.375 at both budgets. The stochastic interactive sample is still too small to
justify a universal agent-improvement claim.

The controlled `semantic_dedup` baseline uses only planner-visible fields
(`action`, hypothesis value, and rounded predicted value) in its signature; it
does not use hidden premise/source IDs.

# Clean fixed-bank API intervention

`scienceworld_frozen_readout_clean_20260929/` contains the pre-action fixed-bank intervention: 12 states, 180 valid DeepSeek readouts at duplication 1/2/4/8 plus an identical-input repeat, 48 public-state-matched candidate replays, and immediate-reward summaries. `report/scienceworld_input_integrity/` records serialized-input checks and keeps legacy mutable trace views separate from causal evidence.

# Three-seed DeepSeek MVP

`deepseek_mvp_20260929/` contains the fixed-bank 3-seed, duplication 1/2/4/8 API diagnostic with flat, trajectory-average, deduplication, source grouping, provenance-value, paired/bootstrap tables, an identical-input control, and `curve.png`.

# ALFWorld TextWorld smoke

`report/alfworld_interactive/` is the validated ALFWorld public-observation comparison. It merges 90 DeepSeek base rows over six task types and 15 selected episode instances, 12 CUDA Qwen rows over task types 1, 2, and 6, and a 12-row cached DeepSeek success-objective provenance ablation (114 rows total). Outputs include success, reward, steps, unnecessary actions, charged tokens, root-confidence first/last/delta diagnostics, belief drift, calibration Brier, paired objective contrasts, episode-cluster bootstrap intervals, duplication-1/4 action consistency, and `curve.png`. The Qwen metadata records `cuda:0`, FP16, and the RTX 4090. The offset-1 DeepSeek extension is error-free and keeps provenance action flips at zero; the combined ALFWorld table remains exploratory. The success-objective ablation succeeds on 0.167 of rows. Qwen has zero success in the selected ten-step horizon, with 6/28 versus 0/28 flips. These rows are exploratory and models are kept separate. Earlier v1--v3 runs omitted persistent task text and v5 had incomplete rollout banks; they are retained only as excluded audit artifacts.
