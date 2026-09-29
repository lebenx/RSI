# MVP submission report

Generated from frozen artifacts; no numbers below are hand-entered.

## Controlled SharedPremiseBench

The scaling bank has 500 tasks total (78 in the held-out test split), five premise families, two counterfactual hypothesis values, and 64 independent condition draws per action. Hidden state is used only for post-hoc reward, optimal-action labels, and calibration audit; the planner-visible bank contains action/value text and IDs. `flat_rollout` is an explicit stipulated duplicate-count pseudo-likelihood mechanism.

At duplication=1: flat confidence 0.510, wrong-premise confidence 0.528, reward 0.613, flip rate 0.000; provenance-preserving confidence 0.510, reward 0.613, flip rate 0.000.
At duplication=2: flat confidence 0.592, wrong-premise confidence 0.521, reward 0.531, flip rate 0.192; provenance-preserving confidence 0.510, reward 0.613, flip rate 0.000.
At duplication=4: flat confidence 0.742, wrong-premise confidence 0.505, reward 0.488, flip rate 0.397; provenance-preserving confidence 0.510, reward 0.613, flip rate 0.000.
At duplication=8: flat confidence 0.926, wrong-premise confidence 0.483, reward 0.311, flip rate 0.551; provenance-preserving confidence 0.510, reward 0.613, flip rate 0.000.
At duplication=16: flat confidence 0.997, wrong-premise confidence 0.475, reward 0.272, flip rate 0.564; provenance-preserving confidence 0.510, reward 0.613, flip rate 0.000.
At duplication=32: flat confidence 1.000, wrong-premise confidence 0.474, reward 0.272, flip rate 0.564; provenance-preserving confidence 0.510, reward 0.613, flip rate 0.000.
At duplication=64: flat confidence 1.000, wrong-premise confidence 0.474, reward 0.272, flip rate 0.564; provenance-preserving confidence 0.510, reward 0.613, flip rate 0.000.

### Task-cluster uncertainty

The controlled estimates use a deterministic 20,000-draw bootstrap over the 78 held-out tasks. At duplication 8, flat root confidence is 0.926 (95% CI 0.912–0.939) and its action-flip rate is 0.551 (95% CI 0.436–0.667); flat reward is 0.311 (95% CI -0.049–0.664). Provenance root confidence is 0.510 (95% CI 0.462–0.555) with zero observed flips, and reward is 0.613 (95% CI 0.398–0.806). The bootstrap quantifies task heterogeneity; it does not convert the stipulated flat estimator into an empirical claim about arbitrary language models. The wide machine-generated table is `main_table_ci.csv`; the long-form intervals are in `controlled_bootstrap.csv`.

### Independent-rollout value axis

On the independent-rollout condition, independent trajectory aggregation has reward 0.825 at budget 1 and 0.778 at budget 64, versus 0.613 for the provenance-preserving duplicate baseline at budget 64. The task-level paired reward advantage remains positive across budgets, but the curve is noisy rather than monotone; this supports useful independent value information without claiming that more samples always improve the selected action. See `independent_scaling.csv`, `independent_paired.csv`, and `independent_scaling.png`.

### Conditional-value estimation error

Across 624 held-out action/premise groups, independent-sample conditional-value MSE falls from 0.052357 at n=1 to 0.000816 at n=64. Pure duplication stays at 0.051736 for every presentation budget. This directly tests the variance-reduction part of the theory; the target is the 64-draw conditional mean and no hidden state is used by the estimator. See `value_estimation_mse.csv` and `value_estimation_mse.png`.

A harmful flip example is documented in `qualitative_examples.md`: a false shared premise causes flat confidence to rise from 0.754 to 1.000 and changes a +1.041 action into a -1.940 action, while provenance stays invariant.

## Real-model presentation controls

The focused DeepSeek control audit archives 96 calls over four scenarios, duplication 1/4, front/reverse/interleaved ordering, paraphrased text, and neutral/cautious prompts. Two malformed JSON responses are retained as failures and never replaced. The mean confidence delta at duplication 4 ranges from about -0.151 to +0.034 across presentation cells, with action flips up to 0.50. The local Qwen2.5-Coder-3B control audit has 96 calls over three seeds, two scenarios per seed, front/reverse placement, paraphrases, and neutral/cautious prompts: neutral prompts remain at confidence 0.9, while the cautious prompt gives dup=4 confidence 0.55 on average with action flips up to 1.0. These are context/prompt sensitivity results, not universal monotone duplication laws. See `../deepseek_controls/summary.csv` and `../local_qwen_controls/summary.csv`.

## Learned provenance recovery

The extractor result is in `../provenance/metadata.json`. It reports premise-edge and source-edge precision/recall/F1, oracle/generated-ID/learned grouping, and noisy downstream reward. Both edge F1 values are 1.0 because the toy benchmark exposes the premise key and source pattern lexically; these are upper-bound sanity checks, not evidence of robust semantic provenance recovery. Noise rates and downstream reward are reported separately.

## ScienceWorld

The guided public-action-shortlist runner uses only task text, public history, and admissible commands; no gold path or hidden state enters the model. The fair v7 protocol gives flat and provenance the same candidate generation, conditional bank, and one LLM readout; provenance receives a sample-grouped bank and carries the evidence-only belief unchanged. Across eight official dev variations, no-imagination succeeds on 0.625 of episodes and flat/provenance on 0.5 at both budgets. Flat reward is 69.0 at duplication 1 and 77.375 at duplication 4 in this stochastic API run, but its paired actions flip 27/64 times (0.422); provenance reward is 77.375 at both budgets with zero flips. The difficult animal variations fail for all methods, so this is evidence for action consistency and duplication robustness, not a broad success-rate gain. A six-variation `find-living-thing` stress run has success 0.333 for no-imagination and 0.167 for both rollout methods at both budgets; these rows are retained as candidate-generation failure cases rather than filtered out. Cluster bootstrap over the eight episodes gives the flat step-weighted flip rate 0.422 (95% CI 0.125–0.688) and provenance 0.000 (0–0); the paired flat reward change is +8.375 (0–25.125), while provenance is exactly 0 in this frozen run. See `scienceworld_v7_bootstrap.csv` for all metrics.


### Paired full-episode method contrast

Within the same eight episodes, provenance minus flat reward is +8.375 (95% CI +0.000–+25.125) at duplication 1 and +0.000 (95% CI +0.000–+0.000) at duplication 4. The corresponding success difference is +0.000 (95% CI +0.000–+0.000) and +0.000 (95% CI +0.000–+0.000). This supports a reward-level planning advantage in the small paired sample, while the success contrast remains zero.

### Legacy snapshot diagnostic (input-integrity caveat)

The archived v3/v5 snapshot rows are retained for provenance, but their trace-derived visible histories were later found to contain the post-action mutable history view. Their numerical readout and replay diagnostics are therefore not used as clean causal evidence. The clean serialized-input intervention below is the authoritative fixed-bank result; see `scienceworld_input_integrity/summary.json`.


### Fixed-grid ScienceWorld expansion

A fixed-grid expansion covers find-plant and find-animal variations 150--155 (12 task/variation pairs, 72 method-budget rows; 66 valid and 6 candidate-interface errors). Errors remain visible in `scienceworld_expand_rows.csv` and are excluded only from means. On find-plant, flat success changes from 0.833 to 0.667 while provenance stays 0.833 to 0.833; on find-animal, flat changes from 0.600 to 0.800 while provenance stays 0.600 to 0.600. The expanded full-episode sample therefore supports duplication-robust action consistency but still does not establish a general success-rate gain. Across aligned steps, flat action flips are 0.410 (95% CI 0.167--0.637) and provenance flips 0.000. See `scienceworld_expand_summary.csv`, `scienceworld_expand_method_bootstrap.csv`, and `scienceworld_expand_action_bootstrap.csv`.


### Expanded state-replay stress

An offline common-continuation replay over the same 12 initial states gives grouped provenance success 0.750, flat duplication-1 success 0.750, and flat duplication-4 success 0.833; grouped provenance also has lower mean reward. This nonpositive stress case is retained because provenance normalization removes duplication sensitivity but cannot repair a weak candidate/value bank. See `scienceworld_expand_counterfactual/summary.csv`.


### Fixed-grid grouped-value null check

The algorithmic unique-sample conditional-value readout is duplication-invariant on all 12 fixed-grid states: success/reward are 0.667/71.0 at both duplication levels. Against flat, its paired reward contrast is +0.00 (95% CI -21.91–+21.91) at duplication 1 and -4.55 (95% CI -17.36–+5.27) at duplication 4. This is a negative boundary result: grouping prevents duplicate sensitivity but does not guarantee better planning when the candidate/value bank is weak. See `scienceworld_expand_value_summary.csv` and `scienceworld_expand_value_paired.csv`.


### Success-aware provenance ablation

Selecting the highest unique-sample conditional success probability is an explicit readout ablation. It is duplication-invariant on the fixed grid (success 0.750, reward 77.9 at both m=1 and m=4). On the 11 matched valid episodes, its paired reward contrast versus flat is +7.55 (95% CI -3.82–+24.91) at m=1 and +3.00 (95% CI -3.09–+9.82) at m=4; the corresponding success contrasts are +0.09 (95% CI -0.18–+0.36) and +0.09 (95% CI -0.18–+0.36). Point estimates are positive but intervals include zero, so this is exploratory planning evidence rather than a broad success claim. Candidate-interface errors remain excluded only from valid-row means. See `scienceworld_expand_success_summary.csv` and `scienceworld_expand_success_paired.csv`.


### Held-out extra ScienceWorld stress

Four additional official dev variations (find-plant/find-animal, 163/164) were run with the same v7 protocol and are kept separate because no-imagination was not run. Flat and provenance both achieve success 0.250 and reward -58.0 at duplication 1, and the same values at duplication 4; the two plant cases terminate on the first public action with a -100 score. This negative result widens the audit while showing that the current candidate shortlist, rather than provenance aggregation, is the limiting factor on these variations. See `scienceworld_v8_extra_summary.csv`.


### v9 public-discovery policy check

A public-interface repair was tested on four additional variations: when the requested category was absent from the current observation, the shortlist offered category-relevant navigation instead of arbitrary `focus on door/room` actions. The duplication-1 run has one transport error; among valid rows flat success is 0.750 (reward 73.25) and provenance success is 0.667 (reward 67.0). The duplication-4 retry has eight transport timeouts and is excluded from method comparison. These files are a policy diagnostic, not a new duplication result; v7 remains the complete fair comparison. See `scienceworld_v9_discovery_summary.csv` and `scienceworld_v9_discovery_rows.csv`.


### Multi-root opaque provenance stress

To remove the single-root and lexical-key shortcuts, a separate evaluator composes 162 train, 43 dev, and 37 test multi-root tasks, hides premise/source names in paraphrased branch reports, randomizes draw metadata, and retains 16 rollouts per root. The same transparent pair extractor falls to premise-edge F1 0.652 (precision 0.484, recall 1.000) and source-edge F1 0.392 (precision 0.250, recall 0.910). Learned grouping reaches downstream action correctness 0.257, reward 0.005, and regret 1.233, versus oracle correctness 1.000, reward 1.238, and regret 0.000. The stress is a diagnostic of recovery limits; the separate lexical-bank noise curves remain in `../provenance/noisy_downstream.csv`. See `../provenance_semantic_hard_final/summary.json`.


### Provenance-recovery margin diagnostic

Proposition 6 gives a sufficient action-margin condition under total-variation recovery error. In the opaque multi-root stress, the empirical learned grouping has the following margin-bin diagnostics: [.05,.10): correctness 0.000, merge 1.000; [.10,.25): correctness 0.000, merge 1.000; [.25,.50): correctness 0.875, merge 1.000; [.50,+inf): correctness 0.089, merge 1.000. Cross-root merges remain high, so the bound identifies recovery error as the active bottleneck rather than predicting a planning gain. See `../provenance_semantic_hard_final/margin_analysis.csv`.


### Provenance identifiability null

A fixed-predictor permutation of opaque test labels gives a finite-sample recovery null: premise: actual F1 0.652, permutation-null 0.652 (0.652--0.652); source: actual F1 0.392, permutation-null 0.354 (0.350--0.358). Premise recovery is indistinguishable from this null because the opaque observable fields do not expose a premise cue; source F1 is only modestly above its null. This supports Proposition 7's observability boundary and explains why downstream action correctness collapses under cross-root merges. See `../provenance_semantic_hard_final/identifiability_null.csv`.


### Provenance recovery breakdown

The lexical recovery audit is broken down by task family and difficulty in `provenance_edge_breakdown.csv`. Its controlled source-label noise curve is in `provenance_noise_curve.csv` and `provenance_noise_curve.png`; the aggregate reward and regret are reported without monotonicity assumptions. The opaque multi-root stress remains the stronger semantic diagnostic, with learned source F1 0.392 and downstream reward 0.005 versus oracle 1.238.


### Real-environment duplication counterfactual

A legacy paired state-replay artifact fixes an archived ScienceWorld state and changes the stipulated multiplicity, but the input-integrity audit found post-action mutable history in its trace-derived evidence. Its numerical rows remain archived for auditability and are excluded from clean causal claims. The pre-action serialized-input intervention below is the authoritative fixed-bank result and reports immediate reward only. See `scienceworld_interactive_counterfactual.csv`, `../interactive_counterfactual_replay/summary.json`, and `scienceworld_input_integrity/summary.json`.


### Refreshed DeepSeek ScienceWorld comparison

A fresh six-variation `find-plant` run (variations 150--155, no gold path or hidden state) completed 60 valid method-budget rows using the same public-discovery protocol. No-imagination succeeds at 1.000 at both duplication levels. Flat succeeds at 0.833/1.000 with reward 84.8/89.0 for m=1/4; LLM-readout provenance succeeds at 0.833/0.833 with reward 84.8/76.5. Algorithmic provenance-value and success-aware readouts are duplication-invariant in success (0.833) but do not exceed flat reward in this sample. Aligned m=1-to-m=4 action-flip summaries are flat 0.500 (95% CI 0.167–0.833); provenance 0.354 (95% CI 0.042–0.688); provenance_success 0.718 (95% CI 0.412–0.944); provenance_value 0.718 (95% CI 0.412–0.944). This is a stronger fresh negative/diagnostic result: provenance reduces duplicate sensitivity in the controlled estimator, but broad interactive planning improvement remains unestablished. See `scienceworld_recharged_summary.csv`, `scienceworld_recharged_paired.csv`, and `scienceworld_recharged_action_consistency.csv` / `scienceworld_recharged_action_bootstrap.csv`.


### Refreshed DeepSeek second-task-family check

The same v9 public-action protocol on six public `find-animal` variations produced six valid rows per method and budget; one transient SSL failure was retried and retained separately. Flat success is 0.500/0.500 with reward 59.8/47.5 at m=1/4, while provenance is 0.500/0.500 with reward 58.5/48.8. No method improves success in this task family; the result is a negative external-validity control. See `scienceworld_recharged_animal/scienceworld_recharged_summary.csv`.


### Refreshed confidence/action/reward table

The archived comparison table records first-to-last `p_true` changes from a closed-loop runner. Because each step can regenerate the visible state and candidate set, these are trace diagnostics rather than a causal fixed-root belief estimate. For `find-plant`, flat confidence changes by 0.383/0.508 and LLM-readout provenance by 0.375/0.508 at m=1/4; their paired action-flip rates are 0.500 and 0.354. For `find-animal`, the corresponding confidence changes are 0.183/0.100 and 0.150/0.017, with action-flip rates 0.464 and 0.336. See `scienceworld_recharged_comparison_all.csv`.


### Expanded fresh-state ScienceWorld replay

A no-new-API intervention replays 12 public states (six `find-plant`, six `find-animal`) with a common archived no-imagination suffix and changes only the first-step readout. Grouped provenance minus flat m=4 is 0.167 success and 12.5 reward on `find-plant` (bootstrap 95% CIs 0.000--0.500, 0.0--37.5); the corresponding `find-animal` contrasts are 0.000 and 0.0. The grouped readout keeps all six plant replay states successful while flat duplication-4 fails one, and remains neutral on the animal control. This is causal episode-level evidence under a selected common continuation, not a broad closed-loop success estimate. See `scienceworld_recharged_counterfactual_all/summary.csv`, `paired.csv`, and `bootstrap.csv`.


### Fresh-key contiguous interactive extension

A fresh-key v9 extension covers six additional `find-plant` variations with 24 valid rows and no API errors. Flat and success-aware provenance both have success 1.000/0.833 and reward 92.0/87.8 at m=1/4; the paired provenance-minus-flat contrasts are zero for success and reward at both budgets (95% reward CI -12.5--12.5 at m=4). Their m=1-to-m=4 action-flip rates are 0.100 and 0.655, showing that readout changes need not change the episode outcome. This null result strengthens the boundary that provenance preservation enforces duplication invariance but does not guarantee planner improvement. See `scienceworld_recharged_extension/summary.csv`, `paired.csv`, `paired_bootstrap.csv`, and `action_bootstrap.csv`.


### Fresh-key second-task-family extension

The same v9 extension on six additional `find-animal` variations also has 24 valid rows and no transport failures. Flat success is 0.000/0.167 with reward 14.2/30.8 at m=1/4; success-aware provenance is 0.000/0.000 with reward 6.0/17.0. The paired provenance-minus-flat reward contrast is -13.8 at m=4 (95% CI -36.3--1.7), a negative external-validity control. See `scienceworld_recharged_animal_extension/summary.csv`, `paired.csv`, and `paired_bootstrap.csv`.


### Fresh-key fair LLM-readout extension

A separate fresh-key extension reuses the archived candidate and conditional-rollout bank but adds the same LLM provenance readout as the primary protocol on six additional `find-plant` variations. It has 24 valid method-budget rows and 0 transport errors. At m=1, flat and provenance both reach success 1.000 and reward 92.0; at m=4, flat reaches 0.833/87.8 while provenance reaches 0.667/83.7. The paired provenance-minus-flat m=4 reward contrast is -4.17 (95% CI -16.67--8.33), and the success contrast is -0.167 (-0.667--0.333). Root-confidence deltas are 0.258/0.242 for flat and 0.258/0.233 for provenance at m=1/4; action-flip rates are 0.100/0.196. This is a fair negative extension, separate from the algorithmic success-aware extension, and does not establish broad planner improvement. See `scienceworld_recharged_llm_extension/summary.csv`, `paired_bootstrap.csv`, `action_bootstrap.csv`, and `metadata.json`.


### Fresh-key fair LLM second-task extension

The same cached-bank intervention covers six additional `find-animal` variations with 24 valid rows and 0 transport errors. At m=1 flat/provenance have success 0.000/0.000 and reward 14.2/17.0; at m=4 they have success 0.167/0.000 and reward 30.8/17.0. The paired provenance-minus-flat m=4 reward contrast is -13.83 (95% CI -46.00--4.50), while action-flip rates are 0.414/0.229. This negative task-family control keeps the fair LLM comparison separate from the success-aware extension. See `scienceworld_recharged_animal_llm_extension/summary.csv`, `paired_bootstrap.csv`, and `action_bootstrap.csv`.


### Consolidated interactive evidence ledger

A read-only ledger now joins 96 valid episode-method-budget rows across 12 refreshed public variations per task family. The primary `refreshed_llm` stratum compares flat with the LLM-readout provenance planner; the separate `fresh_key_success` stratum compares flat with the algorithmic success-aware provenance readout, so their estimates are not pooled as one method. At duplication 4, provenance-minus-flat reward is -12.5 (95% CI -37.5--0.0) for refreshed `find-plant` and 1.3 (0.0--4.0) for refreshed `find-animal`; the fresh-key success-aware contrast is 0.0 (-12.5--12.5) and -13.8 (-36.3--1.7), respectively. Mean aligned action-flip rates for flat/provenance are 0.500/0.354 in the refreshed plant stratum and 0.464/0.336 in the refreshed animal stratum; the fresh-key extension is retained separately in the machine-readable ledger. This consolidation improves auditability while leaving broad interactive success improvement unestablished. See `scienceworld_interactive_aggregate/summary.csv`, `paired.csv`, `paired_bootstrap.csv`, `action_consistency.csv`, and `metadata.json`.


### Machine-generated interactive main table

The submission table contains 32 summary rows and 56 paired contrast rows over the refreshed and fresh-key protocols. It aligns success, reward, steps, token cost, root-confidence deltas, and duplication-1-to-4 action flips; planner readouts stay explicitly labeled by protocol. This table is the numerical source for the interactive paragraphs above and does not pool incompatible readouts. See `interactive_main_table/summary.csv`, `contrasts.csv`, and `metadata.json`.


### Refreshed ScienceWorld state-replay counterfactual

Using the six refreshed public states, the archived candidate/bank, and the same no-imagination suffix, only the first-step readout was changed. Grouped provenance reaches success 1.000 and reward 89.0; flat reaches 1.000/89.0 at m=1 and 0.833/76.5 at m=4. The m=4 flat readout fails on one of six states while grouped provenance succeeds on all six. This is causal episode-level evidence that duplication can alter a real action and outcome, but it is a selected common-continuation diagnostic rather than a broad closed-loop success estimate. See `scienceworld_recharged_counterfactual/summary.csv`.


### Grouped-value interactive exploratory run

A separate four-variation find-plant run uses the same DeepSeek candidate and conditional-rollout generation but applies the algorithmic unique-sample conditional-value readout. At duplication 1, flat success/reward are 0.750/85.75 and grouped provenance is 1.000/92.00; at duplication 4 they are 0.750/85.75 and 0.750/87.75. Paired provenance-minus-flat reward is +6.25 (95% CI +0.00–+18.75) at duplication 1 and +2.00 (95% CI -12.75–+18.75) at duplication 4; success differences are +0.25 (95% CI +0.00–+0.75) and +0.00 (95% CI -0.75–+0.75). This is a small task-family exploratory result, not the broad interactive success claim.


### Clean fixed-bank ScienceWorld readout intervention

To isolate readout effects, we froze 12 pre-action public states (six `find-plant`, six `find-animal`), eight conditional samples, and the candidate set from serialized original API inputs. The clean run completed 180 valid DeepSeek readouts (180 planned, 0 errors), with 48 candidate-action replays matching the public simulator state. Flat `p_true` drift from the evidence-only value rises from 0.483 at m=1 to 0.539/0.550 at m=4/8 on `find-plant`; the corresponding `find-animal` values stay 0.417. The m=4/8 action-flip rates are 0.000/0.000 on plant and 0.000/0.000 on animal, while the independent byte-identical m=1 repeat has zero action and confidence drift. Provenance-value keeps the evidence-only belief fixed and is duplication-invariant; immediate reward contrasts are reported as one-step simulator diagnostics. This experiment does not claim episode success or root calibration. See `scienceworld_frozen_readout_clean_20260929/summary.csv`, `paired.csv`, `bootstrap.csv`, and `audit.json` and `report/scienceworld_input_integrity/summary.json`.


### Expanded four-step fixed-bank ScienceWorld intervention

A preregistered extension covers 96 pre-action public states (the first four decisions of 12 archived variations in each of `find-plant` and `find-animal`). It completed 1440 valid DeepSeek readouts from 1440 planned calls with 384 candidate-action replays, all matching the public simulator state. In `find-plant`, flat confidence drift is 0.008/0.019 at duplication 4/8 and action flips are 0.021/0.042; the corresponding flat immediate-reward changes from m=1 are -0.11/-0.17. Provenance-value and success-aware provenance keep belief and action duplication-invariant; success-aware immediate reward is 16.35 versus flat 17.73/17.62 at m=1/4. Across both families at m=8, flat action flips are 0.031 (95% CI 0.003--0.066) and absolute confidence drift is 0.012 (0.002--0.024), while the repeat-control excess flip rate is 0.017 (-0.007--0.052). No paired comparison had both a reported probability change and an action change; all six harmful immediate-reward flips had unchanged reported probability. This is stronger fixed-state action/outcome evidence, but it reports immediate rewards only and does not claim confidence-mediated action changes, episode success, or broad planner improvement. See `scienceworld_frozen_readout_first4_20260929/{summary.csv,paired.csv,bootstrap.csv,curve.png,audit.json,ANALYSIS.md,episode_bootstrap.csv}`.


### Three-seed DeepSeek controlled task

A minimal fixed-bank API diagnostic covers three seeds, duplication 1/2/4/8, and flat, trajectory-average, simple-dedup, source-grouping, and algorithmic provenance-value methods. It archives 60 valid method-budget rows from 30 request bodies; the 12 flat/trajectory-average request bodies are byte-identical, yet the API can return different outputs, so this is also a direct temperature-zero stochasticity control. Flat mean `p_true` is 0.443/0.443/0.507/0.523 at m=1/2/4/8, with action-flip rates 0.000/0.000/0.333/0.333; provenance-value keeps `p_true` at 0.481 and its action invariant across duplication. The three-seed intervals are wide and this model-specific diagnostic does not support a universal LLM claim. See `deepseek_mvp_20260929/summary.csv`, `paired.csv`, `bootstrap.csv`, `same_body_control.csv`, `curve.png`, and `metadata.json`.


### Local Qwen short-horizon ScienceWorld smoke

A compatible local-Qwen runtime completed a two-step `find-plant` smoke on variation 159 at duplication 1/4 for flat and provenance (4 valid rows, 0 errors). Both methods reached score 8.0, reward 0.0, and success 0.0 at both budgets; flat token cost was 6541/9419 and provenance 6580/6580. The CUDA probe reports cuda:0 on NVIDIA GeForce RTX 4090. The horizon is too short for episode-success inference, so this is a second-model schema/feasibility diagnostic and not a planner result. See `scienceworld_local_qwen_mvp/{summary.csv,paired.csv,metadata.json}`.


### Local Qwen full-episode interactive pilot

A CUDA local-Qwen pilot completes 12 valid episode-method-budget rows on three `find-plant` variations (152, 158, 161), with flat and success-objective provenance at duplication 1/4. On the two-variation grid (152, 158), flat success is 0.000/0.500 at m=1/4 and provenance-success is 1.000/1.000; the held-out variation 161 is successful for both methods at both budgets. Across the three paired episodes, provenance-success minus flat reward is 16.7 at m=1 and 5.7 at m=4. The initial Brier diagnostic is 0.25 for flat and 0.64 for success-objective provenance in this tiny sample. This small second-model pilot is not pooled with DeepSeek and is not a confirmatory success estimate; it only shows that the full local planner path is executable on CUDA and that objective/readout choice can matter. See `local_qwen_episode_pilot/{summary.csv,paired.csv,bootstrap.csv,metadata.json}`.


### Expanded local Qwen CUDA interactive pilot

A same-runner CUDA pilot covers five official `find-plant` variations (152--154, 158, 161), flat and success-aware provenance, and duplication 1/4, for 20 valid episode-method-budget rows and 5 paired episodes. The raw 150--155 extension retains 10 candidate-schema error rows (five repeated method/variation failures at each budget) and excludes them from paired summaries. Mean flat/provenance-success episode success is 0.200/0.800 at m=1 and 0.400/0.800 at m=4. Paired reward differences are 41.80 (95% CI 13.40--72.00) at m=1 and 35.20 (3.40--68.60) at m=4. This remains a small second-model CUDA pilot; it is separate from DeepSeek, uses an objective-specific readout, and does not establish broad interactive improvement. See `local_qwen_expanded_pilot/{summary.csv,paired.csv,bootstrap.csv,metadata.json,raw_errors.csv,curve.png}`.


### Cached local Qwen provenance-value replay

Using the same serialized candidate/bank requests and public environment, a deterministic `provenance_value` replay covers five complete plant variations (152--154, 158, 161), 20 valid rows and five paired episodes at m=1/4. It made zero new model requests and the cache audit is byte-identical. Flat success is 0.200/0.400 at m=1/4 versus 0.800/0.800 for provenance-value; paired reward differences are 41.80 (95% CI 13.40--72.00) and 35.20 (3.40--68.60). The identical m=1/m=4 provenance actions provide a direct second-model duplication-invariance check, while the small sample remains exploratory. See `local_qwen_value_replay/{summary.csv,paired.csv,bootstrap.csv,cache_audit.json,metadata.json}`.


### Local Qwen second-task-family value control

One `find-animal` variation (153) was run end-to-end on CUDA with flat and algorithmic `provenance_value` at m=1/4. Flat failed at both budgets (success 0.000, reward 67), while provenance-value succeeded at both (success 1.000, reward 92); the paired reward contrast is +25 at each budget. This is a one-episode task-family control and is retained as exploratory external-validity evidence. See `local_qwen_animal_value_replay/{summary.csv,paired.csv,bootstrap.csv,metadata.json}`.


### Expanded local Qwen find-animal control

The same CUDA Qwen v9 protocol requested six `find-animal` variations and produced 17 valid rows, seven retained schema errors, and three complete paired variations (150, 153, 155). On complete pairs, flat success is 0.000/0.000 versus provenance-value 0.667/0.667 at m=1/4; paired reward differences are 36.33 (95% CI 9.00--75.00) and 19.67 (9.00--25.00). This second-task-family result remains exploratory and excludes incomplete variation blocks from inference. See `local_qwen_animal_expanded_pilot/{summary.csv,paired.csv,bootstrap.csv,raw_errors.csv,metadata.json}`.


### Local Qwen task-family interactive main table

A stratified CUDA Qwen table joins the five complete plant pairs and three complete animal pairs for flat versus algorithmic `provenance_value` at m=1/4 (32 valid method-budget rows, eight paired episodes). Plant success is 0.200/0.400 for flat and 0.800/0.800 for provenance-value at m=1/4; animal success is 0.000/0.000 versus 0.667/0.667. Provenance reduces mean unnecessary no-change actions on plant from 2.60 to 0.40, and on animal from 3.33 to 0.67; token cost is also lower because the algorithmic readout adds no LLM call. The table reports reward, steps, calibration Brier, and tokens by task family, with paired bootstrap intervals. It is exploratory and does not establish broad interactive improvement. See `local_qwen_interactive_main_table/{summary.csv,paired.csv,contrasts.csv,curve.png,metadata.json}`.


### Fresh CUDA-Qwen ScienceWorld expansion

A fresh CUDA-Qwen paired expansion covers six new official variations (three `find-plant`, three `find-animal`), flat versus algorithmic `provenance_value`, and duplication 1/4. It contains 24 valid rows and 6 complete episode pairs with no schema errors. On find-plant, provenance-minus-flat reward is 41.7 at m=1 and 38.7 at m=4; on find-animal it is 8.7 and 14.3. Success contrasts are 0.667/0.667 for plant and 0.333/0.333 for animal at m=1/4. The provenance actions are identical across duplication for all six pairs, while flat action flips occur at the paired-step rates reported in `action_bootstrap.csv`. This is a fresh second-model GPU result with task-family strata and small-sample bootstrap intervals; it strengthens the planning signal but remains exploratory and does not establish a universal success-rate gain. See `local_qwen_expansion_162_164/{summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,action_bootstrap.csv,metadata.json,curve.png}`.

### Second fresh CUDA-Qwen ScienceWorld expansion

A second held-out CUDA-Qwen expansion covers six official variations (165--167 in `find-plant` and `find-animal`) at duplication 1/4. It retains 23 valid method-budget rows, 5 complete paired episodes, and 1 schema-error rows. On the five complete pairs, provenance-value reward contrasts are 92.0/58.5 for plant and 5.7/5.7 for animal at m=1/4; success contrasts are 1.000/1.000 and 0.000/0.000. Provenance action flips are zero in both family strata; flat flips are 0.438/0.417 for plant/animal. This is independent second-model GPU evidence with one retained malformed readout excluded from paired inference; it strengthens the conditional planning signal while remaining exploratory. See `local_qwen_expansion_165_167/{summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,action_bootstrap.csv,metadata.json,curve.png}`.

### Fair algorithmic flat-value control

A four-pair CUDA-Qwen ScienceWorld control compares algorithmic `flat_value` with `provenance_value` using the same candidate bank, conditional-value objective, and tie-breaking. Flat-value alone applies the declared duplicate-count pseudo-likelihood to the premise belief. All 16 method-budget rows completed without error; reward and success contrasts are 0.0/0.0 for plant and 0.0/0.0 for animal at m=1/4, with zero paired differences in this bank. This null control shows that provenance gains require a readout/value bank whose duplicate-sensitive belief can affect action scores; provenance preservation itself does not promise improvement when every conditional action value is tied. See `local_qwen_fair_flatvalue_168_169/{summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,action_bootstrap.csv,metadata.json,curve.png}`.

### Non-pooled cross-model interactive effects

The cross-model table keeps DeepSeek and Qwen strata separate rather than averaging them and now adds 32 ALFWorld public-observation contrast rows. The Qwen stratum aggregates 27 paired episodes after the fresh CUDA expansion. DeepSeek provenance-minus-flat reward estimates range from -13.83 to 2.83 across protocols and task families, while success-aware estimates range from -13.83 to 0.00; Qwen provenance-value reward contrasts range from 0.00 to 44.21. This heterogeneity is itself part of the result: the mechanism and duplication-invariance guarantee are general at the estimator level, while interactive reward effects depend on model, task family, and readout objective. See `cross_model_interactive_table/{strata.csv,ranges.csv,metadata.json}`.


### CUDA-Qwen effect heterogeneity

The consolidated Qwen study is accompanied by a non-pooled forest plot with 16 reward strata across four fresh expansions, two task families, and duplication 1/4. Each interval resamples episode variations; the display keeps variation dependence visible instead of treating the positive reward range as a universal planner effect. See `local_qwen_effect_heterogeneity/{heterogeneity.csv,forest.png,metadata.json}`.

### Independent held-out CUDA-Qwen null control

A fresh flat-value/provenance-value comparison on plant variations 174--176 contains 12 valid rows and 3 complete pairs with no retained errors. Both methods have zero episode successes and identical reward/final-score means at duplication 1 and 4; aligned action flips are zero. This negative boundary confirms that provenance invariance is a correctness property rather than a universal planning-improvement guarantee. See `local_qwen_heldout174_176/{metadata.json,summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,curve.png}`.

### Independent held-out CUDA-Qwen animal null control

A second fresh flat-value/provenance-value comparison on find-animal variations 177--179 contains 12 valid rows and 3 complete pairs with no retained errors. Both methods have zero episode successes, identical reward/final-score means at duplication 1 and 4, and zero aligned action flips. This separate task-family null reinforces that candidate generation, rather than provenance normalization, is the limiting factor on these states. See `local_qwen_heldout177_179/{metadata.json,summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,curve.png}`.

### ALFWorld TextWorld public-observation smoke

A separate ALFWorld TextWorld check uses only public observations and admissible commands. The validated DeepSeek strata contain 120 valid method-budget rows over 6 task types (108 base rows plus 12 cached success-objective ablation rows): no-imagination succeeds on 0.333 of episodes, flat on 0.222, and algorithmic `provenance_value` on 0.167; duplication-1/4 aligned flat action flips average 0.093. The reported first-to-last root-confidence diagnostic changes by 0.192/0.189 for flat and 0.233/0.233 for provenance at m=1/4; this trace diagnostic is not treated as a fixed-root causal belief update. The success-objective ablation reaches 0.167 row success, so objective choice remains a separate factor. A CUDA Qwen2.5-Coder-3B stratum contains 12 rows over 3 task types and reaches 0.000/0.000 success for flat/provenance, with flat action flips 0.233; its corresponding root-confidence diagnostic is 0.000/0.000. The provenance readout is duplication-invariant in its algorithmic belief trace here, while model/task success remains heterogeneous. These rows verify the multi-environment path and are exploratory; they do not establish a broad interactive-success gain. See `alfworld_interactive/{summary.csv,paired.csv,paired_objectives.csv,action_consistency.csv,bootstrap.csv,action_bootstrap.csv,metadata.json,curve.png}`.


### Second-task-family grouped-value check

One easy official find-animal variation was rerun at duplication 1 and 4 from fresh request directories. Both flat and grouped provenance completed it with reward 92.0 at both budgets; this is a null second-family control with n=1 and is not evidence of a general success-rate gain. See `scienceworld_value_animal_summary.csv` and `scienceworld_value_animal_curve.png`.


### API-limited contiguous extension

The contiguous 156--161 extension is archived separately. Its plant cache completed, while animal rows include retained HTTP 402 insufficient-balance failures; these rows are excluded from all method and success claims. See `api_status.json` and the raw extension summaries.


### ALFWorld fair algorithmic flat-value control

A Python 3.14 TextWorld grammar-namespace compatibility patch enabled a new public-observation ALFWorld control. Across 6 paired rows from three games, `flat_value` and `provenance_value` reuse identical candidate and conditional banks. At duplication 4, flat has mean belief drift 0.218 relative to provenance, while paired success and reward differences are exactly zero. This isolates duplication-invariant belief handling from an extra LLM reader; the small batch does not establish an episode-level planning gain. See `alfworld_interactive/{paired_algorithmic.csv,bootstrap.csv,metadata.json}`.

### Qwen semantic rollout-prompt intervention

A frozen 12-state local-Qwen probe replaces the compact rollout prompt with explicit conditional score semantics. Among eight valid semantic banks, exact template-value copying falls from 1.000 to 0.000 and constant-bank rate from 1.000 to 0.375; four of 12 semantic calls fail the JSON contract. No valid state has a premise-dependent action winner, so this corrects a value-collapse confound without establishing a planning gain. See `qwen_semantic_bank_probe/{protocol.json,summary.csv,rows.csv,ANALYSIS.md}`.

## Limitations

* The controlled experiment is synthetic and the flat effect is deliberately stipulated to isolate the mechanism.*
* The real-model studies are small and model-specific; they do not establish a universal LLM behavioral law.*
* ScienceWorld evidence is now multi-variation but still insufficient for a broad provenance-planner improvement claim.*
* The original local Qwen ScienceWorld smoke returned incomplete conditional-rollout JSON; a compact serialization retry produced one valid flat/provenance pair (n=1, both unsuccessful). A three-variation compact grid then produced six valid rows, with flat success 0.333 and provenance success 0.000; these remain schema/model feasibility diagnostics rather than planner evidence.*