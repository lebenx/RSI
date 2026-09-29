### Expanded fixed-bank ScienceWorld analysis (episode-cluster audit)

This adaptive exploratory extension uses the first four pre-action states from each of 24 archived task/variation episodes (96 states, two task families). All 1,440 serialized requests and parsed responses were independently matched to the frozen inputs and raw provider outputs; 384 candidate replays matched the public simulator state. Multiplicity changes only the true-premise descendants (eight unique samples; presented rows at m=1/2/4/8 are 8/12/20/36). It also changes context length and repetition pattern; it is not a pure manipulation of internal belief.

Primary intervals below resample 24 whole episodes within task family using 20,000 draws, after averaging the three API repetitions and four steps in each episode. The earlier `bootstrap.csv` resamples states and is retained as a secondary diagnostic. These are pointwise exploratory intervals.

| Contrast at duplication 8 | Estimate [episode-cluster 95% interval] |
|---|---:|
| Flat reported P(H=true) change | +0.01163 [+0.00313, +0.02205] |
| Flat action flip rate | +0.03125 [+0.00347, +0.06250] |
| Byte-identical input action flip rate | +0.01389 [+0.00000, +0.03125] |
| Flip rate excess over repeat control | +0.01736 [-0.00694, +0.04861] |
| Flat immediate reward change | -0.08333 [-0.25000, +0.00000] |
| Same-prompt dedup minus flat reward | +0.08333 [+0.00000, +0.25000] |
| Provenance-value minus flat reward | -0.41667 [-1.64609, +0.56250] |
| Success-objective minus flat reward (post hoc) | -0.32292 [-1.56250, +0.60417] |

Across the 864 m=2/4/8 paired comparisons, 23 actions changed and 22 reported probabilities changed, with 0 joint changes. All 6 harmful immediate-reward flips occurred without a reported probability change. These counts include correlated repetitions, budgets, and states; they are descriptive, not independent sample sizes. The data do not demonstrate that confidence inflation mediated action changes.

The deterministic provenance algorithms and cached same-prompt dedup readout are invariant by construction. Dedup reuses one independent m=1 response across budgets, so its zero flip rate is not a new stochastic robustness measurement. Provenance-value does not show a consistent reward advantage over flat or simple dedup. The success-objective analysis was added after outcomes were inspected and is explicitly post hoc. Root truth is unavailable, so reported belief drift is not a calibration or overconfidence estimate. Immediate reward is not final episode reward.

The state selection rule was fixed for this collection, but this expansion followed inspection of earlier null results; there is no externally registered confirmatory protocol. The results remain task-family-specific exploratory diagnostics, not evidence of broad closed-loop planning improvement.

Files: `episode_bootstrap.csv`, `episode_units.csv`, `paired_events.csv`, `joint_events.csv`, `verified_analysis.json`, `episode_curves.png`, and `qualitative_events.md` in `results_submission/scienceworld_frozen_readout_first4_20260929/`.
