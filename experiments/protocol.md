# Controlled main experiment, revision 4

Frozen before the revision-4 run. Earlier MVP and v1–v3 outputs are exploratory
and cannot establish LLM behavior: the model's text did not affect their scores.
No local model training, attention experiment, public environment or source
recovery system is part of this run.

## Environment and information boundaries

H is a binary premise (the designated route has permission). Enumerate both
H=0 and H=1 for two actions. Prior p in {.1,.3,.5,.7,.9}; real binary sensor
accuracy L/(1+L), L in {1.25,2,4}. Given the real sensor, q is its exact Bayesian
posterior under the stated prior. Hidden H and the physical state label are
crossed, not passed to any planner. Both correct and incorrect premises are
reported separately. The grid is deliberately balanced, so aggregate Brier/NLL
on this grid must not be described as population calibration under each prior.

Conditional rollout returns are independent draws from N(mu[h,a], 1.5^2).
Each scenario has 2 utility families (opposing routes and risky-versus-safe),
with random utility offsets, and 30 seeds per factorial cell. Bank n=32 per
(h,a) is frozen. Real execution uses hidden H and a held-out Gaussian random
number shared by all methods. Reward, expected reward, regret relative to the
posterior-optimal action and value RMSE are distinct fields. Negative reward is
allowed. We do not force all independent samples to favor the designated root.

## Interventions

At m in {1,2,4,8,16}, use the same real evidence, environment, candidate actions,
and nested rollout bank:

* pure_duplication: repeat both existing H=1 samples for both actions m times.
* selective_duplication: repeat just one preselected sample for one action m times.
* paraphrase_duplication: repeat the same sample IDs with equivalent text forms.
* independent_rollout: use 2m fresh conditional draws in every (h,a) cell; q is fixed.
* new_real_evidence: retain the base rollout bank and reveal m real sensor draws;
  only this condition may update q.

Copies share sample_id. Independent draws have different sample_id even if their
return or rendered text coincides. Semantic duplicates and source identity are
not interchangeable. Order tests permute identical multisets. Numeric methods
ignore presentation padding; length controls for actual LLM calls match complete
user-prompt character count and verify API token usage (character equality alone
is not token equality). Padding is explicitly non-evidence. LLM results compare
unpadded and padded regimes with two independent presentation orders.

## Seven methods (analytic structure experiment)

1. Flat: weight every presented return by q_h, normalize globally per action.
   Its diagnostic root mass is q*n1/(q*n1+(1-q)*n0). This is a *specified pooled
   estimator*, not evidence that an LLM inherently uses that formula.
2. Single trajectory average: equally average scored returns per action, ignoring
   the enumeration weights; root mass is the enumerated H=1 fraction. This can
   be correct under samples drawn from q; our bank is stratified by H, so it is
   an ablation of ignoring sampling weights, not a calibrated Bayes competitor.
3. Exact dedup: dedup actual text, then apply flat. Text excludes source IDs.
4. Semantic dedup: canonicalize the finite grammar's paraphrases and group by
   (hypothesis,action,displayed return), then flat. This is controlled semantic
   equivalence, NOT a learned embedding/source extractor. All displayed return
   values and action distinctions are preserved; no arbitrary group truncation.
5. Source group average: q-weighted conditional means by (H,action), with no
   sample-identity dedup. It handles uniform copies but may reweight a selectively
   duplicated sample.
6. Standard explicit belief mixing: retain each independent sample_id once,
   estimate each conditional mean, compute Q(a)=sum_h q(h)*mean_return(h,a).
7. Provenance preserving: separate evidence q from generated consequences,
   canonicalize source/sample identity, use the SAME formula as method 6.

Methods 6 and 7 must coincide. They have oracle generation provenance in this
controlled experiment. Novelty or superiority to standard Bayesian planning
cannot be inferred. A correct provenance interface is an assumption, not learned.

## Actual LLM experiment

Single DeepSeek model. The flat baseline concatenates trajectories and asks for
root probability and action; it is not forced to use a wrong formula. Exact and
semantic dedup apply their text preprocessing before the same readout. Single
trajectory scoring asks the same model for each individual score then averages.
Source averaging groups these scores using recorded provenance. Explicit mixing
and provenance methods obtain q from an evidence-only API call and combine known
conditional returns, counting each independent sample once. This is a hybrid
API/symbolic implementation; known numeric returns are not LLM-estimated rewards.
All prompts and responses, returned model identifier, token counts, latency,
parse/API failures, and cache reuse are retained. No fabricated fallback data.
Seeds define task/bank/order RNG; the API is NOT claimed to support seeded output.
Exact replay is from archived responses, not guaranteed future server behavior.

## Statistics and decision rules

Each grid cell has 30 independent scenario seeds. Means, sample SD and 95% t
intervals are reported for every setting. Global summaries first average the
factorial grid within each seed; intervals bootstrap these seed clusters (10,000
paired resamples). Copies are never statistical units. H-correctness strata are
separate. CIs are not multiple-testing adjusted; all are descriptive estimates.
Report paired m=16 minus m=1 changes, action flips, harmful/helpful flips, Brier,
NLL, true-state and expected reward, value RMSE and posterior regret. Precision
tolerance for exact source invariance and mixing equivalence: 1e-12. Empirical
LLM equivalence bound: +/- .02 probability; do not infer invariance from a
non-significant effect. A negative or absent flat-LLM effect is a valid result.

The LLM subset is smaller than the numeric grid and is reported independently;
no analytical row is counted as an LLM decision. Length and ordering findings
refer to measured prompts, not metadata flags. Results are controlled synthetic
findings only; no cross-model or natural-environment claim is permitted.
