# Shared-premise evidence: definitions and propositions

This section states the claims used by the experiments. It distinguishes an
imagined consequence from an observation, and a repeated draw from a new draw.
It does not claim that every language model implements the flat estimator.

## Definitions

Let `E` be evidence actually obtained from the environment before planning,
including source IDs. Let `H` be a finite latent premise or configuration, such
as whether a tool is available. A candidate action is `a`.

A frozen world model `M` produces a rollout

```text
tau_j = (H, a, U_j, Y_j, sample_id_j, parent_ids_j),
Y_j ~ K_M(. | H, a, U_j).
```

`parent_ids_j` records the premise and real evidence used to generate the
rollout. A descendant group `G_g` contains rollouts with the same premise and
parent evidence. A pure duplicate reuses the same `sample_id`, `U_j`, text and
value; only presentation multiplicity changes. A new independent condition
sample has a new `sample_id` and fresh `U_j` while holding `E`, `H`, `a` and `M`
fixed. A new real observation changes `E` and may update the belief over `H`.

We use *independent evidence* only for a new environment observation whose
likelihood is not a deterministic function of the existing evidence. A fresh
conditional rollout is independent value information about `Q(a,h)`; it is not
independent evidence about `H` because it is generated under a stipulated
branch and does not change `E`.

The planner has separate quantities:

```text
q(h) = P(H=h | E),
Q(a,h) = E[Y | H=h, a, E],
U(a) = sum_h q(h) Q(a,h).
```

Imagined descendants may improve an estimate of `Q`; they are not automatically
additional observations in `E`.

### Provenance-preserving aggregation algorithm

For each planning state, the implementation takes the evidence-only belief
`q`, the candidate action set, and a bank of rollout records. It first checks
that every record has a declared premise branch, parent/source lineage, and
sample identifier. It then performs the following deterministic operations:

```text
1. Partition records by (action, premise branch, source group).
2. Within each partition, collapse repeated sample_id records once.
3. Compute the conditional mean for each retained sample/group.
4. Apply one normalized group weight per source group.
5. Mix conditional values with q, without updating q from imagined records.
6. Apply a fixed objective and tie-breaking rule to choose an action.
```

With a hashable lineage key, grouping and duplicate removal take `O(N)` time
and `O(N)` memory for `N` rollout records; the final action/premise mixture is
`O(AH)` for `A` actions and `H` premise branches. A learned extractor changes
only step 1: its uncertainty is measured separately and is not hidden inside
the aggregation result.

## Proposition 1 (no-information invariance)

Let `B` be the complete bank of declared conditional descendants already
available to the planner. If a presentation object `D` is a deterministic copy,
rewrite, or permutation of `B` and contains no new parent evidence, then

```text
P(H | E, B, D) = P(H | E, B).
```

The important conditioning is on `B`: a rollout may contain a *stipulated*
branch label such as `H=true`, but that label is an intervention supplied by the
planner, not a sensor reading. The equality says that re-presenting the same
declared bank cannot update the physical-world belief. An approximate model can
still change its score after seeing `D`; that change is an estimator or interface
failure, not Bayesian evidence.

**Proof sketch.** Since `D=g(B,layout)` for a deterministic presentation map,
`P(d | e,b,m)` is one for the observed `d=g(b,layout)` and zero otherwise. It
therefore cancels from Bayes' rule, leaving `P(h | e,b,m)`. A pure duplicate
changes only presentation multiplicity, while a new real observation changes
`E` and is outside this proposition.

## Proposition 2 (flat pseudo-likelihood bias)

Suppose a flat aggregator treats every *additional* descendant copy as an
independent observation with likelihood ratio `lambda > 1` in favor of `H=1`.
The canonical descendant at duplication 1 is the baseline, so `m` counts
additional copies and total presentation multiplicity is `m+1`. Starting from
`q_0=P(H=1|E)`, after `m` additional copies it reports

```text
logit(q_flat(m)) = logit(q_0) + m log(lambda),
q_flat(m) = sigmoid(logit(q_0) + m log(lambda)).
```

The duplication bias in log odds is

```text
B(m) = logit(q_flat(m)) - logit(q_0) = m log(lambda),
```

and the marginal probability change is
`d q_flat / d m = q_flat(m)(1-q_flat(m)) log(lambda)`.
The probability eventually saturates near one although the information about
`H` has not changed. For `lambda<1` the symmetric effect drives confidence toward
zero. If the premise is false, the induced action error can increase until an
action threshold is crossed.

This is a property of the specified flat pseudo-likelihood estimator, not a
theorem that a transformer or arbitrary rollout reader uses this rule.

## Proposition 3 (provenance-preserving duplication invariance)

Let `G_g(a,h)` be premise/source groups restricted to one action and one
declared branch. Define normalized group estimates

```text
Q_hat(a,h) = sum_g w_g mean_{j in G_g(a,h)}[Y_j],
sum_g w_g = 1.
```

A pure duplicate retains the same group and `sample_id` and does not increase
`w_g`. The provenance-preserving action score is

```text
U_hat(a) = sum_h q(h) Q_hat(a,h).
```

Adding exact copies to a group leaves `q`, `Q_hat` and `U_hat` unchanged.
Independent samples get new sample IDs and can change the group mean; their
value is retained.

**Proof sketch.** Within a group, copying every occurrence multiplies both the
numerator and denominator of the normalized mean by the same integer. Group
weights and premise evidence are unchanged. Substitution into the finite sum for
`U_hat` gives equality before and after copying. A new independent sample adds a
new random draw, so the estimate may change while `q` still depends only on `E`.

## Proposition 4 (independent-rollout value gain)

For independent conditional samples with finite variance
`Var[Y | h,a] = sigma^2_{h,a}`, the sample mean satisfies

```text
E[Q_hat_n(a,h)] = Q(a,h),
Var[Q_hat_n(a,h)] = sigma^2_{h,a}/n.
```

Provenance-preserving aggregation can therefore reduce conditional-value error
as the independent rollout budget grows without changing premise belief. A reward
gain is possible only when the improved value estimate changes the selected
action toward a higher true `U(a)`; duplication alone has no such variance
reduction.

## Corollary (action-flip threshold)

For a binary action with threshold `q=1/2`, a flat aggregator flips from the
baseline action when the number of additional copies satisfies

```text
m > -logit(q_0) / log(lambda)
```

for `lambda>1` and baseline `q_0<1/2`. The corresponding total duplication is
`m+1`. The smallest integer satisfying the inequality is the diagnostic
threshold used by the controlled benchmark; it does not select or filter
real-environment episodes.

## Proposition 5 (objective-agnostic invariance)

Let `S(a)` be any deterministic action statistic computed from the unchanged
real-evidence belief `q` and the provenance-normalized conditional summaries,
including normalized predicted score, conditional success probability, or a
fixed risk-sensitive utility. If the action rule is

```text
a* = argmax_a S(a),
```

then pure descendant duplication cannot change `S(a)` or `a*` (up to a fixed
tie-breaking rule). The invariance guarantee therefore applies to the
aggregation layer independently of whether the downstream planner optimizes
reward, success, or risk.

**Proof sketch.** Proposition 3 leaves every normalized conditional summary and
`q` unchanged under exact copies. A deterministic `S` receives the same inputs
before and after duplication, so all scores and the tie-broken argmax are the
same. This separates the correctness guarantee from the choice of planning
objective; it does not guarantee that one objective is more accurate than
another.

## Proposition 6 (provenance-recovery margin bound)

For a fixed action `a` and premise branch `h`, let `nu_(a,h)` be the weighted
distribution of true source-group conditional means and let `nu_tilde_(a,h)` be
the distribution induced by a recovered provenance graph. Assume all values lie
in an interval of width `R`. If their total-variation distance is at most
`delta_(a,h)`, then

```text
| Q_tilde(a,h) - Q_hat(a,h) | <= R delta_(a,h),
| U_tilde(a) - U_hat(a) | <= R sum_h q(h) delta_(a,h).
```

If the true best action `a*` has margin `Delta_a = U_hat(a*)-U_hat(a)` over
every competitor and

```text
Delta_a > R sum_h q(h) [delta_(a*,h) + delta_(a,h)],
```

then recovered provenance selects the same action as the oracle grouping.

**Proof sketch.** For any two distributions supported on an interval of width
`R`, the difference of expectations is bounded by `R` times total variation.
Apply this bound to each conditional group distribution, then use the triangle
inequality and the convex mixture weights `q(h)`. The displayed margin condition
makes the recovered score of `a*` remain larger than the recovered score of
every competitor. The result separates a correctness guarantee for aggregation
from the separate statistical problem of recovering lineage.

## Scope and falsification

These propositions do not say that adding a different action, different
conditional sample, new hypothesis, or real observation must leave the action
unchanged. They also do not prove a learned provenance extractor will recover the
true graph. Real-model and ScienceWorld experiments therefore report prompt,
order, position, recovery-error and token-cost controls; a null LLM effect remains
a legitimate result.

## Proposition 7 (observability limit for provenance recovery)

Let `X` denote every field exposed to a provenance extractor and let `Z` be a
source or premise label. If the conditional observable distributions are
exchangeable, `P(X | Z=z_1) = P(X | Z=z_2)` for all labels, then no extractor
using only `X` can identify `Z` above its prior-label baseline. In particular,
for balanced binary labels its expected pairwise accuracy is at most `1/2`;
permuting the test labels gives the appropriate finite-sample F1 null. Any
downstream action guarantee therefore requires either observable lineage cues or
an external source identifier.

**Proof sketch.** Under exchangeability, the posterior `P(Z|X)` equals the
prior `P(Z)`. Thus the Bayes-optimal classifier is a constant prior decision,
and any randomized rule has no information about the realized label beyond that
prior. A permutation of labels preserves the joint observable sample and
provides a finite-sample null for a fixed predictor. The proposition is an
identifiability statement, not a claim that every opaque benchmark is exactly
exchangeable; partial observable correlations can yield a small F1 advantage
while still leaving large cross-root merge error and no downstream guarantee.
