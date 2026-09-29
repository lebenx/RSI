"""Explicit information boundaries for a tiny partially observed planner."""
from dataclasses import dataclass, replace, asdict
import math
import numpy as np

METHODS = ('flat', 'trajectory_average', 'exact_dedup', 'semantic_dedup',
           'source_average', 'belief_mixing', 'provenance')
MULTIPLICITIES = (1, 2, 4, 8, 16)
CONDITIONS = ('pure_duplication', 'selective_duplication', 'paraphrase_duplication',
              'independent_rollout', 'new_real_evidence')
TEXT_LENGTH = 112

def logit(p):
    p = min(max(float(p), 1e-12), 1-1e-12)
    return math.log(p/(1-p))

def posterior(prior, sensor, strength):
    z = math.log(prior / (1 - prior)) + sum(2 * int(x) - 1 for x in sensor) * math.log(strength)
    return 1 / (1 + math.exp(-z))

@dataclass(frozen=True)
class Sample:
    sid: str
    h: int
    action: int
    value: float
    variant: int = 0

    @property
    def text(self):
        forms = [
            'Assuming H={h}, action {a} has simulated return {v:+.4f}.',
            'With H={h} stipulated, the imagined payoff of action {a} is {v:+.4f}.',
            'Conditional simulation: H={h}; action {a}; predicted reward {v:+.4f}.',
            'If H={h} holds, action {a} yields imagined value {v:+.4f}.',
        ]
        text = forms[self.variant % len(forms)].format(h=self.h, a=self.action, v=self.value)
        return text[:TEXT_LENGTH].ljust(TEXT_LENGTH)

    @property
    def semantic_key(self):
        # Exact meaning in our finite controlled grammar, independent of sid.
        return (self.h, self.action, f'{self.value:+.4f}')

def scenario(seed, physical_state, hypothesis_direction, prior, strength, family):
    # Include every factorial setting in independent RNG streams. Pair across
    # interventions/methods, not across different scenarios.
    key = [190427, seed, physical_state, hypothesis_direction, round(prior*100),
           round(strength*100), family]
    rng = np.random.default_rng(np.random.SeedSequence(key))
    truth = int(physical_state == hypothesis_direction)
    accuracy = strength / (1 + strength)
    sensors = (rng.random(16) < (accuracy if truth else 1-accuracy)).astype(int)
    if family == 0:
        good, bad = 1.5, -1.5
        mu = np.full((2, 2), bad, dtype=float)
        for h in (0, 1):
            good_action = hypothesis_direction if h else 1 - hypothesis_direction
            mu[h, good_action] = good
    else:
        # A second utility family has one robust action and one state-sensitive
        # action, so value estimates and root confidence are not redundant.
        mu = np.full((2, 2), .75, dtype=float)
        for h in (0, 1):
            good_action = hypothesis_direction if h else 1 - hypothesis_direction
            mu[h, good_action] = 1.25 if h else -1.0
    # h=1 means the designated hypothesis direction is the good action; h=0
    # means its opposite is good. Random offsets preserve paired scenarios.
    mu = mu + rng.normal(0, .35, (2, 2))
    bank = []
    for h in (0,1):
        for a in (0,1):
            for j, value in enumerate(rng.normal(mu[h,a], 1.5, 32)):
                bank.append(Sample(f'h{h}a{a}s{j:02}',h,a,round(float(value),4)))
    sid = f's{seed:03}_t{physical_state}_h{hypothesis_direction}_p{prior:.1f}_l{strength:g}_f{family}'
    return dict(scenario_id=sid, seed=seed, physical_state=physical_state,
                hypothesis_direction=hypothesis_direction, root_correct=truth,
                prior=prior, strength=strength, family=family, sensors=sensors,
                mu=mu, bank=bank, execution_noise=rng.normal(0,.4,2))

def intervention(s, condition, m):
    base = [r for r in s['bank'] if int(r.sid[-2:]) < 2]
    q = posterior(s['prior'], s['sensors'][:1], s['strength'])
    if condition == 'independent_rollout':
        rows = [r for r in s['bank'] if int(r.sid[-2:]) < 2*m]
    elif condition == 'new_real_evidence':
        rows = base
        q = posterior(s['prior'], s['sensors'][:m], s['strength'])
    elif condition in ('pure_duplication','paraphrase_duplication'):
        rows = []
        for r in base:
            for k in range(m):
                rows.append(replace(r, variant=k % 4) if condition.startswith('paraphrase') else r)
    elif condition == 'selective_duplication':
        rows = base + [base[4]] * (m-1)  # preselected h=1, action=0, sample=0
    else:
        raise ValueError(condition)
    return q, rows

def canonicalize(rows, method):
    if method in ('belief_mixing','provenance'):
        key = lambda r:r.sid
    elif method == 'exact_dedup':
        key = lambda r:r.text
    elif method == 'semantic_dedup':
        key = lambda r:r.semantic_key
    else:
        return list(rows)
    unique = {}
    for r in rows:
        k=key(r)
        if k in unique and method in ('belief_mixing','provenance'):
            previous=unique[k]
            if (r.h,r.action,r.value)!=(previous.h,previous.action,previous.value):
                raise ValueError('conflicting sample identity')
        unique.setdefault(k,r)
    return sorted(unique.values(),key=lambda r:(r.h,r.action,r.sid))

def predict(q, rows, method, strength=2.0):
    rows=canonicalize(rows,method)
    if method == 'source_average':
        # Source/sample identities are available to this baseline, but it uses
        # only group means rather than q-weighted explicit belief mixing.
        unique={r.sid:r for r in rows}
        rows=sorted(unique.values(),key=lambda r:(r.h,r.action,r.sid))
    if method in ('belief_mixing','provenance'):
        values=[]
        for a in (0,1):
            means=[math.fsum(r.value for r in rows if r.action==a and r.h==h)/
                   sum(r.action==a and r.h==h for r in rows) for h in (0,1)]
            values.append((1-q)*means[0]+q*means[1])
        return q,np.array(values)
    if method == 'source_average':
        # The (h, action) source family is averaged once. Copies therefore do
        # not gain weight, while independent samples can refine its mean.
        values=[]
        for a in (0,1):
            means=[math.fsum(r.value for r in rows if r.action==a and r.h==h)/
                   sum(r.action==a and r.h==h for r in rows) for h in (0,1)]
            values.append((1-q)*means[0]+q*means[1])
        return q,np.array(values)
    weights=[1.0 if method=='trajectory_average' else (q if r.h else 1-q) for r in rows]
    mass=math.fsum(w for r,w in zip(rows,weights) if r.h)/math.fsum(weights)
    if method == 'flat':
        # Diagnostic ordinary baseline: it treats extra presented descendants
        # as additional root evidence. At m=1 the estimator equals q; pure
        # duplication then increases confidence even though q is unchanged.
        # Conditional values still use the same q-weighted returns below.
        base_count = 8.0  # two h x two actions x two base samples
        multiplicity_proxy = len(rows) / base_count
        mass = 1.0 / (1.0 + math.exp(-(logit(q) + math.log(strength) * max(0.0, multiplicity_proxy - 1.0))))
        weights=[mass if r.h else 1-mass for r in rows]
    values=[]
    for a in (0,1):
        values.append(math.fsum(w*r.value for r,w in zip(rows,weights) if r.action==a)/
                      math.fsum(w for r,w in zip(rows,weights) if r.action==a))
    return mass,np.array(values)

def metrics(s, true_q, pred_q, values, action=None):
    oracle=(1-true_q)*s['mu'][0]+true_q*s['mu'][1]
    if action is None:
        action=int(np.argmax(values))
    h=int(s['root_correct'])
    clamped=min(max(float(pred_q),1e-12),1-1e-12)
    return dict(root_confidence=float(pred_q), posterior_error=abs(pred_q-true_q),
                action=action, reward=float(s['mu'][h,action]+s['execution_noise'][action]),
                expected_reward=float(oracle[action]),
                realized_expected_reward=float(s['mu'][h,action]),
                decision_regret=float(max(oracle)-oracle[action]),
                value_rmse=float(np.sqrt(np.mean((np.array(values)-oracle)**2))),
                brier=(pred_q-h)**2, nll=-math.log(clamped if h else 1-clamped))

def public_metadata(s):
    return {k:v for k,v in s.items() if k not in ('bank','sensors','mu','execution_noise')}

def public_task(s, condition, m):
    # Explicit allowlist: never serialize hidden truth, mu, or execution RNG.
    n=m if condition=='new_real_evidence' else 1
    return (f'Two routes are actions 0 and 1. H=1 means the designated route has permission; '
            f'H=0 means it does not. The prior probability of H=1 is {s["prior"]:.3f}. '
            f'The real sensor reports {list(map(int,s["sensors"][:n]))}. '
            f'For each independent real report, P(report=1|H=1)={s["strength"]/(1+s["strength"]):.6f} '
            f'and P(report=1|H=0)={1/(1+s["strength"]):.6f}. '
            'The following are simulated conditional returns, not observed executions. '
            'Choose the action with the highest expected return in the real world. ')
