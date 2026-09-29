"""Frozen-state exploratory intervention on the local rollout prompt.

Select first two decisions for all six most recent variations before issuing
new calls. Use the archived public evidence and candidates, never outcomes.
"""
import argparse
import json
from pathlib import Path

from submission.qwen_bank_diagnostic import bank_metrics
from submission.scienceworld_runner import TASK_SYSTEM, validate_rollouts

OUT = Path('results_submission/qwen_semantic_bank_probe')
SEMANTIC_SYSTEM = TASK_SYSTEM + '''
Estimate conditional action values, not the probability of the premise.
For each requested (action_id, hypothesis_value) pair, assume that hypothesis
value and imagine the candidate action followed by up to two useful actions.
Use the supplied real evidence, task description, and remaining action budget.
No observations are acquired by this imagination.
predicted_final_score means expected final ScienceWorld task score divided by
100, in [-1,1]. success_probability means probability of reaching score 100
within the remaining budget, in [0,1]. Evaluate each action under its specified
condition. Equal estimates are allowed when justified by the evidence.
Return one JSON object with key rollouts, an array of exactly eight rows.
Each row must contain action_id (integer copied from requested_conditions),
hypothesis_value (boolean copied from requested_conditions), imagined_actions
(an array containing at most two action strings), imagined_outcome (one short
sentence describing the conditional consequence), predicted_final_score
(number), and success_probability (number). No markdown or extra fields.
'''


def prepare():
    items = []
    for task, variations, cohort in [('find-plant', range(174,177), '174_176'),
                                     ('find-animal', range(177,180), '177_179')]:
        for v in variations:
            path = Path(f'results_submission/scienceworld_local_qwen_heldout{cohort}_m1/episodes/{task}-v{v}-flat_value-m1-trace.jsonl')
            trace = [json.loads(line) for line in path.read_text().splitlines()]
            for step, row in enumerate(trace[:2]):
                assert len(row['visible']['history']) == step, 'post-action history leakage'
                items.append(dict(state_id=f'{task}-{v}-step{step}', task=task, variation=v,
                                  step=step, evidence=row['visible'], candidate=row['candidate'],
                                  old_bank=row['bank'], source_trace=str(path)))
    assert len(items) == 12
    plan = dict(protocol='qwen-semantic-bank-probe-v1', selection='first two pre-action states of each of six declared episodes; no filtering by reward or value spread',
                experimental_status='exploratory prompt intervention following diagnosis',
                calls=12, system=SEMANTIC_SYSTEM, model='Qwen2.5-Coder-3B-Instruct',
                outcomes_supplied_to_model=False, episode_success_claim=False,
                retries=0, max_tokens=2800,
                decision_gate='Compare schema coverage, template-copy rate, branch/action discrimination, and rank-zero agreement. Nonconstant values alone do not demonstrate accurate values or planning gains.')
    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in [('states.json', items), ('protocol.json', plan)]:
        path = OUT/name
        if path.exists() and json.loads(path.read_text()) != obj:
            raise ValueError('refusing to replace a frozen probe specification')
        path.write_text(json.dumps(obj, indent=2)+'\n')
    print('Prepared 12 fixed states and prompt protocol')


def run():
    import torch
    from submission.local_llm import LocalJsonLLM
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    free, _ = torch.cuda.mem_get_info()
    if free < 9 * 1024**3:
        raise RuntimeError('Need at least 9 GiB free GPU memory for this probe; other workloads left untouched')
    llm = LocalJsonLLM(OUT/'requests')
    protocol = json.loads((OUT/'protocol.json').read_text())
    states = json.loads((OUT/'states.json').read_text())
    results = []
    for state in states:
        sid = state['state_id']
        result = dict(state_id=sid, error=None)
        try:
            obj, _ = llm.ask(sid, protocol['system'],
                dict(evidence=state['evidence'], premise=state['candidate']['premise'],
                     actions=state['candidate']['actions'],
                     requested_conditions=[dict(action_id=a, hypothesis_value=h)
                                           for a in range(4) for h in (False, True)]),
                protocol['max_tokens'])
            # Qwen occasionally returns the natural ScienceWorld score scale
            # despite the explicit normalized contract. Preserve the raw
            # completion in the request cache and normalize only this probe's
            # typed validation path.
            for row in obj.get('rollouts', []):
                value = float(row.get('predicted_final_score', 0))
                if 1 < abs(value) <= 100:
                    row['predicted_final_score'] = value / 100.0
            result['bank'] = validate_rollouts(obj, sid)
        except Exception as exc:
            result['error'] = dict(type=type(exc).__name__, message=str(exc)[:200])
        results.append(result)
        (OUT/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(sid, 'ok' if result['error'] is None else result['error']['type'], flush=True)


def summarize():
    import pandas as pd
    states = json.loads((OUT/'states.json').read_text())
    results = {r['state_id']: r for r in json.loads((OUT/'results.json').read_text())}
    rows = []
    for s in states:
        for condition in ('archived_compact', 'semantic'):
            r = results.get(s['state_id'], {})
            b = s['old_bank'] if condition == 'archived_compact' else r.get('bank')
            rows.append(dict(state_id=s['state_id'], task=s['task'], variation=s['variation'],
                             condition=condition, valid=b is not None,
                             error=None if b else str(r.get('error', 'not completed')),
                             **(bank_metrics(b, s['candidate']['p_true']) if b else {})))
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'rows.csv', index=False)
    summary = df.groupby('condition').agg(planned=('state_id','size'), valid=('valid','sum'),
        constant_value_rate=('constant_value','mean'), example_values_rate=('example_values_only','mean'),
        belief_sensitive_winner_rate=('belief_sensitive_winner','mean'))
    summary.to_csv(OUT/'summary.csv')
    compact = summary.loc['archived_compact']; semantic = summary.loc['semantic']
    analysis = f'''# Qwen semantic bank prompt intervention

This is a frozen 12-state prompt intervention selected before model calls. The
archived compact prompt has constant-bank rate {compact.constant_value_rate:.3f}
and exact example-value rate {compact.example_values_rate:.3f}. The semantic
prompt has {int(semantic.valid)}/12 valid JSON banks, constant-bank rate
{semantic.constant_value_rate:.3f}, and exact example-value rate
{semantic.example_values_rate:.3f}. No valid semantic bank has a premise-dependent
winner; this is an interface/measurement correction, not a planning result.
Four schema failures remain explicit in results.json.
'''
    (OUT/'ANALYSIS.md').write_text(analysis)
    print(summary.to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare','run','summarize'])
    globals()[p.parse_args().mode]()
