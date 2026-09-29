"""Minimal three-seed DeepSeek duplication experiment.

The prompt exposes only public sensor/task text and conditional rollout text.
Each seed has one fixed bank; flat receives repeated descendants at m=1/2/4/8,
while dedup/grouped readers receive one row per sample. Hidden state is used
only after the call to score the chosen action. This is a model diagnostic,
not evidence that all language models implement the analytic flat estimator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.core import intervention, metrics, predict, public_task, scenario
from submission.llm import JsonLLM, probability, tokens


DEFAULT = Path('results_submission/deepseek_mvp_20260929')
BUDGETS = (1, 2, 4, 8)
METHODS = ('flat', 'trajectory_average', 'simple_dedup', 'source_grouping', 'provenance_value')
SYSTEM = '''You are a small two-action planning evaluator. Return JSON only.
The prior and sensor are real evidence. Conditional rollouts are imagined
consequences and are not new sensor observations. Choose the action with the
highest expected real-world return. Use the supplied action IDs exactly.
Return {"action_id":0,"p_true":0.5,"success_probability":0.5,"reason":"short"}.'''


def public_payload(s, rows, method):
    items = []
    for row in rows:
        items.append({'sample_id': row.sid, 'hypothesis': int(row.h),
                      'action': int(row.action), 'rollout': row.text})
    payload = {'task': public_task(s, 'pure_duplication', 1),
               'actions': [{'action_id': 0}, {'action_id': 1}],
               'imagined_rollouts': items}
    if method == 'source_grouping':
        payload['provenance'] = {'group_key': 'sample_id', 'unique_sample_count': len(items),
                                 'instruction': 'weight each unique source once'}
    return payload


def validate(answer):
    action = int(answer['action_id'])
    if action not in (0, 1):
        raise ValueError('invalid action_id')
    return {'action_id': action, 'p_true': probability(answer['p_true']),
            'success_probability': probability(answer.get('success_probability', 0.5)),
            'reason': str(answer.get('reason', ''))}


def seed_specs():
    return [
        scenario(0, physical_state=0, hypothesis_direction=1, prior=.3, strength=2., family=0),
        scenario(1, physical_state=1, hypothesis_direction=0, prior=.7, strength=2., family=0),
        scenario(2, physical_state=1, hypothesis_direction=1, prior=.5, strength=1.25, family=1),
    ]


def run(out: Path, workers: int = 1):
    del workers  # calls are deliberately sequential for transparent request order
    out.mkdir(parents=True, exist_ok=True)
    request_dir = out / 'requests'
    client = JsonLLM(request_dir, model='deepseek-chat')
    records = []
    scenarios = seed_specs()
    for s in scenarios:
        _, base = intervention(s, 'pure_duplication', 1)
        unique = list(base)
        payload_by_method = {
            'simple_dedup': unique,
            'source_grouping': unique,
        }
        for method in ('flat', 'trajectory_average'):
            for m in BUDGETS:
                payload_by_method[(method, m)] = [r for r in unique for _ in range(m)]
        for method in ('simple_dedup', 'source_grouping'):
            payload_by_method[(method, 1)] = unique
        # one API request per distinct presentation; the other duplication
        # rows for dedup/grouping reuse the byte-identical cached response.
        answers = {}
        for method in ('flat', 'trajectory_average'):
            for m in BUDGETS:
                request_id = f'{s["scenario_id"]}-{method}-m{m}'
                answer, entry = client.ask(request_id, SYSTEM,
                                           public_payload(s, payload_by_method[(method, m)], method), 220)
                answers[(method, m)] = (validate(answer), entry)
        for method in ('simple_dedup', 'source_grouping'):
            request_id = f'{s["scenario_id"]}-{method}-unique'
            answer, entry = client.ask(request_id, SYSTEM,
                                       public_payload(s, payload_by_method[(method, 1)], method), 220)
            answers[(method, 1)] = (validate(answer), entry)
            for m in BUDGETS[1:]:
                answers[(method, m)] = answers[(method, 1)]
        q, _ = intervention(s, 'pure_duplication', 1)
        _, unique_values = predict(q, unique, 'provenance')
        value_action = int(np.argmax(unique_values))
        value_answer = {'action_id': value_action, 'p_true': q,
                        'success_probability': float('nan'), 'reason': 'algorithmic unique-sample mixture'}
        for m in BUDGETS:
            answers[('provenance_value', m)] = (value_answer, {'response': {}, 'parsed': value_answer})
        flat_ref = answers[('flat', 1)][0]['action_id']
        for method in METHODS:
            for m in BUDGETS:
                answer, entry = answers[(method, m)]
                row = metrics(s, q, answer['p_true'], unique_values, action=answer['action_id'])
                records.append({'scenario_id': s['scenario_id'], 'seed': s['seed'],
                                'task_family': s['family'], 'method': method, 'duplication': m,
                                'root_truth': s['root_correct'], 'evidence_q': q,
                                'p_true': answer['p_true'], 'belief_drift': answer['p_true'] - q,
                                'action_id': answer['action_id'],
                                'action_flip_from_flat_m1': int(answer['action_id'] != flat_ref),
                                'reward': row['reward'], 'decision_regret': row['decision_regret'],
                                'brier': (answer['p_true'] - s['root_correct']) ** 2,
                                'tokens': tokens(entry), 'reason': answer['reason']})
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'rows.csv', index=False)
    summary = frame.groupby(['method', 'duplication']).agg(
        n=('seed', 'size'), p_true=('p_true', 'mean'), belief_drift=('belief_drift', 'mean'),
        action_flip_rate=('action_flip_from_flat_m1', 'mean'), reward=('reward', 'mean'),
        decision_regret=('decision_regret', 'mean'), brier=('brier', 'mean'),
        tokens=('tokens', 'mean')).reset_index()
    summary.to_csv(out / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for method in METHODS:
        z = summary[summary.method == method].sort_values('duplication')
        axes[0].plot(z.duplication, z.p_true, marker='o', label=method)
        axes[1].plot(z.duplication, z.reward, marker='o', label=method)
    for ax, ylabel in zip(axes, ('mean p_true', 'realized reward')):
        ax.set_xscale('log', base=2); ax.set_xlabel('duplication'); ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=7, loc='best'); fig.tight_layout()
    fig.savefig(out / 'curve.png', dpi=180); plt.close(fig)
    # Paired seed-level contrasts and deterministic cluster bootstrap intervals.
    paired_rows = []
    for (seed, m), group in frame.groupby(['seed', 'duplication']):
        wide = group.set_index('method')
        for metric in ('p_true', 'belief_drift', 'action_flip_from_flat_m1', 'reward', 'decision_regret', 'brier'):
            if 'flat' in wide.index and 'provenance_value' in wide.index:
                paired_rows.append({'seed': int(seed), 'duplication': int(m), 'metric': metric,
                                    'provenance_minus_flat': float(wide.loc['provenance_value', metric] - wide.loc['flat', metric])})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / 'paired.csv', index=False)
    bootstrap_rows = []
    rng = np.random.default_rng(20260929)
    for (m, metric), group in paired.groupby(['duplication', 'metric']):
        values = group.provenance_minus_flat.to_numpy(float)
        draws = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
        bootstrap_rows.append({'duplication': int(m), 'metric': metric, 'n_seeds': len(values),
                               'estimate': float(values.mean()), 'ci95_low': float(np.quantile(draws, .025)),
                               'ci95_high': float(np.quantile(draws, .975)), 'bootstrap_draws': 20000})
    pd.DataFrame(bootstrap_rows).to_csv(out / 'bootstrap.csv', index=False)
    controls = []
    for s in scenarios:
        for m in BUDGETS:
            left = json.loads((request_dir / f'{s["scenario_id"]}-flat-m{m}.json').read_text())
            right = json.loads((request_dir / f'{s["scenario_id"]}-trajectory_average-m{m}.json').read_text())
            controls.append({'seed': s['seed'], 'duplication': m,
                             'byte_identical_request_body': left['request'] == right['request'],
                             'flat_action': left.get('parsed', {}).get('action_id'),
                             'trajectory_average_action': right.get('parsed', {}).get('action_id'),
                             'action_disagreement': int(left.get('parsed', {}).get('action_id') != right.get('parsed', {}).get('action_id')),
                             'flat_p_true': left.get('parsed', {}).get('p_true'),
                             'trajectory_average_p_true': right.get('parsed', {}).get('p_true')})
    pd.DataFrame(controls).to_csv(out / 'same_body_control.csv', index=False)
    request_files = list(request_dir.glob('*.json'))
    metadata = {'protocol': 'deepseek-mvp-fixed-bank-v1', 'model': 'deepseek-chat',
                'seeds': [0, 1, 2], 'duplications': list(BUDGETS), 'methods': list(METHODS),
                'api_network_calls_this_run': int(client.network_calls),
                'archived_request_files': len(request_files), 'valid_rows': len(frame),
                'paired_rows': len(paired), 'bootstrap_draws': 20000,
                'same_body_control_rows': len(controls),
                'same_body_control_all_identical': bool(all(x['byte_identical_request_body'] for x in controls)),
                'hidden_state_used_in_prompt': False,
                'interpretation': 'model-specific diagnostic; no universal LLM claim',
                'provenance_value': 'algorithmic unique-sample conditional-value mixture',
                'api_seed_supported': False}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get('DEEPSEEK_API_KEY') and not (args.output / 'requests').exists():
        raise SystemExit('DEEPSEEK_API_KEY required for uncached calls')
    run(args.output, args.workers)


if __name__ == '__main__':
    main()
