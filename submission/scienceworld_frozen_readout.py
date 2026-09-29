"""Fixed-bank intervention and identical-input API control on ScienceWorld.

prepare freezes states before any calls. run makes only readout calls; replay
executes all four candidate actions from each prefix, without consulting their
outcomes during selection. summarize reports immediate outcomes, not episodes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from submission.llm import JsonLLM, probability, tokens
from submission.recharged_llm_extension_summary import load_rows
from submission.scienceworld_runner import (
    READOUT_SYSTEM, canonical_bytes, unique_provenance_bank, provenance_decision,
)

DEFAULT = Path('results_submission/scienceworld_frozen_readout_clean_20260929')
BUDGETS = (1, 2, 4, 8)
VARIANTS = ('m1', 'm1_repeat', 'm2', 'm4', 'm8')


def archived_pre_action_visible(trace_path, state):
    """Use the serialized API input, never the old mutable-history trace view."""
    request_dir = Path(trace_path).parent.parent / 'requests'
    sid = state['snapshot_id']
    candidate_entry = json.loads((request_dir / f'{sid}-candidates.json').read_text())
    visible = json.loads(candidate_entry['request']['messages'][1]['content'])
    step = state['outcome']['step']
    if len(visible['history']) != step or any(h['step'] >= step for h in visible['history']):
        raise ValueError('future/current outcome in candidate API input')
    for suffix in ('bank', 'flat-m1'):
        entry = json.loads((request_dir / f'{sid}-{suffix}.json').read_text())
        payload = json.loads(entry['request']['messages'][1]['content'])
        if payload['evidence'] != visible:
            raise ValueError('candidate/bank/readout evidence mismatch')
    return visible


def payloads(state):
    unique = unique_provenance_bank(state['bank'])
    if len(unique) != 8:
        raise ValueError('expected eight frozen conditional samples')
    random.Random(431).shuffle(unique)
    base = {'evidence': state['visible'], 'premise': state['candidate']['premise'],
            'evidence_only_probability': state['candidate']['p_true'],
            'candidates': state['candidate']['actions']}
    result = {}
    for m in BUDGETS:
        bank = [copy.deepcopy(row) for row in unique
                for _ in range(m if row['hypothesis_value'] else 1)]
        result[f'm{m}'] = {**copy.deepcopy(base), 'imagined_rollouts': bank}
        if unique_provenance_bank(bank) != unique_provenance_bank(unique):
            raise ValueError('intervention changed sample content')
    result['m1_repeat'] = copy.deepcopy(result['m1'])
    return result


def prepare(out):
    states = []
    for task in ('find-plant', 'find-animal'):
        frame = load_rows(task)
        for variation in range(156, 162):
            row = frame[(frame.variation == variation) & (frame.duplication == 1)
                        & (frame.experiment_method == 'flat')].iloc[0]
            trace = [json.loads(line) for line in Path(row.trace_file).read_text().splitlines()]
            # Outcome-independent inclusion: step 1 for every contiguous task.
            state = trace[1]
            visible = archived_pre_action_visible(row.trace_file, state)
            states.append({'state_id': f'{task}-v{variation}-s1', 'task': task,
                           'variation': variation, 'step': 1, 'trace_file': row.trace_file,
                           'simplification': row.simplification, 'step_limit': int(row.step_limit),
                           'prefix': [trace[0]['action']], 'visible': visible,
                           'input_source': 'serialized original candidate/bank/flat API request, equality checked',
                           'candidate': state['candidate'],
                           'bank': unique_provenance_bank(state['bank'])})
    manifest = {
        'protocol': 'frozen-bank-identical-prompt-v2-pre-action', 'states': len(states),
        'input_integrity': 'history contains only steps strictly before the frozen decision; original candidate/bank/readout inputs agree',
        'selection': 'step 1 of each archived flat m1 extension, both tasks, variations 156..161; no outcome filter',
        'duplications': list(BUDGETS), 'api_repetitions': 3, 'variants': list(VARIANTS),
        'planned_api_calls': len(states) * 3 * len(VARIANTS),
        'model': 'deepseek-chat', 'temperature': 0, 'system_prompt': READOUT_SYSTEM,
        'api_seed_supported': False, 'schedule_seed': 20260929,
        'control': 'm1_repeat request has byte-identical body to m1; independent request ID',
        'methods': ['flat', 'same_prompt_dedup', 'provenance_value'],
        'dedup_readout': 'use the independent m1_repeat response at each m; structural input invariance',
        'provenance_value': 'deterministic unique-sample belief mixture; no LLM belief override treated as model behavior',
        'outcome': 'immediate simulator reward and best-of-four immediate regret, not episode reward/success',
        'root_truth_labels': 'unavailable; report belief drift, not root calibration or wrong-belief rate',
        'known_limit': 'previously studied dev tasks; exploratory, 12 states; API stochasticity remains',
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('manifest.json', manifest), ('states.json', states)]:
        path = out / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'frozen artifact differs: {path}')
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    print(json.dumps({'states': len(states), 'planned_api_calls': manifest['planned_api_calls']}))


def run_calls(out, workers, retry_failed=False):
    states = json.loads((out / 'states.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    failed = set()
    if retry_failed and (out / 'readouts.json').exists():
        for record in json.loads((out / 'readouts.json').read_text()):
            if record['error'] is not None:
                failed.add((record['state_id'], record['rep'], record['variant']))
    jobs = [(state, rep, variant) for state in states for rep in range(3) for variant in VARIANTS
            if not retry_failed or (state['state_id'], rep, variant) in failed]
    random.Random(manifest['schedule_seed']).shuffle(jobs)

    def call(job):
        state, rep, variant = job
        request_id = f'{state["state_id"]}-r{rep}-{variant}' + ('-retry1' if retry_failed else '')
        client = JsonLLM(out / 'requests', model=manifest['model'])
        record = {'state_id': state['state_id'], 'task': state['task'],
                  'variation': state['variation'], 'rep': rep, 'variant': variant,
                  'request_id': request_id, 'error': None}
        try:
            answer, entry = client.ask(request_id, manifest['system_prompt'], payloads(state)[variant], 450)
            if type(answer['action_id']) is not int or answer['action_id'] not in range(4):
                raise ValueError('invalid action ID')
            record.update(action_id=answer['action_id'], p_true=probability(answer['p_true']),
                          success_probability=probability(answer['success_probability']),
                          tokens=tokens(entry), http_status=entry.get('http_status'),
                          reason=answer.get('reason', ''))
        except Exception as exc:
            record['error'] = type(exc).__name__
        record['network_calls_this_run'] = client.network_calls
        return record

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(call, jobs):
            records.append(record)
            if len(records) % 12 == 0:
                print(json.dumps({'completed': len(records), 'planned': len(jobs),
                                  'errors': sum(r['error'] is not None for r in records)}), flush=True)
    if retry_failed and (out / 'readouts.json').exists():
        previous = json.loads((out / 'readouts.json').read_text())
        key = {(r['state_id'], r['rep'], r['variant']): r for r in records}
        records = [key.get((r['state_id'], r['rep'], r['variant']), r) if r['error'] is not None else r for r in previous]
    (out / 'readouts.json').write_text(json.dumps(records, indent=2))
    print('Readout collection complete', len(records), flush=True)


def replay(out):
    from scienceworld import ScienceWorldEnv
    states = json.loads((out / 'states.json').read_text())
    path = out / 'candidate_outcomes.json'
    outcomes = json.loads(path.read_text()) if path.exists() else []
    done_keys = {(r['state_id'], r['action_id']) for r in outcomes}
    env = ScienceWorldEnv(envStepLimit=20)
    try:
        for state in states:
            for candidate in state['candidate']['actions']:
                if (state['state_id'], candidate['id']) in done_keys:
                    continue
                env.load(state['task'], state['variation'], state['simplification'], generateGoldPath=False)
                obs, info = env.reset()
                for action in state['prefix']:
                    obs, _, done, info = env.step(action)
                    if done:
                        raise ValueError('prefix terminated')
                visible = state['visible']
                matched = (obs == visible['observation'] and info['look'] == visible['look']
                           and info['inv'] == visible['inventory']
                           and info['score'] == visible['observed_score'])
                if not matched:
                    raise ValueError(f'public state replay mismatch: {state["state_id"]}')
                before = float(info['score'])
                obs, reward, terminal, info = env.step(candidate['action'])
                outcomes.append({'state_id': state['state_id'], 'task': state['task'],
                                 'variation': state['variation'], 'action_id': candidate['id'],
                                 'action': candidate['action'], 'public_state_match': matched,
                                 'score_before': before, 'score_after': float(info['score']),
                                 'immediate_reward': float(reward), 'terminal': bool(terminal),
                                 'observation_after': obs})
                path.write_text(json.dumps(outcomes, indent=2))
            print('Replayed', state['state_id'], flush=True)
    finally:
        env.close()


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260929)
    means = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def summarize(out):
    states = json.loads((out / 'states.json').read_text())
    raw = json.loads((out / 'readouts.json').read_text())
    good = {(r['state_id'], r['rep'], r['variant']): r for r in raw if r['error'] is None}
    outcomes = pd.DataFrame(json.loads((out / 'candidate_outcomes.json').read_text()))
    outcome_map = outcomes.set_index(['state_id', 'action_id']).immediate_reward.to_dict()
    best_reward = outcomes.groupby('state_id').immediate_reward.max().to_dict()
    rows, paired = [], []
    for state in states:
        sid, q = state['state_id'], state['candidate']['p_true']
        algorithm = provenance_decision(state['candidate'], state['bank'])
        algorithm_success = provenance_decision(state['candidate'], state['bank'], objective='success')
        for rep in range(3):
            if any((sid, rep, v) not in good for v in VARIANTS):
                continue  # Failures remain in raw; comparisons require complete blocks.
            base, repeat = good[sid, rep, 'm1'], good[sid, rep, 'm1_repeat']
            noise_flip = float(base['action_id'] != repeat['action_id'])
            noise_drift = abs(repeat['p_true'] - base['p_true'])
            for m in BUDGETS:
                answer = good[sid, rep, f'm{m}']
                for method, result, reference in [('flat', answer, base),
                                                  ('same_prompt_dedup', repeat, repeat),
                                                  ('provenance_value', algorithm, algorithm),
                                                  ('provenance_success', algorithm_success, algorithm_success)]:
                    reward = outcome_map[sid, result['action_id']]
                    rows.append({'state_id': sid, 'task': state['task'], 'rep': rep,
                                 'duplication': m, 'method': method,
                                 'root_p_true': result['p_true'], 'belief_drift_from_evidence': result['p_true'] - q,
                                 'action_flip_from_m1': int(result['action_id'] != reference['action_id']),
                                 'immediate_reward': reward, 'immediate_candidate_regret': best_reward[sid] - reward})
                paired.append({'state_id': sid, 'task': state['task'], 'rep': rep, 'duplication': m,
                               'confidence_delta': answer['p_true'] - base['p_true'],
                               'absolute_confidence_change': abs(answer['p_true'] - base['p_true']),
                               'action_flip': int(answer['action_id'] != base['action_id']),
                               'identical_input_action_flip': noise_flip,
                               'identical_input_absolute_confidence_change': noise_drift,
                               'flip_excess_over_repeat': int(answer['action_id'] != base['action_id']) - noise_flip,
                               'absolute_drift_excess_over_repeat': abs(answer['p_true'] - base['p_true']) - noise_drift,
                               'reward_change_from_m1': outcome_map[sid, answer['action_id']] - outcome_map[sid, base['action_id']],
                               'dedup_minus_flat_immediate_reward': outcome_map[sid, repeat['action_id']] - outcome_map[sid, answer['action_id']],
                               'provenance_minus_flat_immediate_reward': outcome_map[sid, algorithm['action_id']] - outcome_map[sid, answer['action_id']]})
    frame, contrasts = pd.DataFrame(rows), pd.DataFrame(paired)
    frame.to_csv(out / 'decisions.csv', index=False)
    contrasts.to_csv(out / 'paired.csv', index=False)
    metrics = ['root_p_true', 'belief_drift_from_evidence', 'action_flip_from_m1', 'immediate_reward', 'immediate_candidate_regret']
    frame.groupby(['task', 'method', 'duplication'])[metrics].mean().reset_index().to_csv(out / 'summary.csv', index=False)
    summary = frame.groupby(['task', 'method', 'duplication'])[metrics].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colors = {'flat': '#c43c39', 'same_prompt_dedup': '#7a5aa6',
              'provenance_value': '#276fbf', 'provenance_success': '#4f9d69'}
    labels = [('belief_drift_from_evidence', 'belief drift'),
              ('action_flip_from_m1', 'action flip'),
              ('immediate_reward', 'immediate reward')]
    for method in ('flat', 'same_prompt_dedup', 'provenance_value', 'provenance_success'):
        z = summary[summary.method == method].groupby('duplication', as_index=False).mean(numeric_only=True)
        if z.empty:
            continue
        for ax, (metric, label) in zip(axes, labels):
            ax.plot(z.duplication, z[metric], 'o-', label=method, color=colors[method])
            ax.set_xscale('log', base=2); ax.set_xlabel('duplication'); ax.set_ylabel(label)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'curve.png', dpi=180); plt.close(fig)
    boot = []
    for task, group in [('all', contrasts), *list(contrasts.groupby('task'))]:
        for m, block in group.groupby('duplication'):
            for metric in contrasts.columns.difference(['state_id', 'task', 'rep', 'duplication']):
                units = block.groupby('state_id')[metric].mean()
                estimate, low, high = interval(units.values)
                boot.append({'task': task, 'duplication': m, 'metric': metric, 'n_states': len(units),
                             'estimate': estimate, 'ci95_low': low, 'ci95_high': high})
    pd.DataFrame(boot).to_csv(out / 'bootstrap.csv', index=False)
    identical_blocks = 0
    for state in states:
        for rep in range(3):
            m1 = out / 'requests' / f'{state["state_id"]}-r{rep}-m1.json'
            repeat = out / 'requests' / f'{state["state_id"]}-r{rep}-m1_repeat.json'
            if m1.exists() and repeat.exists():
                left = json.loads(m1.read_text()).get('request')
                right = json.loads(repeat.read_text()).get('request')
                identical_blocks += int(left == right)
    audit = {'planned_api_calls': len(states) * 3 * 5, 'archived_readouts': len(raw),
             'valid_readouts': len(good), 'errors': sum(r['error'] is not None for r in raw),
             'complete_state_repetition_blocks': int(len(contrasts) / 4),
             'identical_input_request_blocks': identical_blocks,
             'states': len(states), 'candidate_replays': len(outcomes),
             'all_replays_match_public_state': bool(outcomes.public_state_match.all()),
             'raw_llm_probabilities_overwritten': False,
             'episode_success_claim': False, 'root_calibration_claim': False}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'retry', 'replay', 'summarize'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage in ('run', 'retry'):
        run_calls(args.output, args.workers, retry_failed=args.stage == 'retry')
    else:
        {'prepare': prepare, 'replay': replay, 'summarize': summarize}[args.stage](args.output)


if __name__ == '__main__':
    main()
