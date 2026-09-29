"""Raw-input verification, episode-cluster uncertainty, and joint event audit.

Read-only with respect to archived inputs and API responses. The four public
states of each task/variation are correlated; primary intervals resample 24
whole episodes within task family, after averaging repetitions and steps.
This analysis is post hoc and cannot turn the adaptive expansion into a
preregistered confirmatory study.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from submission.scienceworld_frozen_readout import archived_pre_action_visible, payloads
from submission.scienceworld_runner import provenance_decision, unique_provenance_bank

DEFAULT = Path('results_submission/scienceworld_frozen_readout_first4_20260929')
METRICS = (
    'confidence_delta', 'absolute_confidence_change', 'action_flip',
    'repeat_action_flip', 'flip_excess_over_repeat',
    'absolute_drift_excess_over_repeat', 'reward_change_from_m1',
    'dedup_minus_flat_reward', 'provenance_value_minus_flat_reward',
    'provenance_success_minus_flat_reward', 'joint_confidence_and_action_change',
    'harmful_action_flip',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def joint_event(before, after, tolerance=1e-10):
    return (abs(after['p_true'] - before['p_true']) > tolerance,
            after['action_id'] != before['action_id'])


def episode_intervals(frame, metrics=METRICS, draws=20000, seed=20260929):
    """Equal episode weight; stratified bootstrap keeps task mix fixed."""
    units = frame.groupby(['task', 'variation', 'duplication'])[list(metrics)].mean().reset_index()
    records = []
    for scope in ['all', *sorted(units.task.unique())]:
        selected = units if scope == 'all' else units[units.task == scope]
        for duplication, block in selected.groupby('duplication'):
            rng = np.random.default_rng(seed)
            means = np.zeros((draws, len(metrics)))
            for _, stratum in block.groupby('task'):
                values = stratum[list(metrics)].to_numpy(float)
                indices = rng.integers(0, len(values), size=(draws, len(values)))
                means += values[indices].mean(axis=1) * len(values) / len(block)
            for k, metric in enumerate(metrics):
                records.append({'task': scope, 'duplication': int(duplication), 'metric': metric,
                                'n_episodes': len(block), 'estimate': float(block[metric].mean()),
                                'ci95_low': float(np.quantile(means[:, k], .025)),
                                'ci95_high': float(np.quantile(means[:, k], .975)),
                                'bootstrap_draws': draws,
                                'resampling_unit': 'whole task/variation episode, within task family'})
    return units, pd.DataFrame(records)


def validate_raw(directory):
    states, manifest = read(directory/'states.json'), read(directory/'manifest.json')
    records, outcomes = read(directory/'readouts.json'), read(directory/'candidate_outcomes.json')
    expected_states = {(task, v, step) for task in ('find-plant', 'find-animal')
                       for v in range(150, 162) for step in range(4)}
    require(len(states) == 96 and {(s['task'], s['variation'], s['step']) for s in states} == expected_states,
            'unexpected state coverage')
    state_map = {s['state_id']: s for s in states}
    keys = {(s['state_id'], rep, variant) for s in states for rep in range(3)
            for variant in ('m1', 'm1_repeat', 'm2', 'm4', 'm8')}
    by_key = {(r['state_id'], r['rep'], r['variant']): r for r in records}
    require(len(records) == len(by_key) == len(keys) and set(by_key) == keys,
            'missing or duplicated readout blocks')
    reward_map = {(r['state_id'], r['action_id']): r for r in outcomes}
    require(len(outcomes) == len(reward_map) == 384, 'missing or repeated candidate replay')
    actual_requests = set((directory/'requests').glob('*.json'))
    expected_requests = {directory/'requests'/f"{r['request_id']}.json" for r in records}
    require(actual_requests == expected_requests, 'request archive has unexpected coverage')
    tokens, response_models, timestamps, same_input = 0, Counter(), [], 0
    for state in states:
        sid = state['state_id']
        trace = [json.loads(line) for line in Path(state['trace_file']).read_text().splitlines()]
        original = trace[state['step']]
        visible = archived_pre_action_visible(state['trace_file'], original)
        require(visible == state['visible'], f'{sid}: source-visible mismatch')
        old_request = Path(state['trace_file']).parent.parent/'requests'/f"{original['snapshot_id']}-flat-m1.json"
        old_payload = json.loads(read(old_request)['request']['messages'][1]['content'])
        require(old_payload['candidates'] == state['candidate']['actions'], f'{sid}: candidate mismatch')
        require(old_payload['premise'] == state['candidate']['premise'] and
                old_payload['evidence_only_probability'] == state['candidate']['p_true'], f'{sid}: premise mismatch')
        require(unique_provenance_bank(old_payload['imagined_rollouts']) == state['bank'], f'{sid}: bank mismatch')
        require([x['action'] for x in trace[:state['step']]] == state['prefix'], f'{sid}: prefix mismatch')
        for candidate in state['candidate']['actions']:
            outcome = reward_map[sid, candidate['id']]
            require(outcome['public_state_match'] is True and outcome['action'] == candidate['action'],
                    f'{sid}: replay mismatch')
        variants = payloads(state)
        for rep in range(3):
            bodies = {}
            for variant in manifest['variants']:
                row = by_key[sid, rep, variant]
                entry = read(directory/'requests'/f"{row['request_id']}.json")
                require(entry['ok'] is True and entry['http_status'] == 200 and row['error'] is None,
                        f'{sid}: failed request')
                expected_body = {'model': manifest['model'], 'temperature': 0,
                                 'messages': [{'role': 'system', 'content': manifest['system_prompt']},
                                              {'role': 'user', 'content': json.dumps(variants[variant], ensure_ascii=False)}],
                                 'max_tokens': 450, 'response_format': {'type': 'json_object'}}
                require(entry['request'] == expected_body, f'{sid}: serialized request differs')
                parsed = json.loads(entry['response']['choices'][0]['message']['content'])
                require(entry['parsed'] == parsed, f'{sid}: response parsing differs')
                for field in ('p_true', 'action_id', 'success_probability'):
                    require(row[field] == parsed[field], f'{sid}: readout differs from response')
                bodies[variant] = entry['request']
                tokens += entry['response']['usage']['total_tokens']
                response_models[entry['response'].get('model', 'unknown')] += 1
                timestamps.append(entry['utc'])
            same_input += int(bodies['m1'] == bodies['m1_repeat'])
    require(same_input == 288, 'identical-input control differs')
    audit = {'raw_inputs_verified': True, 'requests_verified': len(keys),
             'source_states_verified': len(states), 'episodes': 24, 'candidate_replays_verified': 384,
             'identical_request_blocks_verified': same_input, 'total_api_tokens': tokens,
             'response_models': dict(response_models), 'first_request_utc': min(timestamps),
             'last_request_utc': max(timestamps), 'new_api_calls': 0}
    return state_map, by_key, reward_map, audit


def observations(states, raw, outcomes):
    rows = []
    for sid, state in states.items():
        value = provenance_decision(state['candidate'], state['bank'])
        success = provenance_decision(state['candidate'], state['bank'], objective='success')
        def reward(answer):
            return outcomes[sid, answer['action_id']]['immediate_reward']
        def action(answer):
            return next(c['action'] for c in state['candidate']['actions'] if c['id'] == answer['action_id'])
        for rep in range(3):
            base, repeat = raw[sid, rep, 'm1'], raw[sid, rep, 'm1_repeat']
            for m in (1, 2, 4, 8):
                answer = raw[sid, rep, f'm{m}']
                changed, flipped = joint_event(base, answer)
                repeat_changed, repeat_flipped = joint_event(base, repeat)
                delta_reward = reward(answer) - reward(base)
                rows.append({'state_id': sid, 'task': state['task'], 'variation': state['variation'],
                             'step': state['step'], 'rep': rep, 'duplication': m,
                             'p_m1': base['p_true'], 'p_m': answer['p_true'],
                             'action_m1': action(base), 'action_m': action(answer),
                             'provenance_action': action(value), 'reward_m1': reward(base), 'reward_m': reward(answer),
                             'confidence_delta': answer['p_true'] - base['p_true'],
                             'absolute_confidence_change': abs(answer['p_true'] - base['p_true']),
                             'confidence_changed': int(changed), 'action_flip': int(flipped),
                             'repeat_action_flip': int(repeat_flipped), 'repeat_confidence_changed': int(repeat_changed),
                             'flip_excess_over_repeat': int(flipped) - int(repeat_flipped),
                             'absolute_drift_excess_over_repeat': abs(answer['p_true'] - base['p_true']) - abs(repeat['p_true'] - base['p_true']),
                             'reward_change_from_m1': delta_reward,
                             'dedup_minus_flat_reward': reward(repeat) - reward(answer),
                             'provenance_value_minus_flat_reward': reward(value) - reward(answer),
                             'provenance_success_minus_flat_reward': reward(success) - reward(answer),
                             'joint_confidence_and_action_change': int(changed and flipped),
                             'harmful_action_flip': int(flipped and delta_reward < 0)})
    return pd.DataFrame(rows)


def report(directory, frame, boot, meta):
    def estimate(metric, task='all', dup=8):
        r = boot[(boot.task == task) & (boot.duplication == dup) & (boot.metric == metric)].iloc[0]
        return f"{r.estimate:+.5f} [{r.ci95_low:+.5f}, {r.ci95_high:+.5f}]"
    lines = [
        '### Expanded fixed-bank ScienceWorld analysis (episode-cluster audit)', '',
        'This adaptive exploratory extension uses the first four pre-action states from each of 24 archived '
        'task/variation episodes (96 states, two task families). All 1,440 serialized requests and parsed '
        'responses were independently matched to the frozen inputs and raw provider outputs; 384 candidate '
        'replays matched the public simulator state. Multiplicity changes only the true-premise descendants '
        '(eight unique samples; presented rows at m=1/2/4/8 are 8/12/20/36). It also changes context length '
        'and repetition pattern; it is not a pure manipulation of internal belief.', '',
        'Primary intervals below resample 24 whole episodes within task family using 20,000 draws, after '
        'averaging the three API repetitions and four steps in each episode. The earlier `bootstrap.csv` '
        'resamples states and is retained as a secondary diagnostic. These are pointwise exploratory intervals.', '',
        '| Contrast at duplication 8 | Estimate [episode-cluster 95% interval] |',
        '|---|---:|',
    ]
    for metric, label in [('confidence_delta', 'Flat reported P(H=true) change'),
                          ('action_flip', 'Flat action flip rate'),
                          ('repeat_action_flip', 'Byte-identical input action flip rate'),
                          ('flip_excess_over_repeat', 'Flip rate excess over repeat control'),
                          ('reward_change_from_m1', 'Flat immediate reward change'),
                          ('dedup_minus_flat_reward', 'Same-prompt dedup minus flat reward'),
                          ('provenance_value_minus_flat_reward', 'Provenance-value minus flat reward'),
                          ('provenance_success_minus_flat_reward', 'Success-objective minus flat reward (post hoc)')]:
        lines.append(f'| {label} | {estimate(metric)} |')
    lines += ['',
        f"Across the 864 m=2/4/8 paired comparisons, {meta['action_flip_events']} actions changed and "
        f"{meta['confidence_change_events']} reported probabilities changed, with "
        f"{meta['joint_change_events']} joint changes. All {meta['harmful_flip_events']} harmful immediate-reward "
        'flips occurred without a reported probability change. These counts include correlated repetitions, '
        'budgets, and states; they are descriptive, not independent sample sizes. The data do not demonstrate '
        'that confidence inflation mediated action changes.', '',
        'The deterministic provenance algorithms and cached same-prompt dedup readout are invariant by '
        'construction. Dedup reuses one independent m=1 response across budgets, so its zero flip rate is '
        'not a new stochastic robustness measurement. Provenance-value does not show a consistent reward '
        'advantage over flat or simple dedup. The success-objective analysis was added after outcomes were '
        'inspected and is explicitly post hoc. Root truth is unavailable, so reported belief drift is not '
        'a calibration or overconfidence estimate. Immediate reward is not final episode reward.', '',
        'The state selection rule was fixed for this collection, but this expansion followed inspection '
        'of earlier null results; there is no externally registered confirmatory protocol. The results '
        'remain task-family-specific exploratory diagnostics, not evidence of broad closed-loop planning improvement.', '',
        'Files: `episode_bootstrap.csv`, `episode_units.csv`, `paired_events.csv`, `joint_events.csv`, '
        '`verified_analysis.json`, `episode_curves.png`, and `qualitative_events.md` in '
        '`results_submission/scienceworld_frozen_readout_first4_20260929/`.', '',
    ]
    (directory/'ANALYSIS.md').write_text('\n'.join(lines))
    events = frame[(frame.duplication > 1) & ((frame.action_flip == 1) | (frame.confidence_changed == 1))]
    event_lines = ['# Complete list of changed readouts', '',
                   'All changed comparisons are listed, including null-reward and confidence-only changes. '
                   'Repeated rows are repeated API/budget observations, not independent episodes.', '',
                   '| State | Rep | m | P(H) before → after | Action before → after | Immediate reward before → after |',
                   '|---|---:|---:|---|---|---|']
    for r in events.itertuples():
        event_lines.append(f'| {r.state_id} | {r.rep} | {r.duplication} | {r.p_m1:.3f} → {r.p_m:.3f} | '
                           f'{r.action_m1} → {r.action_m} | {r.reward_m1:g} → {r.reward_m:g} |')
    (directory/'qualitative_events.md').write_text('\n'.join(event_lines)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    specs = [('confidence_delta', 'Change in reported P(H=true)', 1),
             ('flip_excess_over_repeat', 'Action flips minus repeat control (pp)', 100),
             ('reward_change_from_m1', 'Change in immediate reward', 1)]
    for ax, (metric, label, factor) in zip(axes, specs):
        for task, color in [('find-plant', '#b93c3a'), ('find-animal', '#2a6dad')]:
            z = boot[(boot.task == task) & (boot.metric == metric) & (boot.duplication > 1)]
            ax.plot(z.duplication, z.estimate * factor, 'o-', color=color, label=task)
            ax.fill_between(z.duplication, z.ci95_low * factor, z.ci95_high * factor, color=color, alpha=.16)
        ax.axhline(0, color='gray', linewidth=.8); ax.set_xscale('log', base=2)
        ax.set_xticks([2, 4, 8], ['2', '4', '8']); ax.set_xlabel('Duplication'); ax.set_ylabel(label)
    axes[0].legend(fontsize=8); fig.suptitle('Exploratory paired effects; 95% episode-cluster intervals')
    fig.tight_layout(); fig.savefig(directory/'episode_curves.png', dpi=180); plt.close(fig)


def run(directory=DEFAULT):
    states, raw, outcomes, meta = validate_raw(directory)
    frame = observations(states, raw, outcomes)
    units, boot = episode_intervals(frame)
    frame.to_csv(directory/'paired_events.csv', index=False)
    units.to_csv(directory/'episode_units.csv', index=False)
    boot.to_csv(directory/'episode_bootstrap.csv', index=False)
    nonbase = frame[frame.duplication > 1]
    nonbase.groupby(['task', 'duplication', 'confidence_changed', 'action_flip']).size().rename('n_comparisons').reset_index().to_csv(directory/'joint_events.csv', index=False)
    meta.update({'analysis_type': 'posthoc exploratory audit', 'resampling_unit': 'task/variation episode',
                 'bootstrap_draws': 20000, 'paired_nonbaseline_comparisons': len(nonbase),
                 'action_flip_events': int(nonbase.action_flip.sum()),
                 'confidence_change_events': int(nonbase.confidence_changed.sum()),
                 'joint_change_events': int(nonbase.joint_confidence_and_action_change.sum()),
                 'harmful_flip_events': int(nonbase.harmful_action_flip.sum()),
                 'harmful_flips_with_confidence_change': int((nonbase.harmful_action_flip * nonbase.confidence_changed).sum()),
                 'confidence_mediated_action_claim': False, 'episode_success_claim': False,
                 'root_calibration_claim': False, 'provenance_superiority_claim': False,
                 'success_objective_posthoc': True,
                 'selection_caveat': 'Adaptive expansion after prior null; fixed first-four selection within this collection; no external preregistration.',
                 'cached_dedup_invariance': 'Same independent m1_repeat output reused across all budgets.'})
    (directory/'verified_analysis.json').write_text(json.dumps(meta, indent=2)+'\n')
    report(directory, frame, boot, meta)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, default=DEFAULT)
    run(p.parse_args().directory)
