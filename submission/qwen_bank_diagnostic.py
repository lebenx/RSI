"""Audit value collapse and rank-zero confounding using archived traces only.

No new predictions or simulator outcomes are fabricated. Episode comparisons
are descriptive: matching rank-zero actions along a recorded trajectory is not
a fresh closed-loop no-imagination evaluation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results_submission/report/qwen_bank_diagnostic'
SOURCES = {
    'main_27_pairs': 'local_qwen_interactive_expanded_table_v4',
    'fair_168_169': 'local_qwen_fair_flatvalue_168_169',
    'heldout_plant': 'local_qwen_heldout174_176',
    'heldout_animal': 'local_qwen_heldout177_179',
}


def bank_metrics(bank, q):
    unique = {}
    for row in bank:
        if row['sample_id'] in unique and unique[row['sample_id']] != row:
            raise ValueError('conflicting sample identity')
        unique[row['sample_id']] = row
    values = np.array([[np.mean([r['predicted_final_score'] for r in unique.values()
                                if r['action_id'] == a and r['hypothesis_value'] == h])
                        for h in (False, True)] for a in range(4)])
    if not np.isfinite(values).all():
        raise ValueError('incomplete conditional coverage')
    scores = values @ np.array([1-q, q])
    # A fixed winner maximizes every convex combination iff it maximizes both
    # endpoints. np.argmax implements the runner's lowest-ID tie-breaking.
    endpoints = [int(np.argmax(values[:, h])) for h in (0, 1)]
    return {
        'unique_samples': len(unique),
        'constant_value': bool(np.ptp(values) == 0),
        'constant_success': len({r['success_probability'] for r in unique.values()}) == 1,
        'example_values_only': all(r['predicted_final_score'] == .1 and
                                   r['success_probability'] == .2 for r in unique.values()),
        'empty_imagined_actions': all(not r.get('imagined_actions') for r in unique.values()),
        'max_branch_value_difference': float(np.abs(values[:, 1]-values[:, 0]).max()),
        'score_range': float(np.ptp(scores)),
        'top_two_margin': float(np.sort(scores)[-1]-np.sort(scores)[-2]),
        'belief_sensitive_winner': endpoints[0] != endpoints[1],
        'mixture_action_id': int(np.argmax(scores)),
    }


def main():
    steps, episodes = [], []
    for cohort, directory in SOURCES.items():
        frame = pd.read_csv(ROOT / 'results_submission/report' / directory / 'rows.csv')
        for _, ep in frame.iterrows():
            path = ROOT / ep.trace_file
            trace = [json.loads(line) for line in path.read_text().splitlines()]
            episode_steps = []
            for i, row in enumerate(trace):
                c = row['candidate']
                b = bank_metrics(row['bank'], c['p_true'])
                chosen = row['decision']['action_id']
                rec = dict(cohort=cohort, task=ep.task, variation=int(ep.variation),
                           method=ep.method, duplication=int(ep.duplication), step=i,
                           trace_file=ep.trace_file, snapshot_id=row['snapshot_id'], **b,
                           selected_action_id=chosen, rank_zero_match=chosen == 0,
                           direct_action_match=chosen == c['direct_action_id'],
                           candidate_repaired=c.get('candidate_repaired', 0))
                steps.append(rec); episode_steps.append(rec)
            episodes.append(dict(cohort=cohort, task=ep.task, variation=int(ep.variation),
                                 method=ep.method, duplication=int(ep.duplication), steps=len(trace),
                                 reward=ep.reward, success=ep.success,
                                 all_constant_value=all(r['constant_value'] for r in episode_steps),
                                 all_rank_zero=all(r['rank_zero_match'] for r in episode_steps),
                                 all_direct_action=all(r['direct_action_match'] for r in episode_steps)))
    OUT.mkdir(parents=True, exist_ok=True)
    s, e = pd.DataFrame(steps), pd.DataFrame(episodes)
    s.to_csv(OUT/'steps.csv', index=False); e.to_csv(OUT/'episodes.csv', index=False)
    summary = s.groupby(['cohort', 'method'], as_index=False).agg(
        steps=('step', 'size'), constant_value_rate=('constant_value', 'mean'),
        example_values_rate=('example_values_only', 'mean'),
        belief_sensitive_winner_rate=('belief_sensitive_winner', 'mean'),
        rank_zero_match_rate=('rank_zero_match', 'mean'),
        direct_action_match_rate=('direct_action_match', 'mean'),
        repaired_candidate_step_rate=('candidate_repaired', lambda x: float((x>0).mean())))
    summary.to_csv(OUT/'summary.csv', index=False)
    main_s = s[s.cohort == 'main_27_pairs']
    main_e = e[(e.cohort == 'main_27_pairs') & (e.method == 'provenance_value')]
    # Do not treat duplicated method/budget observations as independent banks.
    unique = main_s.drop_duplicates('snapshot_id')
    meta = {'main_steps': len(main_s), 'main_constant_steps': int(main_s.constant_value.sum()),
            'main_unique_snapshots': len(unique),
            'main_unique_constant_snapshots': int(unique.constant_value.sum()),
            'provenance_episode_rows': len(main_e),
            'provenance_rank_zero_episode_rows': int(main_e.all_rank_zero.sum()),
            'new_model_calls': 0, 'new_environment_episodes': 0,
            'inference': 'Historical reward differences are confounded by reader versus rank-zero selection. '
                         'Template-value matching is observed; a causal prompt-copying explanation needs an intervention.',
            'planning_attribution_established': False}
    (OUT/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    lines = ['# Qwen conditional-bank diagnostic', '',
             'Generated with `python -m submission.qwen_bank_diagnostic` from archived traces.', '',
             f"Main table: {meta['main_constant_steps']}/{meta['main_steps']} decision rows have constant conditional scores; "
             f"{meta['main_unique_constant_snapshots']}/{meta['main_unique_snapshots']} unique snapshots do so.", '',
             f"{meta['provenance_rank_zero_episode_rows']}/{meta['provenance_episode_rows']} provenance episode rows "
             'follow candidate ID 0 at every recorded step. This is a trajectory-level behavioral identity, '
             'not a newly executed no-imagination baseline.', '',
             '| Cohort | Method | Steps | Constant scores | Template values | Belief-sensitive winner | Rank 0 match |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.cohort} | {r.method} | {r.steps} | {r.constant_value_rate:.3f} | '
                     f'{r.example_values_rate:.3f} | {r.belief_sensitive_winner_rate:.3f} | {r.rank_zero_match_rate:.3f} |')
    lines += ['', 'The compact local prompt specifies a schema and example values but omits the API prompt\'s '
              'conditional value-estimation instructions. The observed matching values suggest an example-copying '
              'failure; the prompt has not yet been causally isolated. Constant banks force the deterministic '
              'aggregator to choose the lowest action ID independently of premise probability.', '',
              'Historical positive reward differences remain descriptive, but are not evidence that provenance '
              'caused the improvement. The next experiment must compare identical candidate/bank inputs under '
              'an explicitly semantic local rollout prompt, include rank-zero and direct-policy baselines, and '
              'measure value discrimination and calibration before interpreting closed-loop rewards. '
              'Keep constant banks and failures; do not select only instances that show a positive effect.', '',
              'Belief-sensitive winner tests q=0 and q=1 analytically: with linear mixtures and fixed ID '
              'tie-breaking, identical endpoint winners imply the same winner for every q in [0,1]. '
              'All per-step and per-episode observations are retained in the CSVs.']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(meta, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
