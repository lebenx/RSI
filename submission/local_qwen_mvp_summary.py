"""Summarize the short-horizon local-Qwen ScienceWorld smoke.

This is a dependency-light postprocessor for two already archived runner
directories. It intentionally labels the result as schema/feasibility evidence:
the two-step horizon cannot support an episode-success claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(directory: Path):
    rows = json.loads((directory / 'summaries.json').read_text())
    for row in rows:
        row['run_directory'] = str(directory)
    return rows


def run(m1: Path, m4: Path, output: Path):
    rows = load(m1) + load(m4)
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'rows.csv', index=False)
    summary = frame.groupby(['duplication', 'method']).agg(
        n=('variation', 'size'), success=('success', 'mean'), final_score=('final_score', 'mean'),
        reward=('reward', 'mean'), steps=('steps', 'mean'), tokens=('charged_tokens', 'mean'),
        repeated_no_visible_change=('repeated_no_visible_change', 'mean'),
        initial_brier=('initial_success_probability', lambda x: float(((x - frame.loc[x.index, 'success']) ** 2).mean())),
    ).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    wide = frame.pivot_table(index=['task', 'variation'], columns=['method', 'duplication'],
                             values=['success', 'final_score', 'reward', 'steps', 'charged_tokens'], aggfunc='first')
    paired = []
    for (task, variation), row in wide.iterrows():
        rec = {'task': task, 'variation': int(variation)}
        for metric in ('success', 'final_score', 'reward', 'steps', 'charged_tokens'):
            for method, dup in (('flat', 1), ('flat', 4), ('provenance', 1), ('provenance', 4)):
                rec[f'{method}_m{dup}_{metric}'] = row.get((metric, method, dup))
        rec['provenance_minus_flat_m1_reward'] = rec['provenance_m1_reward'] - rec['flat_m1_reward']
        rec['provenance_minus_flat_m4_reward'] = rec['provenance_m4_reward'] - rec['flat_m4_reward']
        rec['flat_m4_minus_flat_m1_reward'] = rec['flat_m4_reward'] - rec['flat_m1_reward']
        paired.append(rec)
    pd.DataFrame(paired).to_csv(output / 'paired.csv', index=False)
    probe_path = Path('results_submission/report/local_qwen_gpu_probe.json')
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    metadata = {
        'protocol': 'local-qwen-short-horizon-v1', 'model': 'Qwen2.5-Coder-3B-Instruct',
        'directories': [str(m1), str(m4)], 'task': 'find-plant', 'variation': 159,
        'duplications': [1, 4], 'methods': ['flat', 'provenance'], 'step_limit': 2,
        'valid_rows': int(len(frame)), 'error_rows': int(frame.error.notna().sum()),
        'episode_success_claim': False, 'planner_claim': False,
        'interpretation': 'schema/feasibility smoke; horizon is too short for episode-success inference',
        'scienceworld_commit': frame.version.iloc[0] if len(frame) else None,
        'inference_device': probe.get('inference_device'),
        'inference_dtype': probe.get('inference_dtype'),
        'gpu_name': probe.get('gpu_name'),
        'gpu_probe_status': probe.get('status'),
        'device_policy': 'LOCAL_QWEN_DEVICE=cuda (CUDA required)',
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m1', type=Path, default=Path('results_submission/scienceworld_local_qwen_mvp_v159_m1'))
    p.add_argument('--m4', type=Path, default=Path('results_submission/scienceworld_local_qwen_mvp_v159_m4'))
    p.add_argument('--output', type=Path, default=Path('results_submission/report/scienceworld_local_qwen_mvp'))
    a = p.parse_args()
    run(a.m1, a.m4, a.output)


if __name__ == '__main__':
    main()
