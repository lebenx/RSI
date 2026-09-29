"""Audit decision-time history in serialized API inputs, independently of traces.

Legacy runner traces referenced a live history list. Its next outcome was
appended before the trace was serialized. Original API request strings are
the authority for what the model saw; trace-derived later readouts can leak.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from submission.scienceworld_runner import canonical_bytes


ROOT = Path('results_submission')


def run():
    rows, index, seen = [], {}, set()
    trace_count = contaminated_traces = 0

    def check(path, scope, step):
        if (str(path), scope) in seen:
            return
        seen.add((str(path), scope))
        if not path.exists():
            return
        entry = json.loads(path.read_text())
        payload = json.loads(entry['request']['messages'][1]['content'])
        visible = payload.get('evidence', payload)
        history = visible.get('history', [])
        checked = step is not None and 'history' in visible
        leak = checked and any(h.get('step', -1) >= step for h in history)
        rows.append({'scope': scope, 'request_file': str(path), 'step': step,
                     'history_length': len(history), 'checked': checked,
                     'current_or_future_outcome': bool(leak),
                     'http_status': entry.get('http_status'), 'ok': entry.get('ok', False)})

    for trace_path in sorted(ROOT.glob('scienceworld*/episodes/*trace.jsonl')):
        for line in trace_path.read_text().splitlines():
            state = json.loads(line)
            if 'snapshot_id' not in state or 'outcome' not in state:
                continue
            sid, step = state['snapshot_id'], state['outcome']['step']
            index[sid] = step
            index[hashlib.sha256(canonical_bytes(state['visible'])).hexdigest()[:24]] = step
            trace_count += 1
            contaminated_traces += any(h.get('step', -1) >= step for h in state['visible'].get('history', []))
            request_dir = trace_path.parent.parent / 'requests'
            for suffix in ('candidates', 'bank', 'flat-m1', 'flat-m2', 'flat-m4', 'flat-m8', 'provenance-readout'):
                check(request_dir / f'{sid}-{suffix}.json', 'original_closed_loop', step)

    for folder in ('interactive_counterfactual', 'scienceworld_snapshot_audit_v4',
                   'scienceworld_frozen_readout_20260929', 'scienceworld_frozen_readout_clean_20260929',
                   'scienceworld_frozen_readout_first4_20260929'):
        for path in sorted((ROOT / folder / 'requests').glob('*.json')):
            match = re.search(r'-s(\d+)-', path.name)
            step = int(match[1]) if match else index.get(path.name.split('-')[0])
            check(path, folder, step)
    output = ROOT / 'report/scienceworld_input_integrity'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'requests.csv', index=False)
    by_scope = {}
    for scope in sorted({r['scope'] for r in rows}):
        group = [r for r in rows if r['scope'] == scope]
        by_scope[scope] = {'requests': len(group), 'checked': sum(r['checked'] for r in group),
                           'current_or_future_outcome': sum(r['current_or_future_outcome'] for r in group)}
    result = {'trace_rows': trace_count, 'trace_rows_with_post_action_history': int(contaminated_traces),
              'scopes': by_scope,
              'interpretation': 'Withdraw contaminated replay/readout causal claims; original API input integrity is assessed separately.',
              'fix': 'public_observation deep-copies history; clean frozen inputs come from original serialized requests.'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run()
