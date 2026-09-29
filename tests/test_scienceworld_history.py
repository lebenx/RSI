"""Regression checks for the discovered decision-time observation leakage."""
import copy
import json

import pytest

from submission.scienceworld_frozen_readout import archived_pre_action_visible
from submission.scienceworld_runner import public_observation


def test_public_snapshot_does_not_acquire_later_outcomes():
    history = [{'step': 0, 'action': 'look around', 'observation': 'room'}]
    info = {'valid': ['look around', 'inventory'], 'taskDesc': 'Inspect the room',
            'look': 'room', 'inv': 'empty', 'score': 0}
    snapshot = public_observation('room', info, history, 7)
    before = copy.deepcopy(snapshot)
    history[0]['observation'] = 'modified later'
    history.append({'step': 1, 'reward': 100, 'action': 'future action'})
    assert snapshot == before


def test_archived_input_rejects_future_outcome_and_cross_call_mismatch(tmp_path):
    requests = tmp_path / 'requests'
    requests.mkdir()
    trace = tmp_path / 'episodes' / 'trace.jsonl'
    state = {'snapshot_id': 'example', 'outcome': {'step': 1}}
    visible = {'history': [{'step': 0, 'action': 'look around'}]}

    def write(suffix, body):
        (requests / f'example-{suffix}.json').write_text(json.dumps({
            'request': {'messages': [{}, {'content': json.dumps(body)}]}}))

    write('candidates', visible)
    for suffix in ('bank', 'flat-m1'):
        write(suffix, {'evidence': visible})
    assert archived_pre_action_visible(trace, state) == visible
    write('flat-m1', {'evidence': {'history': []}})
    with pytest.raises(ValueError, match='evidence mismatch'):
        archived_pre_action_visible(trace, state)
    write('candidates', {'history': [{'step': 1, 'reward': 100}]})
    with pytest.raises(ValueError, match='future/current outcome'):
        archived_pre_action_visible(trace, state)
