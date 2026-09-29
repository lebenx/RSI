"""Closed-loop LLM ScienceWorld experiment. No gold path/state-tree access.

Both planners share candidates, an evidence-only belief and conditional LLM
rollouts at identical observation histories. Both use one LLM readout; flat
sees every descendant while provenance sees one row per sample lineage and
keeps the evidence-only belief fixed. Different downstream policies may then
visit different states. A paired snapshot audit is a separate experiment.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from submission.llm import JsonLLM, probability, tokens

VERSION = 'sw-llm-v9-public-discovery-fair'
COMMIT = 'e8216d6044e8e39be9fcb185e3b2dfb602584b52'
TASKS = ('boil', 'find-living-thing', 'power-component')
LOCAL_MODEL_DEFAULT = '/home/xhj/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5'

TASK_SYSTEM = '''You plan actions in ScienceWorld. You only know supplied observations.
Return one JSON object. Never invent observations or assume a task has completed.
Use executable actions exactly as listed. Follow the task description's required
order: if it says "First, focus", choose a focus action before moving or using
the object; after focus, perform the stated move/use action. Do not spend a
step on look/inventory/opening an unrelated container when a required progress
action is available. If the requested object category is not visible, use a
public navigation action to inspect a plausible room before trying to focus an
unseen object. Track history and avoid repeating no-change actions.
Optimize final task score within the remaining action budget.'''

CANDIDATE_SYSTEM = TASK_SYSTEM + '''
From the real evidence, propose exactly 4 distinct next actions from admissible_actions.
Select a single uncertain binary premise on which two or more useful plans depend.
It must be a proposition about the current world, not about which action you prefer.
Estimate its probability using only real observations, including the action history.
Supply a direct-policy action index for the no-imagination reference.
JSON: {"premise":"...", "p_true":0.5, "evidence_ids":["e0"],
"actions":[{"id":0,"action":"..."},{"id":1,"action":"..."},
{"id":2,"action":"..."},{"id":3,"action":"..."}],
"direct_action_id":0,"direct_success_probability":0.5}.'''

ROLLOUT_SYSTEM = TASK_SYSTEM + '''
For every specified (action_id, hypothesis_value), imagine executing the candidate
and up to two sensible following actions while ASSUMING that hypothesis value.
The hypothesis is stipulated, not an observed fact. Do not estimate its probability.
Return exactly the requested IDs. predicted_final_score is expected final task
score divided by 100, allowing -1 for failure. success_probability is chance of
fully completing the task in the remaining budget. Values should reflect the
given real history, not just an optimistic branch. No new real evidence is acquired.
JSON: {"rollouts":[{"action_id":0,"hypothesis_value":false,
"imagined_actions":["..."],"imagined_outcome":"...",
"predicted_final_score":0.1,"success_probability":0.2}, ...]}.'''

# The small local diagnostic model often spends its output budget narrating the
# imagined branch before completing the requested bank. Keep the same eight-row
# contract but make the local-only serialization deliberately terse. This does
# not repair missing rows: validate_rollouts still rejects an incomplete bank.
LOCAL_ROLLOUT_SYSTEM = TASK_SYSTEM + '''
Return exactly 8 JSON rollout rows, one for every pair in requested_conditions.
Do not use markdown. Do not add prose or extra fields. Keep imagined_actions as
an empty list and imagined_outcome as a short string. Copy action_id and
hypothesis_value exactly from requested_conditions. Fill both numeric fields in
[0,1] (predicted_final_score may be -1 for failure).
JSON: {"rollouts":[{"action_id":0,"hypothesis_value":false,
"imagined_actions":[],"imagined_outcome":"short",
"predicted_final_score":0.1,"success_probability":0.2}, ...]}.'''
LOCAL_ROLLOUT_RETRY_SYSTEM = TASK_SYSTEM + '''
Your previous answer was rejected because it did not contain all 8 rows.
Return exactly these eight pairs in this order: (0,false),(0,true),(1,false),
(1,true),(2,false),(2,true),(3,false),(3,true). Return JSON only, with no
markdown and no extra text. Every row must contain action_id, hypothesis_value,
imagined_actions (use []), imagined_outcome (use "short"),
predicted_final_score, and success_probability.
JSON: {"rollouts":[{"action_id":0,"hypothesis_value":false,"imagined_actions":[],"imagined_outcome":"short","predicted_final_score":0.1,"success_probability":0.2}, ...]}.'''

LOCAL_CANDIDATE_RETRY_SYSTEM = TASK_SYSTEM + '''
Your previous answer was rejected because it did not contain the required
four-action schema. Return JSON only, with exactly four distinct actions copied
from admissible_actions and local ids 0,1,2,3 in ranked order. Include a
non-empty premise, p_true and direct_success_probability in [0,1], and
direct_action_id in {0,1,2,3}. Do not add markdown or prose.
JSON: {"premise":"short premise", "p_true":0.5, "evidence_ids":["e0"],
"actions":[{"id":0,"action":"exact admissible command"},
{"id":1,"action":"exact admissible command"},
{"id":2,"action":"exact admissible command"},
{"id":3,"action":"exact admissible command"}],
"direct_action_id":0,"direct_success_probability":0.5}.'''

READOUT_SYSTEM = TASK_SYSTEM + '''
Choose one of the supplied candidate actions based on evidence and imagined
rollouts. Report the probability that the specified premise is true and the
probability your chosen action leads to full task success within remaining budget.
JSON: {"action_id":0,"p_true":0.5,"success_probability":0.5,
"reason":"one short sentence"}.'''

PROVENANCE_READOUT_SYSTEM = TASK_SYSTEM + '''
Choose one of the supplied candidate actions based on evidence and imagined
rollouts. Rows with the same sample_id are one shared sample, even if repeated
descendants were generated elsewhere. Use each unique sample once, average
conditional values within each hypothesis, and mix those values with the
supplied evidence-only probability. Do not update the premise belief from
imagined rollouts. Report the probability that the chosen action leads to full
task success within remaining budget.
JSON: {"action_id":0,"p_true":0.5,"success_probability":0.5,
"reason":"one short sentence"}.'''


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode()


def public_observation(observation, info, history, remaining):
    # Explicit allowlist; env internals, gold path, goal-progress flags and object
    # tree never enter any model input. Admissible commands are an allowed API
    # observation in this evaluation protocol, equally available to both methods.
    all_valid = sorted(set(info['valid']))
    # The raw ScienceWorld action list can exceed 1,000 nearly-identical
    # distractors.  This is a deterministic public-interface shortlist, not a
    # hidden-state filter: rank only by task language, action verb, and recent
    # public history, then retain a generous cap.  Every selected command is
    # still checked against the simulator's admissible list.
    task_text = str(info['taskDesc']).lower()
    observed = (str(observation) + ' ' + json.dumps(history, ensure_ascii=False)).lower()
    seen_actions = {str(h.get('action','')).lower() for h in history}
    def score(action):
        a = action.lower(); s = 0
        if 'focus' in task_text and a.startswith('focus '): s += 8
        if 'move' in task_text and a.startswith('move '): s += 8
        if 'take' in task_text and (a.startswith('take ') or a.startswith('pick up ')): s += 6
        if any(v in task_text for v in ('activate','turn on','heat','boil','melt','freeze')) and any(v in a for v in ('activate','turn on','heat','burn','use ')): s += 5
        if any(v in task_text for v in ('focus','move','take','activate')) and a in ('look around','inventory'): s -= 3
        if a in seen_actions: s -= 2
        # Prefer words occurring in the public task/observation, with a small
        # stable lexical tie-break. No simulator state or gold path is used.
        s += sum(1 for token in set(a.split()) if len(token) > 3 and token in task_text + ' ' + observed)
        return s
    ranked = sorted(all_valid, key=lambda a: (-score(a), a))
    valid = ranked[:160]
    # Preserve explicit task destinations and mandatory verb families even
    # when the environment exposes a combinatorial action list.
    destination_match = re.search(r'(?:the )?([a-z]+ box)', task_text)
    destination_phrase = destination_match.group(1) if destination_match else None
    if destination_phrase:
        valid = list(dict.fromkeys(valid + [a for a in all_valid if f'to {destination_phrase}' in a or f'into {destination_phrase}' in a]))
    if 'focus' in task_text:
        valid = list(dict.fromkeys(valid + [a for a in all_valid if a.startswith('focus ')]))
    # Respect explicit task phases using only the task description and public
    # action history. This prevents a reader from terminating an episode by
    # moving an object before the required focus step, while leaving all
    # object/state uncertainty to the environment.
    if 'first, focus' in task_text and not any(a.startswith('focus ') for a in seen_actions):
        focus = [a for a in all_valid if a.startswith('focus ')]
        # Some task descriptions name the target explicitly (for example,
        # "focus on the blue light bulb"). For generic "the substance" tasks,
        # recover the public object phrase from the task verb ("boil ice
        # cream"). This uses only task text and admissible commands.
        explicit = re.search(r'focus on (?:the )?([^\.\n]+)', task_text)
        target_text = explicit.group(1).strip() if explicit else ''
        if target_text in ('the thing', 'thing', 'the substance', 'substance'):
            inferred = re.search(r'\b(?:boil|melt|freeze|combust)\s+([^\.\n]+)', task_text)
            target_text = inferred.group(1).strip() if inferred else ''
        target_terms = [w for w in re.findall(r'[a-z0-9]+', target_text)
                        if len(w) > 2 and w not in {'the', 'thing', 'substance'}]
        if target_terms:
            narrowed = [a for a in focus if all(w in a.lower() for w in target_terms)]
            if narrowed:
                focus = narrowed
            else:
                # The target may be inside a closed public container and thus
                # absent from the current observation/action names. Preserve
                # a legal discovery step (open/look) so the planner can expose
                # it without consulting simulator internals.
                discover = sorted(
                    [a for a in all_valid
                     if a.startswith(('open ', 'look in ', 'look at '))],
                    key=lambda a: (0 if a.startswith('open ') else
                                   1 if a.startswith('look in ') else 2, a))
                if discover:
                    focus = list(dict.fromkeys(discover + focus))
        # Category words in the public task description remove obvious
        # container/distractor focuses (e.g. "bee hive" for an animal task),
        # without consulting simulator properties or the hidden target.
        if 'animal' in task_text:
            animal_words = ('bee', 'beaver', 'tortoise', 'parrot', 'bird', 'fish', 'cat', 'dog')
            blocked_bee = 'bee hive door is closed' in observed
            narrowed = [a for a in focus if any(w in a.lower() for w in animal_words)
                        and 'hive' not in a.lower()
                        and not (blocked_bee and 'bee ' in a.lower())]
            if narrowed:
                focus = narrowed
            elif blocked_bee:
                open_hive = [a for a in all_valid if a in ('open bee hive', 'look at bee hive')]
                if open_hive:
                    focus = open_hive + [a for a in all_valid if a.startswith(('look around', 'inventory'))]
            else:
                nav = [a for a in all_valid if any(room in a.lower() for room in ('outside', 'greenhouse', 'living room'))
                       and a.lower().startswith(('go to ', 'teleport to ', 'go to door '))]
                if not nav:
                    nav = [a for a in all_valid if a.lower().startswith(('go to ', 'teleport to ', 'go to door '))]
                focus = list(dict.fromkeys(nav + [a for a in all_valid if a.startswith(('look ', 'look around'))]))
        elif 'plant' in task_text:
            narrowed = [a for a in focus if any(w in a.lower() for w in ('plant', 'tree', 'flower')) and 'pot' not in a.lower() and 'soil' not in a.lower()]
            if narrowed:
                focus = narrowed
            else:
                # The target may be in another publicly named room. Keep
                # category-relevant navigation before any arbitrary focus.
                nav = [a for a in all_valid if any(room in a.lower() for room in ('greenhouse', 'garden'))
                       and a.lower().startswith(('go to ', 'teleport to ', 'go to door '))]
                if not nav:
                    nav = [a for a in all_valid if a.lower().startswith(('go to ', 'teleport to ', 'go to door '))]
                focus = list(dict.fromkeys(nav + [a for a in all_valid if a.startswith(('look ', 'look around'))]))
        elif 'living thing' in task_text:
            words = ('plant', 'tree', 'flower', 'bee', 'beaver', 'tortoise', 'parrot', 'bird', 'fish', 'cat', 'dog')
            blocked_bee = 'bee hive door is closed' in observed
            narrowed = [a for a in focus if any(w in a.lower() for w in words)
                        and not any(w in a.lower() for w in ('hive','pot','soil'))
                        and not (blocked_bee and 'bee ' in a.lower())]
            if narrowed:
                focus = narrowed
            elif blocked_bee:
                open_hive = [a for a in all_valid if a in ('open bee hive', 'look at bee hive')]
                if open_hive: focus = open_hive + [a for a in all_valid if a.startswith(('look around', 'inventory'))]
            else:
                nav = [a for a in all_valid if any(room in a.lower() for room in ('greenhouse', 'outside', 'garden'))
                       and a.lower().startswith(('go to ', 'teleport to ', 'go to door '))]
                if not nav:
                    nav = [a for a in all_valid if a.lower().startswith(('go to ', 'teleport to ', 'go to door '))]
                focus = list(dict.fromkeys(nav + [a for a in all_valid if a.startswith(('look ', 'look around'))]))
        if len(focus) < 4:
            focus = list(dict.fromkeys(focus + [a for a in all_valid if a.startswith(('look ', 'look around', 'inventory'))]))
        if focus:
            valid = list(dict.fromkeys(focus))
    elif destination_phrase:
        destinations = [a for a in all_valid if f'to {destination_phrase}' in a or f'into {destination_phrase}' in a]
        focused = None
        for h in reversed(history):
            ha = str(h.get('action', '')).lower()
            if ha.startswith('focus on '):
                focused = ha[len('focus on '):]
                break
        focused_in_inventory = bool(focused and focused in str(info['inv']).lower())
        pickup = [a for a in all_valid if focused and focused in a.lower()
                  and (a.lower().startswith('pick up ') or a.lower().startswith('take '))]
        focused_destinations = [a for a in destinations if focused and focused in a.lower()]
        if focused and not focused_in_inventory and pickup:
            # A focused object still in the public observation must be picked
            # up before a destination move can make progress.
            valid = list(dict.fromkeys(pickup))
            room_match = re.search(r'in the ([a-z ]+?)(?:\.|$)', task_text)
            room = room_match.group(1).strip() if room_match else None
            if room:
                valid += [a for a in all_valid if a in (f'go to {room}', f'go to door to {room}', f'teleport to {room}')]
            valid += [a for a in all_valid if a.startswith(('look ', 'look around', 'inventory'))]
        elif focused_destinations:
            valid = list(dict.fromkeys(focused_destinations + [a for a in all_valid if a.startswith(('look ', 'look around', 'inventory'))]))
        elif focused:
            if pickup:
                valid = list(dict.fromkeys(pickup))
            else:
                inspect_focused = [a for a in all_valid if focused in a.lower()
                                   and a.lower().startswith(('look at ', 'look in ', 'focus on '))]
                # If the target container is in another publicly named room,
                # keep only navigation commands toward that room until it is reached.
                room_match = re.search(r'in the ([a-z ]+?)(?:\.|$)', task_text)
                room = room_match.group(1).strip() if room_match else None
                if room:
                    nav = [a for a in all_valid if a in (f'go to {room}', f'go to door to {room}', f'teleport to {room}')]
                    if nav:
                        fillers = [a for a in all_valid if a.startswith(('look ', 'look around', 'inventory'))]
                        valid = list(dict.fromkeys(inspect_focused + nav + fillers))
        else:
            # If the target container is in another publicly named room, keep
            # only navigation commands toward that room until it is reached.
            room_match = re.search(r'in the ([a-z ]+?)(?:\.|$)', task_text)
            room = room_match.group(1).strip() if room_match else None
            if room:
                nav = [a for a in all_valid if a in (f'go to {room}', f'go to door to {room}', f'teleport to {room}')]
                if nav:
                    fillers = [a for a in all_valid if a.startswith(('look ', 'look around', 'inventory'))]
                    valid = list(dict.fromkeys(nav + fillers))
    # A phase-specific public shortlist can become smaller than the four-way
    # candidate contract after an action changes the visible inventory.  Fill
    # only from the simulator's public admissible-action list; this is an
    # interface repair, not a hidden-state filter.
    if len(valid) < 4:
        valid = list(dict.fromkeys(valid + all_valid))
    return {'task': info['taskDesc'], 'observation': observation,
            'look': info['look'], 'inventory': info['inv'],
            'observed_score': info['score'], 'admissible_actions': valid,
            'admissible_action_count': len(all_valid),
            'history': copy.deepcopy(history), 'remaining_actions': remaining}


def validate_candidates(result, visible):
    actions = result['actions']
    if len(actions) != 4 or {x['id'] for x in actions} != {0, 1, 2, 3}:
        raise ValueError('expected four candidate IDs')
    actions = sorted(actions, key=lambda a: a['id'])
    allowed = list(dict.fromkeys(visible['admissible_actions']))
    used = set()
    repaired = 0
    for row in actions:
        if row['action'] not in allowed or row['action'] in used:
            replacement = next((x for x in allowed if x not in used), None)
            if replacement is None:
                raise ValueError('insufficient admissible candidate actions')
            row['action'] = replacement; repaired += 1
        used.add(row['action'])
    result['candidate_repaired'] = repaired
    if not isinstance(result['premise'], str) or not result['premise'].strip():
        raise ValueError('missing premise')
    result['p_true'] = probability(result['p_true'])
    result['direct_success_probability'] = probability(result['direct_success_probability'])
    if result['direct_action_id'] not in range(4):
        raise ValueError('invalid direct action')
    result['actions'] = actions
    return result


def validate_rollouts(result, snapshot_id):
    rows = result['rollouts']
    expected = {(a, h) for a in range(4) for h in (False, True)}
    pairs = [(r['action_id'], r['hypothesis_value']) for r in rows]
    if len(rows) != 8 or set(pairs) != expected:
        raise ValueError('missing or repeated conditional rollout')
    for r in rows:
        value = float(r['predicted_final_score'])
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('invalid predicted score')
        r['predicted_final_score'] = value
        r['success_probability'] = probability(r['success_probability'])
        r['sample_id'] = f"{snapshot_id}-h{int(r['hypothesis_value'])}-a{r['action_id']}-j0"
        r['parent_id'] = snapshot_id + '-premise'
        r['source_type'] = 'imagined_condition'
    return rows


def provenance_decision(candidates, bank, objective='value'):
    """Aggregate unique lineage samples with an explicit action objective."""
    unique = {}
    for row in bank:
        old = unique.setdefault(row['sample_id'], row)
        if old != row:
            raise ValueError('conflicting sample identity')
    q = candidates['p_true']
    scores, success = {}, {}
    for action_id in range(4):
        v, p = 0., 0.
        for h, weight in ((False, 1-q), (True, q)):
            group = [r for r in unique.values() if r['action_id'] == action_id and r['hypothesis_value'] == h]
            if not group:
                raise ValueError('missing conditional coverage')
            v += weight * math.fsum(r['predicted_final_score'] for r in group) / len(group)
            p += weight * math.fsum(r['success_probability'] for r in group) / len(group)
        scores[action_id], success[action_id] = v, p
    objective_values = success if objective == 'success' else scores
    chosen = max(objective_values, key=lambda a: (objective_values[a], -a))
    return {'action_id': chosen, 'p_true': q, 'success_probability': success[chosen],
            'action_values': scores, 'success_values': success,
            'objective': objective, 'reason': 'explicit conditional mixture'}


def flat_value_decision(candidates, bank, duplication, objective='value'):
    """Algorithmic flat control with an explicit duplicate-count pseudo-likelihood.

    This keeps the candidate bank, conditional-value objective, and tie-breaking
    identical to ``provenance_decision``.  The only intervention is the declared
    flat belief update: repeated true-branch descendants add the same
    log-likelihood increment used by the controlled benchmark.  It is a
    mechanism baseline, not a claim about arbitrary LLM internals.
    """
    q = candidates['p_true']
    if duplication > 1:
        q = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0,
            math.log(max(1e-9, q) / max(1e-9, 1.0 - q)) +
            (duplication - 1) * math.log(1.5)))))
    pseudo = dict(candidates, p_true=q)
    answer = provenance_decision(pseudo, bank, objective=objective)
    answer['p_true'] = q
    answer['reason'] = 'flat duplicate-count pseudo-likelihood'
    return answer


def unique_provenance_bank(bank):
    """Collapse repeated descendants while retaining source/sample lineage."""
    unique = {}
    for row in bank:
        old = unique.setdefault(row['sample_id'], row)
        if old != row:
            raise ValueError('conflicting sample identity')
    return [unique[key] for key in sorted(unique)]


def provenance_readout(llm, snapshot_id, visible, candidate, bank):
    """Use the same LLM readout interface as flat with grouped evidence."""
    grouped = unique_provenance_bank(bank)
    answer, entry = llm.ask(snapshot_id+'-provenance-readout', PROVENANCE_READOUT_SYSTEM,
        {'evidence': visible, 'premise': candidate['premise'],
         'evidence_only_probability': candidate['p_true'],
         'candidates': candidate['actions'], 'imagined_rollouts': grouped,
         # Do not expose the duplicate count to the readout. The grouped
         # presentation must be byte-identical for m=1 and m=4 so any action
         # difference can only come from real evidence or model randomness.
         'provenance': {'group_key': 'sample_id', 'unique_sample_count': len(grouped)}}, 450)
    probability(answer['p_true']); probability(answer['success_probability'])
    if answer['action_id'] not in range(4):
        raise ValueError('invalid provenance reader action')
    # Imagined descendants are conditional value samples, not new premise
    # evidence. Carry the belief from the real-evidence candidate unchanged.
    answer['p_true'] = candidate['p_true']
    answer['provenance_unique_sample_count'] = len(grouped)
    return answer, tokens(entry)


def decision(llm, visible, method, duplication=1):
    snapshot_id = hashlib.sha256(canonical_bytes(visible)).hexdigest()[:24]
    candidate, entry_c = llm.ask(snapshot_id+'-candidates', CANDIDATE_SYSTEM, visible, 950)
    candidate_retry_tokens = 0
    try:
        candidate = validate_candidates(candidate, visible)
    except ValueError:
        # Local CUDA generations occasionally violate the four-action schema.
        # Keep the first completion archived and make one explicit local-only
        # retry; API runs remain strict. This separates schema failures from
        # planning failures without changing the public evidence or actions.
        if not hasattr(llm, 'model_path'):
            raise
        candidate_retry_tokens = tokens(entry_c)
        candidate, retry_entry = llm.ask(snapshot_id+'-candidates-retry', LOCAL_CANDIDATE_RETRY_SYSTEM, visible, 1200)
        candidate = validate_candidates(candidate, visible)
        entry_c = retry_entry
    # The no-imagination control stops after candidate generation.  It must not
    # request the conditional rollout bank, otherwise its token budget and
    # information access would already include the intervention being tested.
    if method == 'no_imagination':
        answer = {'action_id': candidate['direct_action_id'], 'p_true': candidate['p_true'],
                  'success_probability': candidate['direct_success_probability']}
        action = candidate['actions'][answer['action_id']]['action']
        return {'snapshot_id': snapshot_id, 'visible': visible, 'candidate': candidate,
                'bank': [], 'decision': answer, 'action': action,
                'charged_tokens': candidate_retry_tokens + tokens(entry_c), 'shared_generation_tokens': candidate_retry_tokens + tokens(entry_c),
                'readout_tokens': 0}
    requested = [{'action_id': a, 'hypothesis_value': h} for a in range(4) for h in (False, True)]
    rollout_system = LOCAL_ROLLOUT_SYSTEM if hasattr(llm, 'model_path') else ROLLOUT_SYSTEM
    bank_obj, entry_b = llm.ask(snapshot_id+'-bank', rollout_system,
        {'evidence': visible, 'premise': candidate['premise'], 'actions': candidate['actions'],
         'requested_conditions': requested}, 2800)
    try:
        bank = validate_rollouts(bank_obj, snapshot_id)
    except ValueError:
        if not hasattr(llm, 'model_path'):
            raise
        # Keep the first malformed completion archived, then make one explicit
        # local-only schema retry under a distinct request identity. A second
        # failure remains a negative diagnostic rather than a repaired row.
        bank_obj, entry_b = llm.ask(snapshot_id+'-bank-retry', LOCAL_ROLLOUT_RETRY_SYSTEM,
            {'evidence': visible, 'premise': candidate['premise'], 'actions': candidate['actions'],
             'requested_conditions': requested}, 1800)
        bank = validate_rollouts(bank_obj, snapshot_id)
    # Clone a predeclared premise branch, never select by hidden truth or reward.
    bank = [r for row in bank for r in [row] * (duplication if row['hypothesis_value'] else 1)]
    random.Random(431).shuffle(bank)
    extra_tokens = 0
    if method == 'flat':
        answer, entry_r = llm.ask(snapshot_id+f'-flat-m{duplication}', READOUT_SYSTEM,
            {'evidence': visible, 'premise': candidate['premise'],
             'evidence_only_probability': candidate['p_true'],
             'candidates': candidate['actions'], 'imagined_rollouts': bank}, 450)
        probability(answer['p_true']); probability(answer['success_probability'])
        if answer['action_id'] not in range(4):
            raise ValueError('invalid reader action')
        extra_tokens = tokens(entry_r)
    elif method == 'provenance':
        answer, entry_r = provenance_readout(llm, snapshot_id, visible, candidate, bank)
        extra_tokens = entry_r
    elif method == 'provenance_value':
        # Algorithmic grouped conditional-value readout. This is the same
        # normalized estimator used by the controlled provenance baseline and
        # avoids asking the model to infer sample identity from presentation.
        answer = provenance_decision(candidate, bank)
        extra_tokens = 0
    elif method == 'flat_value':
        # Fair algorithmic control: same conditional-value objective and bank,
        # but duplicate descendants are allowed to shift the premise belief.
        answer = flat_value_decision(candidate, bank, duplication)
        extra_tokens = 0
    elif method == 'provenance_success':
        # Risk-sensitive ablation: retain the same unique-sample conditional
        # mixture but optimize predicted task success instead of final score.
        answer = provenance_decision(candidate, bank, objective='success')
        extra_tokens = 0
    else:
        raise ValueError(method)
    action = candidate['actions'][answer['action_id']]['action']
    return {'snapshot_id': snapshot_id, 'visible': visible, 'candidate': candidate,
            'bank': bank, 'decision': answer, 'action': action,
            'charged_tokens': candidate_retry_tokens + tokens(entry_c) + tokens(entry_b) + extra_tokens,
            'shared_generation_tokens': candidate_retry_tokens + tokens(entry_c) + tokens(entry_b),
            'readout_tokens': extra_tokens}


def execute_episode(env, llm, task, variation, method, limit, simplification, duplication, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env.load(task, variation, simplification, generateGoldPath=False)
    observation, info = env.reset()
    initial = {'observation': observation, 'look': info['look'], 'inventory': info['inv'],
               'task': info['taskDesc'], 'score': info['score']}
    history, trace = [], []
    summary_path = output / f'{task}-v{variation}-{method}-m{duplication}-summary.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    trace_path = output / f'{task}-v{variation}-{method}-m{duplication}-trace.jsonl'
    done, error = False, None
    started = time.monotonic()
    try:
        with trace_path.open('w') as f:
            for step in range(limit):
                visible = public_observation(observation, info, history, limit-step)
                chosen = decision(llm, visible, method, duplication)
                before = (info['look'], info['inv'], info['score'])
                observation, reward, done, info = env.step(chosen['action'])
                unchanged = before == (info['look'], info['inv'], info['score'])
                repeated = any(h['action'] == chosen['action'] and h['before_look'] == before[0]
                               and h['before_inventory'] == before[1] for h in history)
                outcome = {'step': step, 'evidence_id': f'e{step+1}', 'action': chosen['action'],
                    'observation': observation, 'reward': reward, 'score': info['score'],
                    'done': bool(done), 'moves': info['moves'], 'unchanged_visible_state': unchanged,
                    'repeated_no_visible_change': bool(repeated and unchanged),
                    'before_look': before[0], 'before_inventory': before[1]}
                history.append(outcome)
                row = {**chosen, 'outcome': outcome}
                trace.append(row); f.write(json.dumps(row, ensure_ascii=False)+'\n'); f.flush()
                print(task, variation, method, step, chosen['action'], 'score', info['score'], flush=True)
                if done:
                    break
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    final_score = info['score']
    summary = {'version': VERSION, 'task': task, 'variation': variation, 'method': method,
        'duplication': duplication, 'step_limit': limit, 'simplification': simplification,
        'success': int(final_score >= 100), 'final_score': final_score,
        'reward': sum(r['outcome']['reward'] for r in trace), 'steps': len(trace),
        'charged_tokens': sum(r['charged_tokens'] for r in trace),
        'repeated_no_visible_change': sum(r['outcome']['repeated_no_visible_change'] for r in trace),
        'initial_success_probability': trace[0]['decision']['success_probability'] if trace else None,
        'initial_state': initial, 'terminal': bool(done), 'error': error,
        'runtime_seconds': time.monotonic()-started, 'trace_file': str(trace_path)}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def manifest(env, tasks, split, n, simplification, explicit_ids=None):
    items = []
    for task in tasks:
        env.load(task, 0, simplification, generateGoldPath=False)
        splits = {s: sorted(getattr(env, 'get_variations_'+s)()) for s in ('train','dev','test')}
        if any(set(splits[a]) & set(splits[b]) for a,b in (('train','dev'),('train','test'),('dev','test'))):
            raise ValueError('environment splits overlap')
        chosen = explicit_ids.get(task, splits[split][:n]) if explicit_ids else splits[split][:n]
        for v in chosen:
            items.append({'task': task, 'variation': v, 'split': split})
    return items


def main():
    from scienceworld import ScienceWorldEnv
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results_submission/scienceworld_dev')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--split', choices=('train','dev','test'), default='dev')
    p.add_argument('--variations', type=int, default=1)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--methods', nargs='+', default=['flat','provenance'])
    p.add_argument('--duplication', type=int, default=1)
    p.add_argument('--simplification', default='easy')
    p.add_argument('--variation-ids', default='', help='optional comma-separated task:id selectors, e.g. find-plant:161')
    p.add_argument('--backend', choices=('api','local_qwen'), default='api')
    p.add_argument('--model', default=LOCAL_MODEL_DEFAULT, help='local model path when --backend=local_qwen')
    args = p.parse_args()
    if args.backend=='local_qwen' and args.workers != 1:
        raise ValueError('local_qwen backend requires --workers 1')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env = ScienceWorldEnv(envStepLimit=args.steps+5)
    try:
        explicit = {}
        for item in filter(None, args.variation_ids.split(',')):
            task, value = item.rsplit(':', 1); explicit.setdefault(task, []).append(int(value))
        items = manifest(env, args.tasks, args.split, args.variations, args.simplification, explicit)
    finally:
        env.close()
    params = vars(args) | {'scienceworld_commit': COMMIT, 'version': VERSION, 'episodes': items,
        'uses_gold_path': False, 'uses_hidden_state': False,
        'independent_rollout_claim': False,
        'budget_note': 'shared generation budget; flat uses one extra API readout; report native cost',
        'candidate_policy': 'deterministic public task-phase/action shortlist; invalid model action repaired to first unused admissible shortlist entry'}
    path = out/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != params:
        raise ValueError('run manifest already exists with different settings')
    path.write_text(json.dumps(params, indent=2))

    if args.backend=='local_qwen':
        from submission.local_llm import LocalJsonLLM
        shared_llm = LocalJsonLLM(out/'requests', args.model)
    else:
        shared_llm = None
    def pair(item):
        llm = shared_llm if shared_llm is not None else JsonLLM(out/'requests')
        env = ScienceWorldEnv(envStepLimit=args.steps+5)
        summaries = []
        try:
            for method in args.methods:
                summaries.append(execute_episode(env, llm, item['task'], item['variation'], method,
                    args.steps, args.simplification, args.duplication, out/'episodes'))
        finally:
            env.close()
        return summaries
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(pair, items):
            results.extend(result)
    (out/'summaries.json').write_text(json.dumps(results, indent=2))
    print('Finished', len(results), 'episodes;', sum(bool(r['error']) for r in results), 'errors', flush=True)


if __name__ == '__main__':
    main()
