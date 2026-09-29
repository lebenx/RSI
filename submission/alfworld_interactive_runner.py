"""Small paired ALFWorld-TextWorld experiment for provenance-aware planning.

The runner uses the public TextWorld observation and admissible-command API.
For each state, one DeepSeek candidate/conditional-rollout bank is archived and
reused across duplication levels.  ``flat`` sends repeated descendants to a
model readout; ``provenance_value`` collapses sample IDs and performs the
conditional-value mixture algorithmically.  The environment is never queried
for hidden state or a gold plan.

This is an exploratory interactive protocol.  It is intentionally small and
keeps complete request/trace files so that malformed API responses are visible.
Run with the Python 3.11 research venv after installing ALFWorld data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from submission.llm import JsonLLM, probability, tokens


VERSION = "alfworld-textworld-provenance-v4-task-persistent"
TASK_TYPES = (1, 2, 3, 4, 5, 6)

CANDIDATE_SYSTEM = """You are an action planner in ALFWorld TextWorld.
Use only the supplied observation, task text, public action history, and
admissible commands. Return JSON only. Select candidate_count distinct commands
from the supplied list and state one uncertain binary premise about a
world property or whether a target is currently accessible. The premise is a
belief question, not a preference about actions. Estimate its probability from
real observations only. Do not repeat an action that made no visible progress;
prefer the next required task operation. Parse the exact target object(s) and
receptacle(s) from "Your task is" and never manipulate an unrelated distractor.
If a target is not visible, choose a legal discovery or navigation command
toward it instead of picking another object. Rank commands from best to worst
for the task; id 0 must be your best direct action. Return local ids 0 through
candidate_count-1, not indices from the original admissible command list.
Estimate the chance of completing the exact task after your best action.
JSON: {"premise":"...","p_true":0.5,"direct_success_probability":0.5,
"actions":[{"id":0,"action":"exact command"}, ...]}"""

ROLLOUT_SYSTEM = """You are simulating conditional ALFWorld action outcomes.
The observation is real; each hypothesis is stipulated and supplies no new
real evidence. Predict the chance of completing the exact task within the
remaining budget, including sensible follow-up actions. Return one row for every
(action_id, hypothesis_value) pair in requested_conditions. Keep predictions
short and numeric. JSON:
{"rollouts":[{"action_id":0,"hypothesis_value":false,
"predicted_final_score":0.1,"success_probability":0.2,
"imagined_outcome":"short"}, ...]}"""

ROLLOUT_RETRY_SYSTEM = """Return JSON only, with exactly the requested rollout
rows. Copy every action_id and hypothesis_value from requested_conditions,
including JSON booleans. Do not add prose or markdown. For each pair include
predicted_final_score in [-1,1], success_probability in [0,1], and
imagined_outcome as a short string. JSON:
{"rollouts":[{"action_id":0,"hypothesis_value":false,
"predicted_final_score":0.1,"success_probability":0.2,
"imagined_outcome":"short"}, ...]}"""

FLAT_SYSTEM = """You plan in ALFWorld using the supplied real observations and
conditional imagined rollouts. Choose the command with the highest expected
real-world final score, obeying the exact target object/receptacle in the task,
avoiding distractors and repeated no-change commands. Return JSON only:
{"action_id":0,"p_true":0.5,"success_probability":0.5,"reason":"short"}"""


def load_config(config_path: Path, data_dir: Path, task_type: int, step_limit: int) -> dict[str, Any]:
    import yaml

    os.environ["ALFWORLD_DATA"] = str(data_dir)
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["eval_id_data_path"] = "$ALFWORLD_DATA/json_2.1.1/valid_seen"
    config["dataset"]["eval_ood_data_path"] = "$ALFWORLD_DATA/json_2.1.1/valid_unseen"
    config["dataset"]["num_eval_games"] = -1
    config["env"]["task_types"] = [int(task_type)]
    config["general"]["training_method"] = "dagger"
    config["dagger"]["training"]["max_nb_steps_per_episode"] = int(step_limit)
    config["env"]["domain_randomization"] = False
    return config


def patch_textworld_eval_namespace() -> None:
    """Make TextWorld grammar eval work on current Python runtimes.

    TextWorld's EvalSymbol historically used ``locals().update`` before
    calling eval.  CPython 3.14 no longer exposes those injected locals to
    eval, producing ``NameError: r`` while resetting otherwise valid ALFWorld
    games.  Passing the grammar variables explicitly preserves the intended
    behavior and keeps the compatibility fix in the reproducible runner.
    """
    import textworld.envs.pddl.textgen as textgen

    if getattr(textgen.EvalSymbol, "_jev_explicit_namespace", False):
        return

    def derive(self, context=None):
        context = context or self.context
        value = eval(self.expression, {}, dict(context["variables"]))
        return [textgen.TerminalSymbol(value)]

    textgen.EvalSymbol.derive = derive
    textgen.EvalSymbol._jev_explicit_namespace = True


def manager_and_env(config: dict[str, Any], game_file: str):
    patch_textworld_eval_namespace()
    from alfworld.agents.environment import get_environment

    cls = get_environment("AlfredTWEnv")
    manager = cls(config, train_eval="eval_in_distribution")
    manager.game_files = [game_file]
    manager.num_games = 1
    return manager, manager.init_env(batch_size=1)


def discover_games(config_path: Path, data_dir: Path, task_type: int, step_limit: int) -> list[str]:
    from alfworld.agents.environment import get_environment

    config = load_config(config_path, data_dir, task_type, step_limit)
    manager = get_environment("AlfredTWEnv")(config, train_eval="eval_in_distribution")
    return sorted(str(x) for x in manager.game_files)


def scalar(x: Any) -> float:
    if isinstance(x, (list, tuple)):
        return float(x[0]) if x else 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def normalize_info(obs: Any, info: dict[str, Any]) -> tuple[str, list[str], bool]:
    text = str(obs[0] if isinstance(obs, (list, tuple)) else obs)
    commands = info.get("admissible_commands", [])
    if isinstance(commands, (list, tuple)) and commands and isinstance(commands[0], (list, tuple)):
        commands = commands[0]
    commands = [str(x) for x in commands]
    won = info.get("won", [False])
    if isinstance(won, (list, tuple)):
        won = won[0] if won else False
    return text, commands, bool(won)


def visible_payload(initial_observation: str, observation: str, commands: list[str],
                    history: list[dict[str, Any]], remaining: int) -> dict[str, Any]:
    match = re.search(r'Your task is to:\s*(.+)', initial_observation)
    if not match:
        raise ValueError('initial observation has no task description')
    # Task and initial public room persist even after the current feedback changes.
    # No game path, expert plan, hidden state, or current-step outcome is exposed.
    return {
        "task": match.group(1).strip(),
        "initial_observation": initial_observation,
        "observation": observation,
        "admissible_commands": list(commands),
        "candidate_count": min(4, len(commands)),
        "history": [{"step": h['step'], "action": h["action"], "observation": h["observation"]} for h in history],
        "remaining_steps": int(remaining),
    }


def validate_candidate(obj: dict[str, Any], commands: list[str]) -> dict[str, Any]:
    rows = obj.get("actions")
    allowed = list(dict.fromkeys(commands))
    count = min(4, len(allowed))
    if not isinstance(rows, list) or not count:
        raise ValueError('wrong candidate count')
    used: set[str] = set()
    fixed = []
    repaired = 0
    # Preserve selected command order even when a model echoes the original
    # admissible-list index in ``id`` instead of the requested local id.
    selected = []
    for row in rows:
        action = str(row.get("action", ""))
        if action and action not in selected:
            selected.append(action)
    for index in range(count):
        action = selected[index] if index < len(selected) else ""
        if action not in allowed or action in used:
            replacement = next((x for x in allowed if x not in used), None)
            if replacement is None:
                raise ValueError('no distinct admissible candidate command')
            action = replacement
            repaired += 1
        used.add(action)
        fixed.append({"id": index, "action": action})
    premise = str(obj.get("premise", "")).strip()
    if not premise:
        raise ValueError("missing premise")
    return {"premise": premise, "p_true": probability(obj["p_true"]),
            "direct_success_probability": probability(obj['direct_success_probability']),
            "ids_normalized": [r.get('id') for r in rows] != list(range(count)),
            "actions": fixed, "candidate_repaired": repaired}


def validate_bank(obj: dict[str, Any], state_id: str, count: int = 4) -> list[dict[str, Any]]:
    rows = obj.get("rollouts")
    expected = {(a, h) for a in range(count) for h in (False, True)}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('wrong conditional bank size')
    if any(type(r.get('hypothesis_value')) is not bool or type(r.get('action_id')) is not int for r in rows):
        raise ValueError('condition IDs must be integer and JSON boolean')
    if {(r['action_id'], r['hypothesis_value']) for r in rows} != expected:
        raise ValueError('incomplete or repeated condition pairs')
    out = []
    for r in rows:
        value = float(r["predicted_final_score"])
        if not -1 <= value <= 1:
            raise ValueError("predicted_final_score outside [-1,1]")
        p = probability(r["success_probability"])
        action = int(r["action_id"])
        hyp = bool(r["hypothesis_value"])
        out.append({"action_id": action, "hypothesis_value": hyp,
                    "predicted_final_score": value, "success_probability": p,
                    "imagined_outcome": str(r.get("imagined_outcome", "")),
                    "sample_id": f"{state_id}-a{action}-h{int(hyp)}",
                    "parent_id": f"{state_id}-premise", "source_type": "imagined_condition"})
    return sorted(out, key=lambda x: (x["action_id"], x["hypothesis_value"]))


def grouped_value(candidate: dict[str, Any], bank: list[dict[str, Any]]) -> dict[str, Any]:
    unique = {}
    for r in bank:
        if unique.setdefault(r['sample_id'], r) != r:
            raise ValueError('conflicting sample identity')
    bank = list(unique.values())
    q = candidate["p_true"]
    values: dict[int, float] = {}
    successes: dict[int, float] = {}
    for action in range(len(candidate['actions'])):
        by_h = {h: [r for r in bank if r["action_id"] == action and r["hypothesis_value"] == h] for h in (False, True)}
        values[action] = sum((1 - q if not h else q) * sum(r["predicted_final_score"] for r in by_h[h]) / len(by_h[h]) for h in (False, True))
        successes[action] = sum((1 - q if not h else q) * sum(r["success_probability"] for r in by_h[h]) / len(by_h[h]) for h in (False, True))
    chosen = max(values, key=lambda a: (values[a], -a))
    return {"action_id": chosen, "p_true": q, "success_probability": successes[chosen],
            "action_values": values, "success_values": successes, "reason": "unique lineage conditional mixture"}


def grouped_success(candidate: dict[str, Any], bank: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the action by the same unique-lineage mixture, using success probability.

    This is a declared objective ablation: it changes only the final action
    score, while preserving the candidate premise, lineage grouping, and
    conditional bank. It is useful for separating provenance from a reward
    versus success objective mismatch in interactive tasks.
    """
    decision = grouped_value(candidate, bank)
    chosen = max(decision["success_values"], key=lambda a: (decision["success_values"][a], -a))
    return {"action_id": chosen, "p_true": candidate["p_true"],
            "success_probability": decision["success_values"][chosen],
            "action_values": decision["action_values"], "success_values": decision["success_values"],
            "reason": "unique lineage success mixture"}


def flat_value(candidate: dict[str, Any], bank: list[dict[str, Any]], duplication: int) -> dict[str, Any]:
    """Fair algorithmic flat control for the real-environment protocol.

    The candidate and conditional bank are identical to ``grouped_value``.  The
    only intervention is that repeated true-premise descendants are retained in
    the conditional average and shift the premise weight with the declared
    duplicate-count pseudo-likelihood.  This mirrors the controlled benchmark
    and makes the provenance-vs-flat comparison independent of a second LLM
    reader.
    """
    q = candidate["p_true"]
    if duplication > 1:
        logit = math.log(max(1e-9, q) / max(1e-9, 1.0 - q))
        q = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0,
            logit + (duplication - 1) * math.log(1.5)))))
    values: dict[int, float] = {}
    successes: dict[int, float] = {}
    for action in range(len(candidate['actions'])):
        by_h = {h: [r for r in bank if r["action_id"] == action and r["hypothesis_value"] == h]
                for h in (False, True)}
        if any(not by_h[h] for h in (False, True)):
            raise ValueError('missing conditional coverage')
        values[action] = sum((1 - q if not h else q) *
                             sum(r['predicted_final_score'] for r in by_h[h]) / len(by_h[h])
                             for h in (False, True))
        successes[action] = sum((1 - q if not h else q) *
                                sum(r['success_probability'] for r in by_h[h]) / len(by_h[h])
                                for h in (False, True))
    chosen = max(values, key=lambda a: (values[a], -a))
    return {'action_id': chosen, 'p_true': q,
            'success_probability': successes[chosen],
            'action_values': values, 'success_values': successes,
            'reason': 'flat duplicate-count pseudo-likelihood'}


def request_bundle(llm, state_id: str, visible: dict[str, Any], commands: list[str],
                   method: str) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    candidate_obj, candidate_entry = llm.ask(state_id + "-candidate", CANDIDATE_SYSTEM,
                                             {**visible, "admissible_commands": commands}, 850)
    candidate = validate_candidate(candidate_obj, commands)
    if method == 'no_imagination':
        return candidate, [], tokens(candidate_entry)
    requested = [{"action_id": a, "hypothesis_value": h} for a in range(len(candidate['actions'])) for h in (False, True)]
    payload = {**visible, "premise": candidate["premise"], "p_true": candidate["p_true"],
               "actions": candidate["actions"], "requested_conditions": requested}
    bank_obj, bank_entry = llm.ask(state_id + "-bank", ROLLOUT_SYSTEM, payload, 1500)
    try:
        bank = validate_bank(bank_obj, state_id, len(candidate['actions']))
    except ValueError:
        # Local Qwen occasionally truncates the eight-row contract. Preserve
        # the malformed completion and issue one deterministic terse retry;
        # DeepSeek failures remain visible rather than silently retried.
        if not getattr(llm, 'is_local_cuda', False):
            raise
        bank_obj, bank_entry = llm.ask(state_id + "-bank-retry", ROLLOUT_RETRY_SYSTEM, payload, 1800)
        bank = validate_bank(bank_obj, state_id, len(candidate['actions']))
    return candidate, bank, tokens(candidate_entry) + tokens(bank_entry)


def run_episode(config_path: Path, data_dir: Path, game_file: str, task_type: int, method: str,
                duplication: int, steps: int, out: Path, llm: JsonLLM) -> dict[str, Any]:
    config = load_config(config_path, data_dir, task_type, steps)
    manager, env = manager_and_env(config, game_file)
    game_id = hashlib.sha256('/'.join(Path(game_file).parts[-3:]).encode()).hexdigest()[:12]
    episode_id = f"t{task_type}-g{game_id}"
    trace_path = out / "traces" / f"{episode_id}-{method}-m{duplication}.jsonl"
    summary_path = out / "summaries" / f"{episode_id}-{method}-m{duplication}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        env.close()
        return json.loads(summary_path.read_text())
    obs_raw, info = env.reset()
    observation, commands, won = normalize_info(obs_raw, info)
    initial_observation = observation
    history: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    confidence: list[float] = []
    total_reward = 0.0
    final_score = 0.0
    done = False
    errors = []
    started = time.monotonic()
    try:
        for step in range(steps):
            visible = visible_payload(initial_observation, observation, commands, history, steps - step)
            state_id = f"{episode_id}-s{hashlib.sha256(json.dumps(visible, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:18]}"
            try:
                candidate, bank, generation_tokens = request_bundle(llm, state_id, visible, commands, method)
                confidence.append(candidate["p_true"])
                presented = [dict(r) for r in bank for _ in range(duplication if r["hypothesis_value"] else 1)]
                # Keep relative base order fixed, so m=1 flat and exact_dedup
                # really share identical readout requests.
                if method in ('flat', 'exact_dedup'):
                    readout_bank = bank if method == 'exact_dedup' else presented
                    readout_m = 1 if method == 'exact_dedup' else duplication
                    readout_obj, readout_entry = llm.ask(
                        f"{state_id}-reader-m{readout_m}", FLAT_SYSTEM,
                        {**visible, "premise": candidate["premise"], "evidence_only_probability": candidate["p_true"],
                         "candidates": candidate["actions"], "imagined_rollouts": readout_bank}, 400)
                    action_id = readout_obj["action_id"]
                    if type(action_id) is not int:
                        raise ValueError('action ID must be a JSON integer')
                    p_true = probability(readout_obj["p_true"])
                    success_probability = probability(readout_obj.get("success_probability", 0.5))
                    decision = {"action_id": action_id, "p_true": p_true, "success_probability": success_probability,
                                "reason": str(readout_obj.get("reason", ""))}
                    charged_tokens = generation_tokens + tokens(readout_entry)
                elif method in ("flat_value", "provenance_value", "provenance_success"):
                    if method == "flat_value":
                        decision = flat_value(candidate, presented, duplication)
                    elif method == "provenance_value":
                        decision = grouped_value(candidate, presented)
                    else:
                        decision = grouped_success(candidate, presented)
                    action_id = int(decision["action_id"])
                    p_true = candidate["p_true"]
                    success_probability = decision["success_probability"]
                    charged_tokens = generation_tokens
                elif method == 'no_imagination':
                    action_id = 0
                    decision = dict(action_id=0, p_true=candidate['p_true'],
                        success_probability=candidate['direct_success_probability'], reason='best direct candidate')
                    success_probability = decision['success_probability']
                    charged_tokens = generation_tokens
                else:
                    raise ValueError(method)
                if action_id not in range(len(candidate['actions'])):
                    raise ValueError("readout action id outside candidate range")
                action = candidate["actions"][action_id]["action"]
            except Exception as exc:
                errors.append({"step": step, "type": type(exc).__name__, "message": str(exc)[:300]})
                break
            before_observation = observation
            before_commands = tuple(commands)
            next_raw, scores, done_raw, next_info = env.step([action])
            done = done_raw[0] if isinstance(done_raw, (list, tuple)) else bool(done_raw)
            observation, commands, won = normalize_info(next_raw, next_info)
            cumulative_score = scalar(scores)
            reward = cumulative_score - final_score
            final_score = cumulative_score
            total_reward += reward
            unchanged = observation == before_observation and tuple(commands) == before_commands
            repeated = any(h["action"] == action and h['before_observation'] == before_observation for h in history)
            row = {"step": step, "state_id": state_id, "visible": visible,
                   "action": action, "action_id": action_id, 'bank': presented,
                   "candidate": candidate, "decision": decision,
                   'belief_drift': decision['p_true'] - candidate['p_true'],
                   "duplication": duplication, "method": method,
                   "predicted_success_probability": success_probability,
                   "charged_tokens": charged_tokens, "reward": reward,
                   "observation": observation, "won": won, "done": bool(done),
                   'score': final_score, 'before_observation': before_observation,
                   "unchanged_visible_state": unchanged,
                   "repeated_no_visible_change": bool(unchanged and repeated)}
            history.append(row)
            trace_rows.append(row)
            with trace_path.open('a') as trace_file:
                trace_file.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f't{task_type} {game_id} {method} m{duplication} step={step+1} {action} won={won}', flush=True)
            if done or won:
                break
    finally:
        env.close()
    trace_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in trace_rows) + ("\n" if trace_rows else ""))
    summary = {"protocol": VERSION, "task_type": int(task_type), "game_file": game_file,
               "game_id": game_id, "method": method, "duplication": int(duplication),
               "step_limit": int(steps), "success": int(bool(won)), "reward": total_reward,
               "final_score": final_score, "steps": len(trace_rows),
               "unnecessary_actions": sum(int(x["repeated_no_visible_change"]) for x in trace_rows),
               "charged_tokens": sum(int(x["charged_tokens"]) for x in trace_rows),
               "root_confidence_first": confidence[0] if confidence else None,
               "root_confidence_last": confidence[-1] if confidence else None,
               # Premises can change across steps: a first-last difference is
               # NOT a root-belief update. Report same-state readout drift.
               "mean_belief_drift": sum(r['belief_drift'] for r in trace_rows)/len(trace_rows) if trace_rows else None,
               "initial_success_probability": trace_rows[0]['decision']['success_probability'] if trace_rows else None,
               "calibration_brier": (trace_rows[0]['decision']['success_probability']-won)**2 if trace_rows else None,
               'terminal': bool(done), 'completed_without_error': not errors,
               "confidence_steps": len(confidence), "errors": errors,
               "trace_file": str(trace_path), "runtime_seconds": time.monotonic() - started}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("task_type", "game_id", "method", "duplication", "success", "reward", "steps", "errors")}, ensure_ascii=False), flush=True)
    return summary


def write_tables(out: Path, rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = ["task_type", "game_id", "method", "duplication", "success", "reward", "final_score", "steps", "unnecessary_actions", "charged_tokens", "root_confidence_first", "root_confidence_last", "mean_belief_drift", "initial_success_probability", "calibration_brier", "confidence_steps", "errors", "trace_file"]
    with (out / "rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict)) else row.get(k) for k in fields})
    # Simple paired contrasts, preserving task strata.
    grouped: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["task_type"]), row["game_id"], int(row["duplication"])), {})[row["method"]] = row
    contrasts = []
    for (task_type, game_id, duplication), methods in sorted(grouped.items()):
        if "flat" not in methods or "provenance_value" not in methods:
            continue
        for metric in ("success", "reward", "final_score", "steps", "unnecessary_actions", "charged_tokens", "mean_belief_drift", "calibration_brier"):
            left, right = methods["provenance_value"].get(metric), methods["flat"].get(metric)
            if left is None or right is None:
                continue
            contrasts.append({"task_type": task_type, "game_id": game_id, "duplication": duplication,
                              "metric": metric, "provenance_value_minus_flat": float(left) - float(right)})
    with (out / "paired_contrasts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "game_id", "duplication", "metric", "provenance_value_minus_flat"])
        writer.writeheader(); writer.writerows(contrasts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results_submission/alfworld_interactive_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("external/alfworld_data"))
    parser.add_argument("--config", type=Path, default=Path("external/alfworld_src/configs/eval_config.yaml"))
    parser.add_argument("--task-types", default="1,2,3")
    parser.add_argument("--games-per-type", type=int, default=1,
                        help="number of deterministic valid_seen games per task type")
    parser.add_argument("--game-offset", type=int, default=0,
                        help="zero-based offset into sorted valid_seen games")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--duplications", default="1,4")
    parser.add_argument("--methods", default="flat,provenance_value")
    parser.add_argument("--backend", choices=("deepseek", "cuda_worker"), default="deepseek")
    parser.add_argument("--model", default="/home/xhj/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5")
    parser.add_argument("--cuda-python", default="/home/xhj/miniconda3/bin/python")
    args = parser.parse_args()
    task_types = [int(x) for x in args.task_types.split(",") if x]
    duplications = [int(x) for x in args.duplications.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == 'deepseek':
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise SystemExit("DEEPSEEK_API_KEY required for uncached ALFWorld calls")
        llm = JsonLLM(args.output / "requests", model="deepseek-chat")
    else:
        if os.environ.get('LOCAL_QWEN_DEVICE', 'cuda') != 'cuda':
            raise SystemExit('LOCAL_QWEN_DEVICE=cuda is required; CPU fallback is disabled')
        from submission.cuda_json_worker import CudaProcessLLM
        llm = CudaProcessLLM(args.output / 'requests', args.cuda_python, args.model)
    if args.games_per_type < 1 or args.game_offset < 0:
        raise SystemExit("--games-per-type must be positive and --game-offset nonnegative")
    specs = []
    for task_type in task_types:
        games = discover_games(args.config, args.data_dir, task_type, args.steps)
        if not games:
            raise RuntimeError(f"no valid_seen game for task type {task_type}")
        selected = games[args.game_offset:args.game_offset + args.games_per_type]
        if not selected:
            raise RuntimeError(f"game offset {args.game_offset} is outside task type {task_type} ({len(games)} games)")
        specs.extend({"task_type": task_type, "game_file": game_file,
                      "available_games": len(games), "selection_index": args.game_offset + i}
                     for i, game_file in enumerate(selected))
    (args.output / "episode_manifest.json").write_text(json.dumps({"protocol": VERSION, "specs": specs,
        "task_types": task_types, "games_per_type": args.games_per_type,
        "game_offset": args.game_offset, "duplications": duplications, "methods": methods,
        "data_dir": str(args.data_dir), "steps": args.steps}, indent=2))
    rows = []
    for spec in specs:
        for method in methods:
            for duplication in duplications:
                rows.append(run_episode(args.config, args.data_dir, spec["game_file"], spec["task_type"],
                                        method, duplication, args.steps, args.output, llm))
    if hasattr(llm, 'close'):
        llm.close()
    write_tables(args.output, rows)
    metadata = {"protocol": VERSION, "model": ("deepseek-chat" if args.backend == 'deepseek' else args.model),
                "backend": args.backend, "task_types": task_types,
                "games_per_type": args.games_per_type, "game_offset": args.game_offset,
                "duplications": duplications, "methods": methods, "episodes": len(specs),
                "valid_rows": len(rows), "api_network_calls": llm.network_calls,
                "api_network_retries": getattr(llm, "network_retries", 0), "cache_hits": llm.cache_hits,
                "inference": getattr(llm, 'metadata', None),
                "uses_hidden_state_in_prompt": False, "uses_gold_plan": False,
                "interpretation": "small exploratory ALFWorld TextWorld paired evaluation; no universal LLM claim"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
