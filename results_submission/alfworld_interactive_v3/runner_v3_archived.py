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
import os
import random
import time
from pathlib import Path
from typing import Any

from submission.llm import JsonLLM, probability, tokens


VERSION = "alfworld-textworld-provenance-v3"
TASK_TYPES = (1, 2, 3, 4, 5, 6)

CANDIDATE_SYSTEM = """You are an action planner in ALFWorld TextWorld.
Use only the supplied observation, task text, public action history, and
admissible commands. Return JSON only. Select exactly four distinct command
indices from the supplied list and state one uncertain binary premise about a
world property or whether a target is currently accessible. The premise is a
belief question, not a preference about actions. Estimate its probability from
real observations only. Do not repeat an action that made no visible progress;
prefer the next required task operation. Parse the exact target object(s) and
receptacle(s) from "Your task is" and never manipulate an unrelated distractor.
If a target is not visible, choose a legal discovery or navigation command
toward it instead of picking another object. The four returned ids MUST be exactly
0, 1, 2, 3 in that order and are local ids for your four selected commands,
not indices from the admissible command list. JSON:
{"premise":"...","p_true":0.5,"actions":[{"id":0,"action":"exact command"},{"id":1,"action":"exact command"},{"id":2,"action":"exact command"},{"id":3,"action":"exact command"}]}"""

ROLLOUT_SYSTEM = """You are simulating conditional ALFWorld action outcomes.
The observation is real; each hypothesis is stipulated and supplies no new
real evidence. Return exactly one JSON object with eight rows, one for every
(action_id, hypothesis_value) pair in requested_conditions. Keep predictions
short and numeric. JSON:
{"rollouts":[{"action_id":0,"hypothesis_value":false,
"predicted_final_score":0.1,"success_probability":0.2,
"imagined_outcome":"short"}, ...]}"""

FLAT_SYSTEM = """You are the flat evidence reader for an ALFWorld planner.
Use the real observation and the supplied conditional imagined rollouts. The
listed descendants are presented as independent evidence for the premise, even
when they share a sample_id. Choose the command with the highest expected
real-world final score, obeying the exact target object/receptacle in the task,
avoiding distractors and repeated no-change commands. Return JSON only:
{"action_id":0,"p_true":0.5,"success_probability":0.5,"reason":"short"}"""


def load_config(config_path: Path, data_dir: Path, task_type: int, step_limit: int) -> dict[str, Any]:
    import yaml

    os.environ["ALFWORLD_DATA"] = str(data_dir)
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["eval_id_data_path"] = "$ALFWORLD_DATA/json_2.1.1/valid_seen"
    config["dataset"]["eval_ood_data_path"] = "$ALFWORLD_DATA/json_2.1.1/valid_unseen"
    config["dataset"]["num_eval_games"] = 1
    config["env"]["task_types"] = [int(task_type)]
    config["general"]["training_method"] = "dagger"
    config["dagger"]["training"]["max_nb_steps_per_episode"] = int(step_limit)
    config["env"]["domain_randomization"] = False
    return config


def manager_and_env(config: dict[str, Any], game_file: str):
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


def visible_payload(observation: str, commands: list[str], history: list[dict[str, Any]], remaining: int) -> dict[str, Any]:
    # Keep the interface public and bounded.  The task description is present
    # in the initial TextWorld observation; no game file or state internals are
    # included.
    return {
        "observation": observation[:12000],
        "admissible_commands": commands[:100],
        "history": [{"action": h["action"], "observation": h["observation"][:1200]} for h in history[-8:]],
        "remaining_steps": int(remaining),
    }


def validate_candidate(obj: dict[str, Any], commands: list[str]) -> dict[str, Any]:
    rows = obj.get("actions")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("candidate action list must have four rows")
    allowed = list(dict.fromkeys(commands))
    if len(allowed) < 4:
        raise ValueError("fewer than four admissible commands")
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
    for index in range(4):
        action = selected[index] if index < len(selected) else ""
        if action not in allowed or action in used:
            action = next(x for x in allowed if x not in used)
            repaired += 1
        used.add(action)
        fixed.append({"id": index, "action": action})
    premise = str(obj.get("premise", "")).strip()
    if not premise:
        raise ValueError("missing premise")
    return {"premise": premise, "p_true": probability(obj["p_true"]),
            "actions": fixed, "candidate_repaired": repaired}


def validate_bank(obj: dict[str, Any], state_id: str) -> list[dict[str, Any]]:
    rows = obj.get("rollouts")
    expected = {(a, h) for a in range(4) for h in (False, True)}
    if not isinstance(rows, list) or {(int(r.get("action_id", -1)), bool(r.get("hypothesis_value"))) for r in rows} != expected or len(rows) != 8:
        raise ValueError("conditional bank must contain eight unique pairs")
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
    q = candidate["p_true"]
    values: dict[int, float] = {}
    successes: dict[int, float] = {}
    for action in range(4):
        by_h = {h: [r for r in bank if r["action_id"] == action and r["hypothesis_value"] == h] for h in (False, True)}
        values[action] = sum((1 - q if not h else q) * sum(r["predicted_final_score"] for r in by_h[h]) / len(by_h[h]) for h in (False, True))
        successes[action] = sum((1 - q if not h else q) * sum(r["success_probability"] for r in by_h[h]) / len(by_h[h]) for h in (False, True))
    chosen = max(values, key=lambda a: (values[a], -a))
    return {"action_id": chosen, "p_true": q, "success_probability": successes[chosen],
            "action_values": values, "success_values": successes, "reason": "unique lineage conditional mixture"}


def request_bundle(llm: JsonLLM, state_id: str, visible: dict[str, Any], commands: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    candidate_obj, candidate_entry = llm.ask(state_id + "-candidate", CANDIDATE_SYSTEM,
                                             {**visible, "admissible_commands": commands}, 850)
    candidate = validate_candidate(candidate_obj, commands)
    requested = [{"action_id": a, "hypothesis_value": h} for a in range(4) for h in (False, True)]
    bank_obj, bank_entry = llm.ask(state_id + "-bank", ROLLOUT_SYSTEM,
                                   {**visible, "premise": candidate["premise"], "p_true": candidate["p_true"],
                                    "actions": candidate["actions"], "requested_conditions": requested}, 1500)
    bank = validate_bank(bank_obj, state_id)
    return candidate, bank, tokens(candidate_entry) + tokens(bank_entry)


def run_episode(config_path: Path, data_dir: Path, game_file: str, task_type: int, method: str,
                duplication: int, steps: int, out: Path, llm: JsonLLM) -> dict[str, Any]:
    config = load_config(config_path, data_dir, task_type, steps)
    manager, env = manager_and_env(config, game_file)
    game_id = hashlib.sha256(game_file.encode()).hexdigest()[:12]
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
    history: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    confidence: list[float] = []
    total_reward = 0.0
    errors = []
    started = time.monotonic()
    try:
        for step in range(steps):
            visible = visible_payload(observation, commands, history, steps - step)
            state_id = f"{episode_id}-s{hashlib.sha256(json.dumps(visible, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:18]}"
            try:
                candidate, bank, generation_tokens = request_bundle(llm, state_id, visible, commands)
                confidence.append(candidate["p_true"])
                presented = [dict(r) for r in bank]
                if method == "flat":
                    presented = [dict(r) for r in bank for _ in range(duplication if r["hypothesis_value"] else 1)]
                    random.Random(1701 + step).shuffle(presented)
                    readout_obj, readout_entry = llm.ask(
                        f"{state_id}-flat-m{duplication}", FLAT_SYSTEM,
                        {**visible, "premise": candidate["premise"], "evidence_only_probability": candidate["p_true"],
                         "candidates": candidate["actions"], "imagined_rollouts": presented}, 600)
                    action_id = int(readout_obj["action_id"])
                    p_true = probability(readout_obj["p_true"])
                    success_probability = probability(readout_obj.get("success_probability", 0.5))
                    decision = {"action_id": action_id, "p_true": p_true, "success_probability": success_probability,
                                "reason": str(readout_obj.get("reason", ""))}
                    charged_tokens = generation_tokens + tokens(readout_entry)
                elif method == "provenance_value":
                    decision = grouped_value(candidate, bank)
                    action_id = int(decision["action_id"])
                    p_true = candidate["p_true"]
                    success_probability = decision["success_probability"]
                    charged_tokens = generation_tokens
                else:
                    raise ValueError(method)
                if action_id not in range(4):
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
            reward = scalar(scores)
            total_reward += reward
            unchanged = observation == before_observation and tuple(commands) == before_commands
            repeated = any(h["action"] == action and h["unchanged_visible_state"] for h in history)
            row = {"step": step, "action": action, "action_id": action_id,
                   "candidate": candidate, "decision": decision,
                   "duplication": duplication, "method": method,
                   "predicted_success_probability": success_probability,
                   "charged_tokens": charged_tokens, "reward": reward,
                   "observation": observation, "won": won, "done": bool(done),
                   "unchanged_visible_state": unchanged,
                   "repeated_no_visible_change": bool(unchanged and repeated)}
            history.append(row)
            trace_rows.append(row)
            if done or won:
                break
    finally:
        env.close()
    trace_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in trace_rows) + ("\n" if trace_rows else ""))
    summary = {"protocol": VERSION, "task_type": int(task_type), "game_file": game_file,
               "game_id": game_id, "method": method, "duplication": int(duplication),
               "step_limit": int(steps), "success": int(bool(won)), "reward": total_reward,
               "final_score": total_reward, "steps": len(trace_rows),
               "unnecessary_actions": sum(int(x["repeated_no_visible_change"]) for x in trace_rows),
               "charged_tokens": sum(int(x["charged_tokens"]) for x in trace_rows),
               "root_confidence_first": confidence[0] if confidence else None,
               "root_confidence_last": confidence[-1] if confidence else None,
               "root_confidence_delta": (confidence[-1] - confidence[0]) if confidence else None,
               "confidence_steps": len(confidence), "errors": errors,
               "trace_file": str(trace_path), "runtime_seconds": time.monotonic() - started}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("task_type", "game_id", "method", "duplication", "success", "reward", "steps", "errors")}, ensure_ascii=False), flush=True)
    return summary


def write_tables(out: Path, rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = ["task_type", "game_id", "method", "duplication", "success", "reward", "final_score", "steps", "unnecessary_actions", "charged_tokens", "root_confidence_first", "root_confidence_last", "root_confidence_delta", "confidence_steps", "errors", "trace_file"]
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
        for metric in ("success", "reward", "final_score", "steps", "unnecessary_actions", "charged_tokens", "root_confidence_delta"):
            contrasts.append({"task_type": task_type, "game_id": game_id, "duplication": duplication,
                              "metric": metric, "provenance_value_minus_flat": float(methods["provenance_value"][metric]) - float(methods["flat"][metric])})
    with (out / "paired_contrasts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "game_id", "duplication", "metric", "provenance_value_minus_flat"])
        writer.writeheader(); writer.writerows(contrasts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results_submission/alfworld_interactive_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("external/alfworld_data"))
    parser.add_argument("--config", type=Path, default=Path("external/alfworld_src/configs/eval_config.yaml"))
    parser.add_argument("--task-types", default="1,2,3")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--duplications", default="1,4")
    parser.add_argument("--methods", default="flat,provenance_value")
    args = parser.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY required for uncached ALFWorld calls")
    task_types = [int(x) for x in args.task_types.split(",") if x]
    duplications = [int(x) for x in args.duplications.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    args.output.mkdir(parents=True, exist_ok=True)
    llm = JsonLLM(args.output / "requests", model="deepseek-chat")
    specs = []
    for task_type in task_types:
        games = discover_games(args.config, args.data_dir, task_type, args.steps)
        if not games:
            raise RuntimeError(f"no valid_seen game for task type {task_type}")
        specs.append({"task_type": task_type, "game_file": games[0], "available_games": len(games)})
    (args.output / "episode_manifest.json").write_text(json.dumps({"protocol": VERSION, "specs": specs,
        "task_types": task_types, "duplications": duplications, "methods": methods,
        "data_dir": str(args.data_dir), "steps": args.steps}, indent=2))
    rows = []
    for spec in specs:
        for method in methods:
            for duplication in duplications:
                rows.append(run_episode(args.config, args.data_dir, spec["game_file"], spec["task_type"],
                                        method, duplication, args.steps, args.output, llm))
    write_tables(args.output, rows)
    metadata = {"protocol": VERSION, "model": "deepseek-chat", "task_types": task_types,
                "duplications": duplications, "methods": methods, "episodes": len(specs),
                "valid_rows": len(rows), "api_network_calls": llm.network_calls, "cache_hits": llm.cache_hits,
                "uses_hidden_state_in_prompt": False, "uses_gold_plan": False,
                "interpretation": "small exploratory ALFWorld TextWorld paired evaluation; no universal LLM claim"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
