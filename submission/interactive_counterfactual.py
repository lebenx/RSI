"""Replay a duplication intervention in a real ScienceWorld state.

The candidate actions and conditional rollouts come from an archived DeepSeek
trace.  We replay the public action prefix to the same simulator state, change
only the readout-selected first action, and apply the archived no-imagination
suffix as a common continuation.  This is a paired state-replay diagnostic,
not a replacement for a broad closed-loop success estimate.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def grouped_decision(candidate, bank):
    """Select from unique sample lineage using conditional value mixing."""
    unique = {}
    for row in bank:
        old = unique.setdefault(row["sample_id"], row)
        if old != row:
            raise ValueError("conflicting sample identity")
    q = float(candidate["p_true"])
    values = {}
    success = {}
    for action_id in range(4):
        value = 0.0
        chance = 0.0
        for hypothesis, weight in ((False, 1.0 - q), (True, q)):
            group = [r for r in unique.values()
                     if r["action_id"] == action_id
                     and r["hypothesis_value"] == hypothesis]
            if not group:
                raise ValueError("missing conditional coverage")
            value += weight * sum(float(r["predicted_final_score"])
                                  for r in group) / len(group)
            chance += weight * sum(float(r["success_probability"])
                                   for r in group) / len(group)
        values[action_id] = value
        success[action_id] = chance
    chosen = max(values, key=lambda action: (values[action], -action))
    return {"action_id": chosen, "action": candidate["actions"][chosen]["action"],
            "p_true": q, "success_probability": success[chosen],
            "action_values": values, "unique_samples": len(unique)}


def _latest_flat(summary, task, variation, step, duplication):
    matches = [x for x in summary
               if x.get("method") == "flat"
               and x.get("task") == task
               and int(x.get("variation")) == int(variation)
               and int(x.get("step")) == int(step)
               and int(x.get("duplication")) == int(duplication)
               and "action_id" in x]
    if not matches:
        raise ValueError(f"missing flat readout: {task} v{variation} s{step} m{duplication}")
    return matches[-1]


def replay_branch(task, variation, simplification, prefix, selected, suffix, step_limit):
    # Import lazily so the analysis and unit tests do not require Java/py4j.
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv(envStepLimit=step_limit + 5)
    try:
        env.load(task, variation, simplification, generateGoldPath=False)
        observation, info = env.reset()
        history = []
        for action in prefix:
            observation, reward, done, info = env.step(action)
            history.append({"action": action, "reward": reward,
                            "score": info["score"], "done": bool(done)})
            if done:
                return {"error": "prefix terminated", "history": history,
                        "final_score": info["score"], "success": int(info["score"] >= 100)}
        observation, reward, done, info = env.step(selected)
        history.append({"action": selected, "reward": reward,
                        "score": info["score"], "done": bool(done)})
        for action in suffix:
            if done:
                break
            observation, reward, done, info = env.step(action)
            history.append({"action": action, "reward": reward,
                            "score": info["score"], "done": bool(done)})
        return {"history": history, "final_score": info["score"],
                "success": int(info["score"] >= 100),
                "reward": sum(float(row["reward"]) for row in history),
                "steps": len(history), "terminal": bool(done)}
    finally:
        env.close()


def run(output, summary_path, cases, simplification="easy", step_limit=12):
    summary = json.loads(Path(summary_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for task, variation, step in cases:
        flat_trace = Path(glob.glob(
            f"results_submission/scienceworld_v7_dev8_m1/episodes/"
            f"{task}-v{variation}-flat-m1-trace.jsonl")[0])
        no_trace = Path(glob.glob(
            f"results_submission/scienceworld_v7_dev8_m1/episodes/"
            f"{task}-v{variation}-no_imagination-m1-trace.jsonl")[0])
        flat_rows = _rows(flat_trace)
        no_rows = _rows(no_trace)
        state = flat_rows[step]
        prefix = [row["action"] for row in flat_rows[:step]]
        suffix = [row["action"] for row in no_rows[step + 1:]]
        grouped = grouped_decision(state["candidate"], state["bank"])
        choices = [("provenance_grouped", "all", grouped["action_id"],
                    grouped["action"], grouped["p_true"], grouped["success_probability"])]
        for duplication in (1, 4, 8):
            row = _latest_flat(summary, task, variation, step, duplication)
            choices.append(("flat", duplication, row["action_id"], row["action"],
                            row.get("p_true"), row.get("success_probability")))
        for method, duplication, action_id, action, p_true, predicted_success in choices:
            replay = replay_branch(task, variation, simplification, prefix, action,
                                   suffix, step_limit)
            records.append({"task": task, "variation": variation, "step": step,
                            "method": method, "duplication": duplication,
                            "action_id": action_id, "action": action,
                            "p_true": p_true,
                            "predicted_success_probability": predicted_success,
                            "prefix": prefix, "common_suffix": suffix,
                            **replay})
    (output / "summary.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    fields = ["task", "variation", "step", "method", "duplication", "action_id",
              "action", "p_true", "predicted_success_probability", "final_score",
              "reward", "steps", "success", "terminal", "error"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results_submission/interactive_counterfactual/summary.json")
    parser.add_argument("--output", default="results_submission/interactive_counterfactual_replay")
    parser.add_argument("--simplification", default="easy")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cases", nargs="*", default=["find-plant:162:1", "find-plant:158:1"])
    args = parser.parse_args()
    cases = []
    for item in args.cases:
        task, variation, step = item.rsplit(":", 2)
        cases.append((task, int(variation), int(step)))
    records = run(args.output, args.summary, cases, args.simplification, args.steps)
    print("wrote", len(records), "paired replay rows")


if __name__ == "__main__":
    main()
