"""Expand the clean fixed-bank ScienceWorld readout intervention to 24 states.

The states are the first post-action public states from the archived v8/v9
plant/animal runs (variations 150--161).  Candidate and rollout inputs are
read from their serialized API request bodies and checked before any new
readout call.  The imported runner then performs only readout calls, replays
all candidate actions, and reports immediate outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from submission.scienceworld_frozen_readout import (
    BUDGETS, VARIANTS, archived_pre_action_visible, run_calls, replay, summarize,
)
from submission.scienceworld_runner import READOUT_SYSTEM, unique_provenance_bank


ROOT = Path("results_submission")
ARCHIVE_RUNS = {
    ("find-plant", "early"): ROOT / "scienceworld_recharged_findplant_m1",
    ("find-animal", "early"): ROOT / "scienceworld_recharged_v9_findanimal_m1",
}


def row_for(task: str, variation: int):
    if variation <= 155:
        directory = ARCHIVE_RUNS[(task, "early")]
        rows = json.loads((directory / "summaries.json").read_text())
        candidates = [r for r in rows if r.get("task") == task and int(r.get("variation", -1)) == variation
                      and r.get("method") == "flat" and int(r.get("duplication", -1)) == 1
                      and r.get("error") is None]
    else:
        from submission.recharged_llm_extension_summary import load_rows
        frame = load_rows(task)
        candidates = [r.to_dict() for _, r in frame.iterrows()
                      if int(r.variation) == variation and r.experiment_method == "flat"
                      and int(r.duplication) == 1 and r.error is None]
    if len(candidates) != 1:
        raise ValueError(f"expected one archived row for {task}:{variation}, got {len(candidates)}")
    return candidates[0]


def prepare(output: Path, start: int = 150, stop: int = 162, all_steps: bool = False, max_step: int | None = None) -> None:
    if max_step is not None and max_step <= 0:
        raise ValueError('max_step must be positive')
    states = []
    for task in ("find-plant", "find-animal"):
        for variation in range(start, stop):
            row = row_for(task, variation)
            trace_path = Path(row["trace_file"])
            trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
            if not trace:
                raise ValueError(f"trace too short: {trace_path}")
            selected = range(len(trace)) if all_steps else (1,)
            if max_step is not None:
                selected = [step for step in selected if step < max_step]
            for step in selected:
                state = trace[step]
                visible = archived_pre_action_visible(str(trace_path), state)
                states.append({
                    "state_id": f"{task}-v{variation}-s{step}",
                    "task": task,
                    "variation": variation,
                    "step": step,
                    "trace_file": str(trace_path),
                    "simplification": row["simplification"],
                    "step_limit": int(row["step_limit"]),
                    "prefix": [entry["action"] for entry in trace[:step]],
                    "visible": visible,
                    "input_source": "serialized original candidate/bank/flat API request, equality checked",
                    "candidate": state["candidate"],
                    "bank": unique_provenance_bank(state["bank"]),
                })
    manifest = {
        "protocol": "frozen-bank-identical-prompt-v3-expanded-pre-action",
        "states": len(states),
        "input_integrity": "history contains only steps strictly before the frozen decision; original candidate/bank/readout inputs agree",
        "selection": f"{'all pre-action trace steps' if all_steps else 'step 1'} of archived flat m1 runs, both tasks, variations {start}..{stop - 1}; no outcome filter",
        "max_step_exclusive": max_step,
        "duplications": list(BUDGETS),
        "api_repetitions": 3,
        "variants": list(VARIANTS),
        "planned_api_calls": len(states) * 3 * len(VARIANTS),
        "model": "deepseek-chat",
        "temperature": 0,
        "system_prompt": READOUT_SYSTEM,
        "api_seed_supported": False,
        "schedule_seed": 20260929,
        "control": "m1_repeat request has byte-identical body to m1; independent request ID",
        "methods": ["flat", "same_prompt_dedup", "provenance_value"],
        "dedup_readout": "use the independent m1_repeat response at each m; structural input invariance",
        "provenance_value": "deterministic unique-sample belief mixture; no LLM belief override treated as model behavior",
        "outcome": "immediate simulator reward and best-of-four immediate regret, not episode reward/success",
        "root_truth_labels": "unavailable; report belief drift, not root calibration or wrong-belief rate",
        "known_limit": "selected archived task families; readout intervention is causal for the fixed public states, not a broad closed-loop success estimate",
    }
    output.mkdir(parents=True, exist_ok=True)
    # Check both files before writing either; never replace the design of a
    # completed run when a caller changes the selection arguments.
    artifacts = [(output / 'manifest.json', manifest), (output / 'states.json', states)]
    for path, data in artifacts:
        if path.exists() and json.loads(path.read_text()) != data:
            raise ValueError(f'frozen artifact differs: {path}; use a new output directory')
    for path, data in artifacts:
        if not path.exists():
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps({"states": len(states), "planned_api_calls": manifest["planned_api_calls"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "replay", "summarize"))
    parser.add_argument("--output", type=Path, default=ROOT / "scienceworld_frozen_readout_expanded_20260929")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=150)
    parser.add_argument("--stop", type=int, default=162)
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.start, args.stop, args.all_steps, args.max_step)
    elif args.command == "run":
        run_calls(args.output, args.workers, args.retry_failed)
    elif args.command == "replay":
        replay(args.output)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
