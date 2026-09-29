"""Replay the expanded ScienceWorld initial-state intervention offline.

The expansion runner already archived the candidate and conditional bank for
each fixed-grid variation.  This script changes only the first-step readout
and replays the same no-imagination suffix for flat m=1, flat m=4, and grouped
provenance.  It therefore adds no API calls and keeps the intervention causal
at the public-state level.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from submission.interactive_counterfactual import _rows, grouped_decision, replay_branch


def run(m1, m4, output, simplification="easy", steps=12):
    m1 = Path(m1); m4 = Path(m4); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = sorted((m1 / "episodes").glob("*-flat-m1-trace.jsonl"))
    for flat1_path in traces:
        stem = flat1_path.name[:-len("-flat-m1-trace.jsonl")]
        if "-v" not in stem:
            continue
        task, variation = stem.rsplit("-v", 1)
        flat4_path = m4 / "episodes" / f"{stem}-flat-m4-trace.jsonl"
        no_path = m1 / "episodes" / f"{stem}-no_imagination-m1-trace.jsonl"
        if not flat4_path.exists() or not no_path.exists():
            continue
        flat1 = _rows(flat1_path); flat4 = _rows(flat4_path); no = _rows(no_path)
        if not flat1 or not flat4 or not no:
            continue
        first = flat1[0]
        grouped = grouped_decision(first["candidate"], first["bank"])
        choices = [
            ("flat", 1, flat1[0]["action"], flat1[0]["decision"].get("p_true")),
            ("flat", 4, flat4[0]["action"], flat4[0]["decision"].get("p_true")),
            ("provenance_grouped", 1, grouped["action"], grouped["p_true"]),
        ]
        suffix = [row["action"] for row in no[1:]]
        for method, duplication, action, p_true in choices:
            replay = replay_branch(task, int(variation), simplification, [], action, suffix, steps)
            rows.append({"task": task, "variation": int(variation), "step": 0,
                         "method": method, "duplication": duplication,
                         "action": action, "p_true": p_true,
                         "common_suffix_steps": len(suffix), **replay})
    (output / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    fields = ["task", "variation", "step", "method", "duplication", "action",
              "p_true", "common_suffix_steps", "final_score", "reward", "steps",
              "success", "terminal", "error"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "states": len({(x['task'], x['variation']) for x in rows}),
                      "output": str(output)}, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", default="results_submission/scienceworld_expand_m1")
    parser.add_argument("--m4", default="results_submission/scienceworld_expand_m4")
    parser.add_argument("--output", default="results_submission/report/scienceworld_expand_counterfactual")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    run(args.m1, args.m4, args.output, steps=args.steps)


if __name__ == "__main__":
    main()
