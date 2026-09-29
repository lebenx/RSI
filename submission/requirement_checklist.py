"""Render the numbered research requirements against the machine evidence audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


GROUPS = [
    (
        "1. Theory and mechanism",
        ["theory propositions"],
        "State the invariance and bias results as formal propositions; do not claim a universal transformer law.",
    ),
    (
        "2. Formal benchmark",
        [
            "benchmark scale/splits",
            "benchmark task families",
            "benchmark graph and public lineage contract",
            "benchmark lineage visualization",
        ],
        "Use the 5,000-task benchmark with private graph/lineage and filtered public views.",
    ),
    (
        "3. Baselines",
        ["eleven baselines"],
        "Report all eleven methods on the same frozen rollout bank and disclose the stipulated flat estimator.",
    ),
    (
        "4. Real-model diagnostics",
        ["DeepSeek audit", "DeepSeek three-seed MVP", "local Qwen audit", "ScienceWorld local Qwen short-horizon smoke", "local Qwen full-episode interactive pilot", "local Qwen expanded full-episode interactive pilot", "local Qwen cached provenance-value replay", "local Qwen second-task-family value control", "local Qwen expanded second-task-family pilot", "local Qwen CUDA inference probe"],
        "Describe prompt/order/position sensitivity; avoid a universal LLM behavioral claim.",
    ),
    (
        "5. Interactive agent validation",
        ["interactive metric coverage", "local Qwen interactive schema smoke", "ScienceWorld paired traces", "ScienceWorld fixed-grid expansion", "ScienceWorld refreshed DeepSeek comparison", "ScienceWorld fresh-key extension", "ScienceWorld fresh-key second-task extension", "ScienceWorld fresh-key fair LLM extension", "ScienceWorld fresh-key fair LLM second-task extension", "ScienceWorld refreshed second-task-family check", "ScienceWorld refreshed comparison table", "ScienceWorld consolidated interactive ledger", "ScienceWorld interactive main table", "local Qwen interactive main table", "cross-model interactive stratified table", "ScienceWorld expanded fresh-state replay", "ScienceWorld refreshed state-replay counterfactual", "ScienceWorld clean fixed-bank readout intervention", "ScienceWorld expanded four-step fixed-bank intervention", "ScienceWorld fixed-bank episode-cluster analysis", "ScienceWorld serialized-input integrity audit", "broad interactive success gain"],
        "Claim duplication-robust action consistency and exploratory planning signals; broad success improvement remains open.",
    ),
    (
        "6. Learned provenance recovery",
        ["learned provenance lexical audit", "learned provenance semantic-hard audit", "learned provenance identifiability null", "provenance recovery margin diagnostic", "provenance recovery breakdown/curve"],
        "Report lexical recovery as an upper bound and opaque multi-root recovery as the semantic limitation.",
    ),
    (
        "7. Scaling",
        ["scaling 1..64", "controlled task-cluster uncertainty", "controlled uncertainty figure", "independent-rollout value axis", "conditional-value variance reduction"],
        "Use the 1/2/4/8/16/32/64 curves and task-cluster intervals.",
    ),
    (
        "8. Ablations",
        ["ablation methods", "fixed snapshot state replay", "real-environment duplication counterfactual"],
        "Keep negative and null ablations visible; do not turn invariance into a planning guarantee.",
    ),
    (
        "9. Paper outputs",
        ["paper outputs", "reviewer evidence matrix", "secret hygiene"],
        "Use the generated report, draft, tables, figures, and evidence matrix as the reproducibility boundary.",
    ),
]


def render(audit_path: str | Path, output: str | Path) -> None:
    audit = json.loads(Path(audit_path).read_text())
    checks = {row["requirement"]: row for row in audit["checks"]}
    lines = [
        "# Submission requirement checklist",
        "",
        "Generated from `results_submission/report/evidence_audit.json`. A status is an evidence boundary, not a conference-readiness guarantee.",
        "",
        "| Requirement | Status | Evidence | Allowed paper wording |",
        "|---|---|---|---|",
    ]
    for title, names, wording in GROUPS:
        rows = [checks[name] for name in names]
        statuses = [row["status"] for row in rows]
        if "not_established" in statuses:
            status = "not_established"
        elif "missing" in statuses:
            status = "missing"
        elif "partial" in statuses:
            status = "partial"
        else:
            status = "verified"
        evidence = "; ".join(row["evidence"] for row in rows).replace("|", "\\|")
        lines.append(f"| {title} | **{status}** | `{evidence}` | {wording} |")
    lines += [
        "",
        f"Machine audit totals: {audit['verified_count']} verified, {audit['partial_count']} partial, {audit['not_established_count']} not established.",
        "",
        "The interactive success gate is intentionally not promoted to a positive claim: primary v7 flat/provenance success is tied, and the fixed-grid task-family changes are mixed.",
    ]
    Path(output).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="results_submission/report/evidence_audit.json")
    parser.add_argument("--output", default="paper/submission_checklist.md")
    args = parser.parse_args()
    render(args.audit, args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
