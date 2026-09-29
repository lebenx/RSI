"""Render an explicit paper claim boundary from the machine evidence audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CLAIMS = [
    ("Supported", "Shared-premise duplication can bias a stipulated flat estimator and change actions without new real observations.", ["theory propositions", "scaling 1..64", "controlled task-cluster uncertainty"]),
    ("Supported", "Provenance-preserving aggregation is invariant to exact descendant copies under the stated lineage and tie-breaking assumptions.", ["theory propositions", "ablation methods", "ScienceWorld expanded four-step fixed-bank intervention"]),
    ("Supported", "Independent conditional rollouts reduce value-estimation error while leaving real-evidence belief unchanged.", ["independent-rollout value axis", "conditional-value variance reduction"]),
    ("Supported with model scope", "DeepSeek and local Qwen responses show presentation-, prompt-, and objective-dependent duplication effects.", ["DeepSeek audit", "local Qwen audit", "local Qwen interactive main table"]),
    ("Exploratory", "On the archived CUDA Qwen ScienceWorld subset, algorithmic provenance-value improves paired reward/success relative to flat in both task-family strata.", ["local Qwen interactive main table", "local Qwen expanded second-task-family pilot"]),
    ("Exploratory", "A fresh six-pair CUDA-Qwen ScienceWorld expansion reproduces positive provenance-value reward contrasts and zero provenance duplication flips across plant and animal strata; its confidence intervals remain small-sample diagnostics.", ["local Qwen fresh CUDA paired expansion"]),
    ("Exploratory", "A held-out CUDA-Qwen expansion on variations 165--167 reproduces positive provenance-value reward contrasts and zero provenance duplication flips on five complete pairs; one malformed readout is excluded from paired inference.", ["local Qwen second fresh CUDA paired expansion"]),
    ("Limited", "A fair algorithmic flat-value control on four CUDA-Qwen pairs is null: provenance and duplicate-sensitive flat-value have identical reward and success in this conditional bank, showing that provenance preservation is not by itself a planning guarantee.", ["local Qwen fair algorithmic flat-value control"]),
    ("Exploratory", "A retry-validated held-out CUDA-Qwen expansion adds four complete ScienceWorld pairs: provenance-value improves the plant stratum by 46 reward points and 0.5 success, while the animal stratum is null; the result is task-family dependent and small-sample.", ["local Qwen retry-validated held-out expansion"]),
    ("Exploratory", "Across the consolidated 23-pair CUDA-Qwen table, provenance-value reward contrasts are positive in both task-family strata and provenance actions remain duplication-invariant; the table is stratified and is not a pooled superiority estimate.", ["local Qwen consolidated 23-pair table"]),
    ("Exploratory", "Across the larger 27-pair CUDA-Qwen table, provenance-value retains positive reward contrasts in both task-family strata, while success intervals remain small-sample and the analysis is not pooled as a universal planner claim.", ["local Qwen consolidated 27-pair table"]),
    ("Limited", "The 172--173 CUDA-Qwen expansion has zero episode successes in both methods but retains reward-level provenance gains in selected animal/plant states; this is a planning-failure diagnostic, not a success-rate claim.", ["local Qwen fourth fresh CUDA expansion"]),
    ("Limited", "An independent held-out CUDA-Qwen flat-value/provenance-value expansion on plant variations 174--176 is exactly null at the episode level, showing that provenance invariance does not guarantee a planning gain when candidate values have no decision margin.", ["local Qwen independent held-out null expansion"]),
    ("Limited", "A second independent held-out CUDA-Qwen expansion on find-animal variations 177--179 is also exactly null, confirming that the negative boundary is not specific to the plant task family.", ["local Qwen independent held-out animal null expansion"]),
    ("Supported with model scope", "A frozen local-Qwen semantic rollout prompt reduces compact-template value copying and flat-bank collapse, but its valid states still show no premise-dependent winner and four states fail the JSON contract; this repairs a measurement confound without establishing planning gains.", ["local Qwen semantic bank prompt intervention"]),
    ("Supported with model scope", "CUDA-Qwen reward effects vary across task family and fresh expansion; the non-pooled heterogeneity analysis reports this variation rather than assuming a universal effect.", ["local Qwen effect heterogeneity"]),
    ("Limited", "The public-observation ALFWorld comparison shows duplication-invariant provenance traces and lower aligned action sensitivity in the selected strata, but does not establish a general success-rate gain.", ["ALFWorld TextWorld paired interactive comparison"]),
    ("Supported with model scope", "A fair DeepSeek ALFWorld algorithmic control produces duplication-sensitive belief drift for flat_value (mean 0.218 at m=4) and zero drift for provenance_value, while the three-game reward and success contrasts are exactly null; this is evidence for estimator invariance, not an episode-level gain.", ["ALFWorld fair algorithmic flat-value control"]),
    ("Limited", "Learned provenance recovery can fail under opaque multi-root observations; downstream loss is governed by recovery error and action margin.", ["learned provenance semantic-hard audit", "provenance recovery margin diagnostic", "learned provenance identifiability null"]),
    ("Do not claim", "All language models over-count duplicates, or provenance universally improves interactive success.", ["broad interactive success gain"]),
    ("Do not claim", "The fixed-bank ScienceWorld readout experiment establishes confidence-mediated episode improvement.", ["ScienceWorld fixed-bank episode-cluster analysis"]),
]


def main() -> None:
    audit_path = ROOT / "results_submission/report/evidence_audit.json"
    audit = json.loads(audit_path.read_text())
    checks = {x["requirement"]: x for x in audit["checks"]}
    lines = [
        "# Paper claim ledger", "",
        "Generated by `python -m submission.claim_ledger` from the machine evidence audit.",
        "Use the wording below as the boundary for the abstract, conclusion, and rebuttal.", "",
        "| Boundary | Claim wording | Evidence status | Authoritative artifacts |",
        "|---|---|---|---|",
    ]
    for boundary, wording, names in CLAIMS:
        rows = [checks[name] for name in names]
        statuses = "; ".join(f"{r['requirement']}: {r['status']}" for r in rows)
        evidence = "; ".join(r["evidence"] for r in rows).replace("|", "\\|")
        lines.append(f"| **{boundary}** | {wording} | {statuses} | `{evidence}` |")
    lines += ["", f"Machine audit: {audit['verified_count']} verified, {audit['partial_count']} partial, {audit['not_established_count']} not established.", ""]
    (ROOT / "paper/claim_ledger.md").write_text("\n".join(lines))
    print("wrote paper/claim_ledger.md")


if __name__ == "__main__":
    main()
