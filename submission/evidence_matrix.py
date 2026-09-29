"""Render the machine audit as a reviewer-facing evidence matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


INTERPRETATION = {
    "theory propositions": "Formal mechanism, proof sketches, provenance-recovery margin bound, and an observability limit are present.",
    "benchmark scale/splits": "Benchmark scale and leakage-aware splits are verified.",
    "benchmark task families": "All five requested families are represented.",
    "benchmark lineage visualization": "A generated lineage graph example is archived with the benchmark.",
    "benchmark graph and public lineage contract": "Private rows contain graph/lineage; public rows omit hidden graph fields and retain visible rollouts.",
    "eleven baselines": "The controlled evaluator exposes the required methods.",
    "DeepSeek audit": "Model-specific real-LLM diagnostic is archived.",
    "DeepSeek three-seed MVP": "A 3-seed, 1/2/4/8 fixed-bank API diagnostic archives flat, dedup, grouping, and provenance-value outputs plus an identical-input stochasticity control.",
    "local Qwen audit": "Second-model diagnostic is archived.",
    "scaling 1..64": "All seven rollout budgets are present.",
    "controlled task-cluster uncertainty": "Task-level 95% bootstrap intervals are archived for the frozen held-out bank.",
    "controlled uncertainty figure": "Scaling curves include task-cluster 95% error bars for the main methods.",
    "main table with confidence intervals": "The main duplication table is machine-generated with confidence intervals for each reported metric.",
    "independent-rollout value axis": "Independent conditional samples are reported in a separate budget axis with paired reward/regret contrasts.",
    "conditional-value variance reduction": "Held-out lineage directly measures conditional-value MSE under fresh samples versus pure copies.",
    "ablation methods": "All seven component ablations are present.",
    "learned provenance lexical audit": "Lexical result is an upper-bound sanity check.",
    "learned provenance semantic-hard audit": "Opaque multi-root recovery exposes the semantic limitation.",
    "learned provenance identifiability null": "Fixed-predictor test-label permutations quantify the finite-sample chance F1 baseline for opaque recovery.",
    "provenance recovery margin diagnostic": "Margin-binned downstream analysis connects recovery error to the Proposition 6 action-stability bound.",
    "provenance recovery breakdown/curve": "Family/difficulty and noise downstream artifacts are generated.",
    "ScienceWorld paired traces": "Primary fair interactive traces and paired statistics are archived.",
    "interactive metric coverage": "Interactive summaries expose success, reward, regret proxy, steps, tokens, calibration, and action consistency.",
    "local Qwen interactive schema smoke": "One compact paired episode plus a three-variation six-row grid are archived as feasibility diagnostics and excluded from planner claims.",
    "ScienceWorld local Qwen short-horizon smoke": "Four valid flat/provenance rows establish second-model schema/feasibility only; the two-step horizon excludes episode-success claims.",
    "local Qwen full-episode interactive pilot": "Twelve valid CUDA local-Qwen episode rows complete a balanced m=1/4 plant pilot on three variations; it is explicitly small, separate from DeepSeek, and excluded from confirmatory success claims.",
    "local Qwen expanded full-episode interactive pilot": "Twenty valid CUDA local-Qwen rows cover five official plant variations and five paired episodes; raw candidate-schema failures are retained separately and excluded from paired summaries.",
    "local Qwen cached provenance-value replay": "A byte-identical cached-request replay isolates the deterministic core aggregator on five paired plant episodes with zero new model calls.",
    "local Qwen second-task-family value control": "One complete find-animal episode supplies a separate task-family control for flat versus provenance-value on CUDA; it remains exploratory.",
    "local Qwen expanded second-task-family pilot": "Seventeen valid rows and three complete find-animal pairs are archived; seven schema failures and incomplete blocks remain explicit and excluded from paired estimates.",
    "local Qwen interactive main table": "A stratified 32-row, eight-pair legacy table aligns success, reward, regret proxy, unnecessary actions, calibration Brier, steps, and tokens across plant and animal families.",
    "local Qwen fresh CUDA paired expansion": "A fresh six-pair CUDA-Qwen expansion covers three new plant and three new animal variations with zero errors; it reports paired reward/success and duplication action consistency while keeping episode-success claims exploratory.",
    "local Qwen second fresh CUDA paired expansion": "A held-out five-pair CUDA-Qwen expansion on variations 165--167 reports positive task-family reward contrasts and zero provenance duplication flips; one malformed readout is retained and excluded.",
    "local Qwen fair algorithmic flat-value control": "A four-pair CUDA-Qwen control holds the conditional-value objective fixed and compares duplicate-sensitive flat-value belief updates with provenance-value; this bank is a null planning control with zero paired reward differences.",
    "local Qwen retry-validated held-out expansion": "A fresh four-pair CUDA-Qwen expansion records one local-only candidate-schema retry policy, zero retained errors, a positive plant-stratum paired contrast, and a null animal-stratum control; all results remain exploratory.",
    "local Qwen consolidated 23-pair table": "A machine-generated 95-row, 23-pair CUDA-Qwen table joins the legacy pilot with three fresh expansions while preserving plant/animal strata and planner_claim=false.",
    "local Qwen consolidated 27-pair table": "A machine-generated 111-row, 27-pair CUDA-Qwen table adds a fourth fresh expansion; it keeps plant/animal strata separate and planner_claim=false.",
    "local Qwen fourth fresh CUDA expansion": "A 16-row, four-pair CUDA-Qwen expansion on variations 172--173 completes with zero retained errors under the explicit local candidate-schema retry; plant and animal outcomes remain exploratory failure/planning diagnostics.",
    "local Qwen independent held-out null expansion": "A fresh 12-row, three-pair CUDA-Qwen flat-value/provenance-value comparison on variations 174--176 is error-free and exactly null in reward, success, steps, and aligned actions; it is retained as an independent negative boundary rather than pooled as a universal planner result.",
    "local Qwen independent held-out animal null expansion": "A second fresh 12-row, three-pair CUDA-Qwen flat-value/provenance-value comparison on find-animal variations 177--179 is error-free and exactly null in reward, success, steps, and aligned actions; it supplies a separate task-family negative control.",
    "local Qwen semantic bank prompt intervention": "A frozen 12-state prompt intervention reduces exact template-value copying from 1.000 to 0.000 and constant-bank rate from 1.000 to 0.375 among eight valid semantic completions; four completions fail schema validation and no premise-dependent winner appears, so this is interface evidence rather than planning evidence.",
    "local Qwen effect heterogeneity": "A 48-row, non-pooled expansion-by-family heterogeneity table and forest plot reports paired reward, success, and unnecessary-action contrasts with episode-level bootstrap intervals.",
    "cross-model interactive stratified table": "A non-pooled 104-row table, including 32 ALFWorld public-observation rows, displays DeepSeek and Qwen effects by protocol, task family, objective, and duplication; it is a heterogeneity display rather than a superiority meta-analysis.",
    "ALFWorld fair algorithmic flat-value control": "A new DeepSeek ALFWorld batch holds candidate and conditional-bank generation fixed, comparing duplicate-sensitive flat_value with provenance_value; belief drift appears only in flat at duplication 4 while reward and success are null in this small batch.",
    "local Qwen CUDA inference probe": "The local-Qwen path loaded the archived model on CUDA and generated a probe response; this verifies runtime placement, not planner quality.",
    "ScienceWorld contiguous extension": "Partial contiguous extension is archived; API transport failures are retained and excluded from claims.",
    "ScienceWorld refreshed DeepSeek comparison": "A fresh six-variation API comparison is archived, including negative planning results and paired action consistency; it does not establish broad success improvement.",
    "ScienceWorld refreshed state-replay counterfactual": "A common-continuation replay isolates first-step readout multiplicity and shows an episode-level consequence without adding API calls.",
    "ScienceWorld refreshed second-task-family check": "A second public task family is archived as a negative external-validity control with retained interface errors.",
    "ScienceWorld refreshed comparison table": "One machine-generated table aligns root-confidence change, paired action flips, success, and reward across both refreshed task families.",
    "ScienceWorld expanded fresh-state replay": "A no-new-API paired intervention covers 12 public states and bootstraps grouped provenance versus flat duplication-4 outcomes under a common continuation.",
    "ScienceWorld fresh-key extension": "A separate 24-row DeepSeek v9 extension has no transport failures and reports a null success-aware provenance contrast on six additional plant variations.",
    "ScienceWorld fresh-key second-task extension": "A matched 24-row animal extension supplies a negative external-validity control with the same public-action protocol and archived action/reward contrasts.",
    "ScienceWorld fresh-key fair LLM extension": "A 24-row fresh-key plant extension uses the same LLM provenance readout as flat with reused candidate/bank requests; it is a fair negative planning control.",
    "ScienceWorld fresh-key fair LLM second-task extension": "A matched 24-row animal extension repeats the fair LLM-readout comparison and supplies a negative external-validity control.",
    "ScienceWorld consolidated interactive ledger": "A read-only 96-row ledger joins the refreshed LLM-readout comparison and fresh-key success-aware extension while retaining protocol and task-family strata.",
    "ScienceWorld interactive main table": "A machine-generated 32-row summary and 56-row paired-contrast table aligns real-environment metrics across three explicitly labeled protocols.",
    "real-environment duplication counterfactual": "Legacy trace-derived replay is retained for auditability; input-integrity findings withdraw its clean causal claim. The serialized-input intervention is authoritative.",
    "grouped-value interactive exploratory run": "Four-task full-episode grouped-value diagnostic is verified.",
    "broad interactive success gain": "Keep this claim out of the paper: broad success improvement is not established.",
    "fixed snapshot state replay": "Corrected state replay is archived.",
    "ScienceWorld clean fixed-bank readout intervention": "A pre-action fixed-bank API intervention reports belief drift, action flips, and one-step reward with 180 valid readouts and no episode-success claim.",
    "ScienceWorld expanded four-step fixed-bank intervention": "A preregistered 96-state intervention supplies 1,440 valid readouts and 384 public-state-matched candidate replays; it tests confidence/action/reward consequences through four decisions without claiming episode success.",
    "ScienceWorld fixed-bank episode-cluster analysis": "The expanded intervention is audited at the whole task/variation episode level; zero joint probability/action changes keep confidence mediation unestablished, and all six harmful immediate-reward flips had unchanged reported probability.",
    "ScienceWorld serialized-input integrity audit": "Original serialized API inputs are separated from mutable trace views; the clean intervention has no current/future outcome in its evidence.",
    "paper outputs": "Draft, generated main/ablation/interactive tables, figure inventory, report, and qualitative examples exist.",
    "reproducibility manifest": "SHA256 hashes, environment versions, and regeneration commands cover the frozen paper/report artifacts.",
    "reviewer evidence matrix": "This matrix is generated from the machine audit.",
    "secret hygiene": "No API secret is present in source or paper files.",
}


def render(audit_path, output):
    audit = json.loads(Path(audit_path).read_text())
    lines = [
        "# Evidence matrix",
        "",
        "This file is generated by `python -m submission.evidence_matrix` from `results_submission/report/evidence_audit.json`.",
        "Statuses are evidence boundaries, not claims of conference acceptance.",
        "",
        "| Claim or deliverable | Status | Authoritative evidence | Reviewer interpretation |",
        "|---|---|---|---|",
    ]
    for check in audit["checks"]:
        name = check["requirement"]
        evidence = check["evidence"].replace("|", "\\|")
        interpretation = INTERPRETATION.get(name, check.get("note", ""))
        lines.append(f"| {name} | **{check['status']}** | `{evidence}` | {interpretation} |")
    lines += [
        "",
        f"Current count: {audit['verified_count']} verified, {audit['partial_count']} partial, {audit['not_established_count']} not established.",
    ]
    Path(output).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="results_submission/report/evidence_audit.json")
    parser.add_argument("--output", default="paper/evidence_matrix.md")
    args = parser.parse_args()
    render(args.audit, args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
