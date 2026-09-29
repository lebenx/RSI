"""Generate compact paper tables and a figure inventory from frozen artifacts."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def f(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def ci(x: pd.Series, digits: int = 3) -> str:
    return f"{f(x['estimate'], digits)} [{f(x['ci95_low'], digits)}, {f(x['ci95_high'], digits)}]"


def main_table() -> list[str]:
    p = ROOT / "results_submission/report/main_table_ci.csv"
    d = pd.read_csv(p).sort_values(["budget", "method"])
    lines = [
        "### Controlled benchmark main table",
        "",
        "Entries are estimate [95% task-cluster bootstrap CI]; reward is higher-is-better, regret is lower-is-better, and flip rate measures action changes relative to duplication 1.",
        "",
        "| Budget | Method | Root confidence | Reward | Regret | Action flip |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, r in d.iterrows():
        lines.append("| {budget} | {method} | {root} | {reward} | {regret} | {flip} |".format(
            budget=int(r.budget), method=r.method,
            root=ci(pd.Series({"estimate": r.estimate_root_confidence, "ci95_low": r.ci95_low_root_confidence, "ci95_high": r.ci95_high_root_confidence}), 3),
            reward=ci(pd.Series({"estimate": r.estimate_reward, "ci95_low": r.ci95_low_reward, "ci95_high": r.ci95_high_reward}), 3),
            regret=ci(pd.Series({"estimate": r.estimate_regret, "ci95_low": r.ci95_low_regret, "ci95_high": r.ci95_high_regret}), 3),
            flip=ci(pd.Series({"estimate": r.estimate_action_flip_rate, "ci95_low": r.ci95_low_action_flip_rate, "ci95_high": r.ci95_high_action_flip_rate}), 3),
        ))
    lines += ["", "Source: `results_submission/report/main_table_ci.csv`.", ""]
    return lines


def endpoint_table() -> list[str]:
    d = pd.read_csv(ROOT / "results_submission/report/ablation_endpoint.csv")
    lines = [
        "### Component ablation endpoint",
        "",
        "| Method | Root confidence | Wrong-premise confidence | Reward | Regret | Action correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in d.iterrows():
        lines.append(f"| {r.method} | {f(r.root_confidence)} | {f(r.wrong_premise_confidence)} | {f(r.reward)} | {f(r.regret)} | {f(r.action_correct)} |")
    lines += ["", "Source: `results_submission/report/ablation_endpoint.csv`; full family/difficulty rows: `results_submission/ablations/ablation_summary.csv`.", ""]
    return lines


def qwen_table() -> list[str]:
    d = pd.read_csv(ROOT / "results_submission/report/local_qwen_interactive_expanded_table_v4/contrasts.csv")
    d = d[d.metric.isin(["success", "reward"])].sort_values(["task_family", "duplication", "metric"])
    lines = [
        "### CUDA-Qwen ScienceWorld paired contrasts",
        "",
        "Provenance-value minus flat; 27 complete paired episodes across find-plant and find-animal. Episode-success claims remain exploratory.",
        "",
        "| Family | Duplication | Metric | n | Contrast [95% CI] |",
        "|---|---:|---|---:|---:|",
    ]
    for _, r in d.iterrows():
        lines.append(f"| {r.task_family} | {int(r.duplication)} | {r.metric} | {int(r.n_episodes)} | {f(r.estimate)} [{f(r.ci95_low)}, {f(r.ci95_high)}] |")
    lines += ["", "Source: `results_submission/report/local_qwen_interactive_expanded_table_v4/contrasts.csv`.", ""]
    return lines


def fair_table() -> list[str]:
    d = pd.read_csv(ROOT / "results_submission/report/local_qwen_fair_flatvalue_168_169/bootstrap.csv")
    d = d[d.metric.isin(["success", "reward", "final_score"])].sort_values(["task_family", "duplication", "metric"])
    lines = [
        "### Fair algorithmic flat-value control",
        "",
        "Provenance-value minus duplicate-sensitive flat-value on the same conditional-value bank; all paired contrasts are exactly zero in this 4-episode control.",
        "",
        "| Family | Duplication | Metric | n | Contrast [95% CI] |",
        "|---|---:|---|---:|---:|",
    ]
    for _, r in d.iterrows():
        lines.append(f"| {r.task_family} | {int(r.duplication)} | {r.metric} | {int(r.n_episodes)} | {f(r.estimate)} [{f(r.ci95_low)}, {f(r.ci95_high)}] |")
    lines += ["", "Source: `results_submission/report/local_qwen_fair_flatvalue_168_169/bootstrap.csv`.", ""]
    return lines


def heterogeneity_table() -> list[str]:
    d = pd.read_csv(ROOT / "results_submission/report/local_qwen_effect_heterogeneity/heterogeneity.csv")
    d = d[d.metric == "reward"].sort_values(["expansion", "task_family", "duplication"])
    lines = [
        "### Qwen effect heterogeneity",
        "",
        "Reward contrasts are shown separately by fresh expansion, task family, and duplication; intervals resample paired episode variations and are not pooled across protocols.",
        "",
        "| Expansion | Family | Duplication | n | Provenance minus flat [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in d.iterrows():
        lines.append(f"| {r.expansion} | {r.task_family} | {int(r.duplication)} | {int(r.n_episodes)} | {f(r.estimate)} [{f(r.ci95_low)}, {f(r.ci95_high)}] |")
    lines += ["", "Source: `results_submission/report/local_qwen_effect_heterogeneity/heterogeneity.csv`.", ""]
    return lines


def qwen_retry_table() -> list[str]:
    d = pd.read_csv(ROOT / "results_submission/report/local_qwen_expansion_170_171_retry/bootstrap.csv")
    d = d[d.metric.isin(["success", "reward", "repeated_no_visible_change"])].sort_values(["task_family", "duplication", "metric"])
    lines = [
        "### Held-out CUDA-Qwen retry-validated expansion",
        "",
        "A fresh four-episode paired expansion after one explicit local-only candidate-schema retry. The first malformed completions remain archived in `results_submission/scienceworld_local_qwen_expansion_170_171_m1` and `_m4`; no API run is repaired.",
        "",
        "| Family | Duplication | Metric | n | Provenance minus flat [95% CI] |",
        "|---|---:|---|---:|---:|",
    ]
    for _, r in d.iterrows():
        lines.append(f"| {r.task_family} | {int(r.duplication)} | {r.metric} | {int(r.n_episodes)} | {f(r.estimate)} [{f(r.ci95_low)}, {f(r.ci95_high)}] |")
    lines += ["", "Source: `results_submission/report/local_qwen_expansion_170_171_retry/{metadata,summary,paired,bootstrap,action_consistency,raw_errors}.csv`.", ""]
    return lines


def qwen_heldout_null_table() -> list[str]:
    plant = pd.read_csv(ROOT / "results_submission/report/local_qwen_heldout174_176/bootstrap.csv")
    animal = pd.read_csv(ROOT / "results_submission/report/local_qwen_heldout177_179/bootstrap.csv")
    plant["task_family"] = "find-plant"
    animal["task_family"] = "find-animal"
    d = pd.concat([plant, animal], ignore_index=True)
    d = d[d.metric.isin(["success", "reward", "final_score", "steps"])].sort_values(["duplication", "metric"])
    lines = [
        "### Independent held-out CUDA-Qwen null control",
        "",
        "Three new find-plant variations (174--176) and three find-animal variations (177--179) under the fair algorithmic flat-value/provenance-value protocol. All paired contrasts are zero; this is a negative boundary check, not a pooled planner estimate.",
        "",
        "| Family | Duplication | Metric | n | Provenance minus flat [95% CI] |",
        "|---|---:|---|---:|---:|",
    ]
    for _, r in d.iterrows():
        lines.append(f"| {r.task_family} | {int(r.duplication)} | {r.metric} | {int(r.n_episodes)} | {f(r.estimate)} [{f(r.ci95_low)}, {f(r.ci95_high)}] |")
    lines += ["", "Source: `results_submission/report/local_qwen_heldout174_176/{metadata,summary,paired,bootstrap,action_consistency}.csv`; `local_qwen_heldout177_179/{metadata,summary,paired,bootstrap,action_consistency}.csv`.", ""]
    return lines


def deepseek_table() -> list[str]:
    d = pd.read_csv(ROOT / "results_submission/deepseek_mvp_20260929/summary.csv")
    d = d[d.method.isin(["flat", "provenance_value", "exact_dedup", "source_grouping"])]
    lines = [
        "### DeepSeek three-seed MVP",
        "",
        "| Method | Duplication | p(true) | Belief drift | Action flip | Reward | Decision regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in d.sort_values(["duplication", "method"]).iterrows():
        lines.append(f"| {r.method} | {int(r.duplication)} | {f(r.p_true)} | {f(r.belief_drift)} | {f(r.action_flip_rate)} | {f(r.reward)} | {f(r.decision_regret)} |")
    lines += ["", "Source: `results_submission/deepseek_mvp_20260929/summary.csv`.", ""]
    return lines


def figures() -> list[str]:
    candidates = [
        ("Controlled scaling", "results_submission/scienceworld_frozen_readout_first4_20260929/curve.png"),
        ("Controlled episode curves", "results_submission/scienceworld_frozen_readout_first4_20260929/episode_curves.png"),
        ("Qwen paired expansion", "results_submission/report/local_qwen_interactive_expanded_table/curve.png"),
        ("Qwen fair flat-value control", "results_submission/report/local_qwen_fair_flatvalue_168_169/curve.png"),
        ("Qwen effect heterogeneity", "results_submission/report/local_qwen_effect_heterogeneity/forest.png"),
        ("Qwen independent held-out null", "results_submission/report/local_qwen_heldout174_176/curve.png"),
        ("DeepSeek MVP scaling", "results_submission/deepseek_mvp_20260929/curve.png"),
        ("ALFWorld public-observation comparison", "results_submission/report/alfworld_interactive/curve.png"),
    ]
    lines = ["### Figure inventory", "", "| Figure | Artifact | Status |", "|---|---|---|"]
    for name, path in candidates:
        status = "available" if (ROOT / path).exists() else "missing"
        lines.append(f"| {name} | `{path}` | {status} |")
    lines += ["", "Qualitative failure and counterexample traces: `results_submission/report/qualitative_examples.md`.", ""]
    return lines


def run(output: str = "paper/tables.md", figure_output: str = "paper/figure_manifest.md") -> None:
    text = [
        "# Generated paper tables",
        "",
        "Generated by `python -m submission.paper_tables` from frozen report artifacts. Values are not re-estimated here.",
        "",
    ]
    for section in (main_table(), endpoint_table(), qwen_table(), heterogeneity_table(), fair_table(), qwen_retry_table(), qwen_heldout_null_table(), deepseek_table(), figures()):
        text.extend(section)
    out = ROOT / output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(text) + "\n")
    fig = ROOT / figure_output
    fig.write_text("# Figure manifest\n\n" + "\n".join(figures()) + "\n")
    print(f"wrote {output} and {figure_output}")


if __name__ == "__main__":
    run()
