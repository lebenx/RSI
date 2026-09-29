"""Build a submission PDF from the current paper artifacts."""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = PAPER / "submission.tex"
PDF = PAPER / "submission.pdf"

def esc_plain(text: str) -> str:
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "–": "--", "—": "---", "−": "-", "…": r"\ldots{}",
        "≤": r"$\le$", "≥": r"$\ge$", "→": r"$\rightarrow$",
        "±": r"$\pm$", "×": r"$\times$", "≈": r"$\approx$",
        "α": r"$\alpha$", "β": r"$\beta$", "θ": r"$\theta$",
        "’": "'", "“": "'", "”": "'",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text

def esc(text: str) -> str:
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    tick = chr(96)
    chunks = re.split("(" + tick + "[^" + tick + "]*" + tick + r"|\*\*[^*]+\*\*)", text)
    out = []
    for chunk in chunks:
        if chunk.startswith(tick) and chunk.endswith(tick):
            out.append(r"\texttt{\detokenize{" + chunk[1:-1] + "}}")
        elif chunk.startswith("**") and chunk.endswith("**"):
            out.append(r"\textbf{" + esc_plain(chunk[2:-2]) + "}")
        else:
            out.append(esc_plain(chunk))
    return "".join(out)

def paragraph(lines):
    return esc(" ".join(x.strip() for x in lines if x.strip()))

def convert_draft():
    lines = (PAPER / "draft.md").read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("# ").strip()
    body = []
    i = 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("## Abstract"):
            acc = []
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                if lines[i].strip():
                    acc.append(lines[i])
                i += 1
            body.append("\\begin{abstract}\n" + paragraph(acc) + "\n\\end{abstract}")
            continue
        if line.startswith("## "):
            body.append("\\section{" + esc(line[3:].strip()) + "}")
            i += 1
            continue
        if line.startswith(">"):
            acc = []
            while i < len(lines) and lines[i].startswith(">"):
                acc.append(lines[i].lstrip("> "))
                i += 1
            body.append("\\begin{quote}\n" + paragraph(acc) + "\n\\end{quote}")
            continue
        if line.startswith(("- ", "* ")):
            items = []
            while i < len(lines) and lines[i].startswith(("- ", "* ")):
                items.append("\\item " + esc(lines[i][2:].strip()))
                i += 1
            body.append("\\begin{itemize}[leftmargin=1.5em]\n" + "\n".join(items) + "\n\\end{itemize}")
            continue
        acc = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", ">", "- ", "* ")):
            acc.append(lines[i])
            i += 1
        body.append(paragraph(acc))
    return title, "\n\n".join(body)

def markdown_table(lines, heading):
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return ""
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("### "):
            break
        if line.strip().startswith("|") and "---" not in line:
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    if len(rows) < 2:
        return ""
    n = len(rows[0])
    spec = "Y" + "r" * (n - 1) if n > 1 else "Y"
    out = ["\\begin{table}[t]", "\\centering", "\\scriptsize",
           "\\begin{tabularx}{\\linewidth}{" + spec + "}", "\\toprule"]
    out.append(" & ".join(esc_plain(c) for c in rows[0]) + r" \\")
    out.append("\\midrule")
    for row in rows[1:]:
        row = row + [""] * (n - len(row))
        out.append(" & ".join(esc_plain(c) for c in row[:n]) + r" \\")
    out += ["\\bottomrule", "\\end{tabularx}",
            "\\caption{" + esc(heading.lstrip("# ")) + "}", "\\end{table}"]
    return "\n".join(out)

def figure(path, caption, width="0.92\\linewidth"):
    if not (ROOT / path).exists():
        return "% missing figure: " + path
    return ("\\begin{figure}[t]\n\\centering\n"
            f"\\includegraphics[width={width}]{{../{path}}}\n"
            f"\\caption{{{esc(caption)}}}\n\\end{{figure}}")

def build():
    title, draft = convert_draft()
    table_lines = (PAPER / "tables.md").read_text(encoding="utf-8").splitlines()
    audit = json.loads((ROOT / "results_submission/report/evidence_audit.json").read_text())
    audit_note = (f"The frozen evidence audit reports {audit['verified_count']} verified, "
                  f"{audit['partial_count']} partial, and "
                  f"{audit['not_established_count']} not-established requirements. "
                  "The partial and open items are stated explicitly in the limitations.")
    appendix = [
        "\\clearpage\\section{Selected quantitative evidence}",
        esc(audit_note),
        markdown_table(table_lines, "### Controlled benchmark main table"),
        markdown_table(table_lines, "### Component ablation endpoint"),
        markdown_table(table_lines, "### DeepSeek three-seed MVP"),
        "\\clearpage\\section{Figures}",
        figure("results_submission/report/controlled_curves_ci.png",
               "Controlled task-cluster uncertainty: flat duplication changes confidence and actions, while provenance-preserving aggregation is invariant."),
        figure("results_submission/report/local_qwen_effect_heterogeneity/forest.png",
               "Non-pooled CUDA-Qwen effect heterogeneity across task families and fresh expansions."),
        figure("results_submission/scienceworld_frozen_readout_first4_20260929/episode_curves.png",
               "Fixed-input ScienceWorld readout intervention, shown by episode cluster."),
        figure("results_submission/report/alfworld_interactive/curve.png",
               "ALFWorld public-observation exploratory comparison; model and task strata are kept separate."),
        "\\clearpage\\section{Reproducibility note}",
        esc("All numerical artifacts, request manifests, raw traces, figures, and commands are retained under results_submission/. The reproducibility manifest is results_submission/reproducibility_manifest.json. The PDF is a rendering of the current draft and selected generated tables; it does not add experimental claims."),
    ]
    tex = r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=0.78in]{geometry}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlist{nosep}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\title{TITLE}
\author{Anonymous Authors}
\date{}
\begin{document}
\maketitle
DRAFT
APPENDIX
\end{document}
"""
    tex = tex.replace("TITLE", esc(title)).replace("DRAFT", draft).replace("APPENDIX", "\n".join(appendix))
    TEX.write_text(tex, encoding="utf-8")
    subprocess.run(["tectonic", "--keep-logs", TEX.name],
                   cwd=PAPER, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return PDF

if __name__ == "__main__":
    print(build())
