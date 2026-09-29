# Paper-scale controlled results

Run the analytic grid:

```bash
python -m experiments.run_numeric --output results_paper --seeds 30
python -m experiments.summarize_numeric --input results_paper/numeric_raw.csv --output results_paper
python experiments/write_numeric_report.py
```

Run the archived DeepSeek subset (the key is read only from the environment):

```bash
DEEPSEEK_API_KEY=... python -m experiments.llm_subset --output results_paper/llm_subset_v2
python experiments/write_llm_report.py
```

`numeric_raw.csv` has 630,000 rows. `numeric_cells.csv` retains every prior,
likelihood, state, direction and utility-family cell; `numeric_claims.csv` is
seed-clustered; `numeric_m16_minus_m1.csv` contains paired dose changes;
`numeric_method_contrasts.csv` contains 10,000-draw paired bootstrap contrasts.
The API directory contains the archived request/response JSONL, parsed rows and
95% intervals. API keys are never written to the archive.

Read [the frozen protocol](../experiments/protocol.md) before interpreting any
table. The analytic provenance method is an oracle mechanism diagnostic, and the
DeepSeek subset is not a cross-model claim.
