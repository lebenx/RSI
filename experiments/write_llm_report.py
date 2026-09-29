from pathlib import Path
import pandas as pd

def md(df, cols):
    x=df[cols].copy();
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]): x[c]=x[c].map(lambda v:f'{v:.3f}')
    lines=['| '+' | '.join(x.columns)+' |','| '+' | '.join('---' for _ in x.columns)+' |']
    for _,r in x.iterrows(): lines.append('| '+' | '.join(str(v) for v in r)+' |')
    return '\n'.join(lines)

def main():
 p=Path('results_paper/llm_subset_v2');r=pd.read_csv(p/'llm_raw.csv');s=pd.read_csv(p/'llm_summary.csv')
 lines=['# DeepSeek LLM subset','',
  'This is a real `deepseek-chat` run with 12 scenarios, 3 information conditions, m∈{1,2,4,8}, 7 derived methods, 300 archived API calls and no synthetic fallback. Every API response parsed successfully; controlled rollout strings had equal length and presentation order was randomized. The prompts never contained hidden physical state or reward labels.', '',
  'The item scorer and evidence-only call are used to construct deduplication, source averaging, explicit mixing and provenance rows. The flat baseline has a separate whole-list API call. These derived rows are still an API subset, not a population estimate.', '']
 for title,root,cond in [('Wrong root / pure duplication',False,'pure_duplication'),('Correct root / independent rollout',True,'independent_rollout'),('Wrong root / paraphrase duplication',False,'paraphrase_duplication')]:
  q=s[(s.root_correct==root)&(s.condition==cond)&(s.multiplicity.isin([1,8]))]
  lines += [f'## {title}','',md(q,['multiplicity','method','root_confidence_mean','root_confidence_ci95_low','root_confidence_ci95_high','action_flip_mean','expected_reward_mean','decision_regret_mean']),'']
 lines += ['## Interpretation','',
  'The analytical grid shows the specified flat-counting failure strongly. The DeepSeek subset does not show a stable monotone confidence increase for the flat whole-list call: its direction varies by scenario and its intervals are wide. This is a valid negative diagnostic, not evidence that every LLM implements the analytical flat estimator.',
  'The stable invariant in the API subset is structural: evidence-only q and provenance/belief-mixing rows remain unchanged across pure duplicates, while conditional-return rows can change under independent samples. A larger API study or a frozen local/open model is required before claiming a model-level effect.', '',
  'Archived prompts and responses: `api_archive.jsonl`; row-level data: `llm_raw.csv`; summary with 95% intervals: `llm_summary.csv`.']
 (p/'llm_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
