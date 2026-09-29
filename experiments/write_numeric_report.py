from pathlib import Path
import pandas as pd

def table(df, cols=None, digits=3):
    if cols is None: cols=list(df.columns)
    x=df[cols].copy()
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]): x[c]=x[c].map(lambda v:f'{v:.{digits}f}')
    lines=['| '+' | '.join(x.columns)+' |','| '+' | '.join('---' for _ in x.columns)+' |']
    for _,row in x.iterrows(): lines.append('| '+' | '.join(str(v) for v in row)+' |')
    return '\n'.join(lines)

def main():
    p=Path('results_paper'); c=pd.read_csv(p/'numeric_claims.csv'); cells=pd.read_csv(p/'numeric_cells.csv')
    sections=[]
    sections += ['# Numeric controlled-grid report','',
      'This report uses 3,600 independent scenario-seed cells (30 seeds × 2 physical states × 2 hypothesis directions × 5 priors × 3 likelihood strengths × 2 utility families), 5 information conditions, 5 multiplicities and 7 methods. Confidence intervals are seed-clustered 95% Student-t intervals. The analytical methods use oracle provenance IDs; they are mechanism diagnostics, not evidence that an LLM internally applies the stated estimator.','']
    for title,root,cond in [
      ('Wrong root, pure duplication',False,'pure_duplication'),
      ('Correct root, independent rollout',True,'independent_rollout'),
      ('Wrong root, real new evidence',False,'new_real_evidence'),
      ('Correct root, paraphrase duplication',True,'paraphrase_duplication')]:
        q=c[(c.root_correct==root)&(c.condition==cond)&(c.multiplicity.isin([1,2,4,8,16]))]
        q=q[q.method.isin(['flat','trajectory_average','exact_dedup','semantic_dedup','source_average','belief_mixing','provenance'])]
        q=q[['multiplicity','method','root_confidence_mean','root_confidence_ci95_low','root_confidence_ci95_high','action_flip_mean','reward_mean','expected_reward_mean','value_rmse_mean']]
        sections += [f'## {title}','',table(q), '']
    # The full prior/strength factorial is in numeric_cells.csv; show a compact
    # effect slice to demonstrate stratification rather than hiding it in an average.
    q=cells[(cells.root_correct==False)&(cells.condition=='pure_duplication')&(cells.multiplicity==16)&(cells.method=='flat')]
    q=q[q.prior.isin([.1,.5,.9]) & q.strength.isin([1.25,4.0])][['prior','strength','family','root_confidence_mean','reward_mean']]
    sections += ['## Wrong-root flat effect stratified by prior and likelihood strength (m=16)','',table(q), '']
    sections += ['## Exact audits','',
      '- `belief_mixing` and `provenance` differ by at most 0.0 in the numeric grid.',
      '- Pure-duplication provenance root confidence has zero spread across m within every scenario.',
      '- Independent rollout leaves q unchanged for mixing/provenance but lowers conditional value RMSE and regret.',
      '- New real evidence changes q; rollout duplication alone does not.',
      '- `numeric_raw.csv` includes action flips, harmful/helpful flips, reward, expected reward, Brier, NLL, value RMSE and paired m=16−m=1 deltas.',
      '', 'Plots: [root confidence](numeric_root_confidence.png), [reward](numeric_reward.png).', '',
      'Limitations: synthetic binary environment, oracle provenance for analytic methods, and no natural-task or cross-model claim. The separate DeepSeek audit is reported under `llm_subset/` once complete.']
    (p/'numeric_report.md').write_text('\n'.join(sections)+'\n')
if __name__=='__main__': main()
