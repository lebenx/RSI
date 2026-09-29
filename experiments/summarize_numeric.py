"""Make paper-facing summaries and plots from the numeric grid."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["root_confidence", "posterior_error", "decision_regret", "value_rmse", "reward", "expected_reward", "brier", "nll", "action_flip", "harmful_flip", "helpful_flip"]

def ci(x):
    x=np.asarray(x,dtype=float); n=len(x); mean=float(x.mean())
    sd=float(x.std(ddof=1)) if n>1 else 0.0
    margin=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else float('nan')
    return mean,sd,mean-margin,mean+margin,n

def summarize(df, group_cols, seed_first=True):
    # Collapse to independent seeds before intervals. A row within a seed is a
    # factorial task setting, never a statistical replicate.
    base_cols=group_cols + ['seed'] if seed_first else group_cols
    if seed_first:
        x=df.groupby(base_cols,as_index=False)[METRICS].mean()
    else: x=df
    rows=[]
    for keys,g in x.groupby(group_cols,sort=True,dropna=False):
        if not isinstance(keys,tuple): keys=(keys,)
        r=dict(zip(group_cols,keys))
        for m in METRICS:
            mean,sd,lo,hi,n=ci(g[m]); r[m+'_mean']=mean;r[m+'_std']=sd;r[m+'_ci95_low']=lo;r[m+'_ci95_high']=hi
        r['n_seeds']=len(g);rows.append(r)
    return pd.DataFrame(rows)

def paired_change(df, group_cols):
    x=df.groupby(group_cols+['seed','multiplicity'],as_index=False)[METRICS].mean()
    p=x[x.multiplicity==16].drop(columns=['multiplicity']).merge(x[x.multiplicity==1].drop(columns=['multiplicity']),on=group_cols+['seed'],suffixes=('_m16','_m1'))
    rows=[]
    for keys,g in p.groupby(group_cols,sort=True,dropna=False):
        if not isinstance(keys,tuple): keys=(keys,)
        r=dict(zip(group_cols,keys))
        for m in ['root_confidence','posterior_error','decision_regret','value_rmse','reward','expected_reward','brier','nll','action_flip','harmful_flip','helpful_flip']:
            z=g[m+'_m16']-g[m+'_m1'];mean,sd,lo,hi,n=ci(z)
            r[m+'_delta_mean']=mean;r[m+'_delta_std']=sd;r[m+'_delta_ci95_low']=lo;r[m+'_delta_ci95_high']=hi
        r['n_seeds']=len(g);rows.append(r)
    return pd.DataFrame(rows)

def bootstrap_diff(df, group_cols, metric, a='flat', b='provenance', n=10000):
    # Paired seed-cluster bootstrap for a method contrast; deterministic RNG.
    x=df.groupby(group_cols+['seed','method'],as_index=False)[metric].mean()
    p=x[x.method==a].drop(columns='method').merge(x[x.method==b].drop(columns='method'),on=group_cols+['seed'],suffixes=('_a','_b'))
    if not len(p): return None
    z=(p[metric+'_a']-p[metric+'_b']).to_numpy(); rng=np.random.default_rng(9841)
    draws=rng.choice(z,len(z)*n,replace=True).reshape(n,len(z)).mean(axis=1)
    return dict(contrast=f'{a}-{b}',metric=metric,mean=float(z.mean()),ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),n_seeds=len(z))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='results_paper/numeric_raw.csv');ap.add_argument('--output',default='results_paper');a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=True)
    df=pd.read_csv(a.input)
    claim_cols=['root_correct','condition','multiplicity','method']
    claim=summarize(df,claim_cols)
    claim.to_csv(out/'numeric_claims.csv',index=False)
    cell=summarize(df,['physical_state','hypothesis_direction','root_correct','prior','strength','family','condition','multiplicity','method'])
    cell.to_csv(out/'numeric_cells.csv',index=False)
    change=paired_change(df,['root_correct','condition','method']);change.to_csv(out/'numeric_m16_minus_m1.csv',index=False)
    contrasts=[]
    for root in [False,True]:
      for cond in ['pure_duplication','selective_duplication','paraphrase_duplication','independent_rollout','new_real_evidence']:
       for m in [1,16]:
        q=df[(df.root_correct==root)&(df.condition==cond)&(df.multiplicity==m)]
        for metric in ['root_confidence','reward','expected_reward','decision_regret','value_rmse','brier','nll']:
         z=bootstrap_diff(q,['root_correct','condition','multiplicity'],metric,n=10000)
         if z: contrasts.append(z|{'root_correct':root,'condition':cond,'multiplicity':m})
    pd.DataFrame(contrasts).to_csv(out/'numeric_method_contrasts.csv',index=False)
    # Plot compact claim curves. No plot is used for inference; source data is CSV.
    import matplotlib.pyplot as plt
    for metric,ylab,name in [('root_confidence','reported P(H=1)','numeric_root_confidence.png'),('reward','realized reward','numeric_reward.png')]:
      fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True if metric=='reward' else False)
      for ax,root in zip(axs,[False,True]):
       for cond,color in [('pure_duplication','#c44e52'),('independent_rollout','#4c72b0'),('new_real_evidence','#55a868')]:
        q=claim[(claim.root_correct==root)&(claim.condition==cond)]
        for method,ls in [('flat','-'),('source_average','--'),('belief_mixing',':'),('provenance','-.')]:
         z=q[q.method==method].sort_values('multiplicity')
         if not len(z): continue
         ax.plot(z.multiplicity,z[metric+'_mean'],ls=ls,color=color,alpha=.8,label=f'{cond}/{method}' if root==False else None)
         ax.fill_between(z.multiplicity,z[metric+'_ci95_low'],z[metric+'_ci95_high'],color=color,alpha=.05)
       ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16]);ax.set_xlabel('m');ax.set_title('root_correct='+str(root));ax.grid(alpha=.2)
      axs[0].set_ylabel(ylab);axs[0].legend(fontsize=7,ncol=2,loc='best');fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig)
    print('raw',df.shape,'claims',claim.shape,'cells',cell.shape,'changes',change.shape,'contrasts',len(contrasts))
    # Audits for exact method equalities and duplicate invariance.
    x=df[(df.method=='belief_mixing')].merge(df[df.method=='provenance'],on=['scenario_id','condition','multiplicity'],suffixes=('_mix','_prov'))
    print('max_mixing_provenance_abs_diff',float((x.root_confidence_mix-x.root_confidence_prov).abs().max()))
    q=df[(df.condition=='pure_duplication')&(df.method=='provenance')]
    spread=q.groupby(['scenario_id']).root_confidence.agg(lambda x:float(x.max()-x.min())).max()
    print('max_pure_duplicate_provenance_spread',float(spread))

if __name__=='__main__': main()
