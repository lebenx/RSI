"""Mechanism ablations for the controlled SharedPremiseBench.

All variants consume the same visible rollout bank; only belief separation,
sample identity, and aggregation are changed. Hidden labels are used for
post-hoc reward/action correctness only.
"""
from __future__ import annotations
import argparse,json,math,random
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from submission.benchmark_eval import load, rows_for, true_action, mixed_values, action_means, dedup, sigmoid, logit

METHODS=('remove_provenance','remove_belief_separation','remove_sample_identity','attention_only','simple_dedup','random_grouping','oracle_grouping')
def choose(task,rows,method,budget):
 q=float(task['root_premises'][0]['prior'])
 if method=='remove_provenance':
  conf=sigmoid(logit(q)+(sum(max(0,n-1) for n in Counter(r['sample_id'] for r in rows).values())/max(1,len(set(r['sample_id'] for r in rows))))*math.log(1.5))
  vals=mixed_values(task,rows,conf)
 elif method=='remove_belief_separation':
  conf=q; vals=action_means(rows)
 elif method=='remove_sample_identity':
  conf=q; vals=mixed_values(task,rows,q)
 elif method=='attention_only':
  conf=q; vals={}
  for a in task['actions']:
   z=[r['value'] for r in rows if r['action']==a]
   w=np.exp(np.clip(np.asarray(z)*2,-20,20)); vals[a]=float(np.sum(w*np.asarray(z))/max(1e-9,np.sum(w)))
 elif method=='simple_dedup':
  # A transparent exact-text/action grouping baseline. It preserves no
  # learned semantic relation and is intentionally weaker than provenance.
  conf=q; vals=mixed_values(task,dedup(rows,lambda r:(r['text'],r['action'],r['hypothesis_value'])),q)
 elif method=='random_grouping':
  conf=q; rng=random.Random(task['seed']+budget); kept=[]
  # wrong groups merge unrelated sources with probability proportional to budget.
  for r in rows:
   if rng.random()<1/max(1,budget): kept.append(r)
  vals=mixed_values(task,kept or rows,q)
 elif method=='oracle_grouping':
  conf=q; vals=mixed_values(task,dedup(rows,lambda r:r['sample_id']),q)
 return conf,vals

def main():
 p=argparse.ArgumentParser();p.add_argument('--benchmark',default='results_submission/scaling_benchmark');p.add_argument('--output',default='results_submission/ablations');p.add_argument('--budgets',default='1,2,4,8,16,32,64');a=p.parse_args()
 test=list(load(Path(a.benchmark)/'test.jsonl')); out=Path(a.output);out.mkdir(parents=True,exist_ok=True); rows=[]
 for t in test:
  oracle=true_action(t); hidden=bool(t['hidden_state']['value'])
  opt=np.mean([r['value'] for r in t['rollout_lineage'] if r['action']==oracle and bool(r['hypothesis_value'])==hidden])
  for b in map(int,a.budgets.split(',')):
   bank=rows_for(t,'pure_duplication',b)
   for m in METHODS:
    conf,vals=choose(t,bank,m,b); act=max(vals,key=vals.get)
    true=[r['value'] for r in t['rollout_lineage'] if r['action']==act and bool(r['hypothesis_value'])==hidden]
    reward=float(np.mean(true)) if true else -1
    rows.append({'task_id':t['task_id'],'family':t['family'],'difficulty':t['difficulty'],'budget':b,'method':m,'root_confidence':conf,'wrong_premise_confidence':conf if not hidden else 1-conf,'action':act,'oracle_action':oracle,'action_correct':int(act==oracle),'reward':reward,'regret':opt-reward})
 df=pd.DataFrame(rows); df.to_csv(out/'ablation_raw.csv',index=False); df.groupby(['family','difficulty','budget','method']).agg(n=('reward','size'),root_confidence_mean=('root_confidence','mean'),wrong_premise_confidence_mean=('wrong_premise_confidence','mean'),reward_mean=('reward','mean'),regret_mean=('regret','mean'),action_correct_mean=('action_correct','mean')).reset_index().to_csv(out/'ablation_summary.csv',index=False)
 (out/'metadata.json').write_text(json.dumps({'methods':METHODS,'budgets':[int(x) for x in a.budgets.split(',')],'hidden_labels_used_for':'post-hoc reward and correctness only'},indent=2))
 print('wrote',len(df),'rows')
if __name__=='__main__': main()
