"""Evaluate identical rollout banks with the paper baseline family.

The runner separates planner-visible rollout values from hidden labels. Hidden
labels are used only for post-hoc reward and provenance audit. The flat method's
confidence counter is intentionally explicit: it counts additional occurrences
of the same sample as pseudo-evidence. This is a mechanism baseline, not a claim
about arbitrary LLM internals.
"""
from __future__ import annotations
import argparse, json, math, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

METHODS = ('no_imagination','flat_rollout','independent_trajectory','mean_pooling',
           'exact_dedup','semantic_dedup','source_grouping','bayesian_mixing',
           'risk_classifier','oracle_provenance','provenance_preserving')
BUDGETS = (1,2,4,8,16)
SCALING_BUDGETS = (1,2,4,8,16,32,64)

def sigmoid(x): return 1/(1+math.exp(-max(-40,min(40,x))))
def logit(p): return math.log(max(1e-9,p)/max(1e-9,1-p))

def load(path):
    with Path(path).open() as f:
        for line in f: yield json.loads(line)

def rows_for(task, condition, budget):
    rows=task['rollout_lineage']
    if condition == 'pure_duplication':
        # The base bank is an independent conditional bank. A pure copy reuses
        # sample_id and text, so frequency but not information changes.
        return [dict(r, copy_index=k) for r in rows for k in range(budget)]
    # New condition samples get new IDs in this audit. The standard benchmark
    # has four draws/action; scaling files can provide up to 64.
    selected=[r for r in rows if r['draw_id'] < budget]
    return [dict(r) for r in selected]

def dedup(rows, key):
    out={}
    for r in rows: out.setdefault(key(r),r)
    return list(out.values())

def semantic_signature(row):
    """Planner-visible semantic key; never uses hidden premise/source IDs.

    The branch label is a stipulated condition, not a recovered source. The
    action and rounded predicted return keep independent value samples apart,
    while paraphrases with the same conditional prediction collapse.
    """
    return (row['action'], bool(row.get('hypothesis_value')), round(float(row['value']), 5))

def action_means(rows):
    out={}
    for action,g in __import__('itertools').groupby(sorted(rows,key=lambda r:r['action']),key=lambda r:r['action']):
        vals=[r['value'] for r in g]; out[action]=float(np.mean(vals))
    return out

def conditional_means(rows):
    out={}
    for h in (False,True):
        for action in sorted({r['action'] for r in rows}):
            vals=[r['value'] for r in rows if bool(r['premise_ids']) and bool(h)==bool(r['premise_ids'] and h)]
            # Ground-truth conditional configuration is encoded in draw lineage
            # only for diagnostics; values are grouped by action here. The
            # benchmark has one hidden premise, so use source/premise groups
            # rather than exposing hidden_state to a planner.
            out[(h,action)]=float(np.mean(vals)) if vals else 0.0
    return out

def true_action(task):
    values=defaultdict(list)
    hidden=bool(task['hidden_state']['value'])
    for r in task['rollout_lineage']:
        if bool(r.get('hypothesis_value')) == hidden: values[r['action']].append(r['value'])
    return max(values,key=lambda a:np.mean(values[a]))

def q_weighted_values(task, rows):
    # In this benchmark each rollout is labeled with one shared premise; the
    # calibrated reference estimates conditional value by action and uses the
    # task prior as its q. This is the planner-visible oracle interface.
    q=float(task['root_premises'][0]['prior']); values={}
    for action in task['actions']:
        vals=[r['value'] for r in rows if r['action']==action]
        values[action]=float(np.mean(vals)) if vals else -1e9
    return values

def select_no_imagination(task):
    for token in ('inspect','verify','measure','conserve','fallback','safe'):
        for a in task['actions']:
            if token in a: return a
    return task['actions'][0]

def confidence(task, rows, method, condition, budget):
    q=float(task['root_premises'][0]['prior']); hidden=bool(task['hidden_state']['value'])
    if method in ('flat_rollout',):
        counts=Counter(r['sample_id'] for r in rows)
        duplicate_excess=sum(max(0,n-1) for n in counts.values()) if condition=='pure_duplication' else 0
        duplicate_excess /= max(1, len(counts))
        # The flat reader treats every repeated descendant as positive support
        # for the stated premise H=true. Whether H is actually true is hidden
        # from the planner and enters only post-hoc reward/calibration metrics.
        return sigmoid(logit(q)+duplicate_excess*math.log(1.5))
    return q

def mixed_values(task, rows, q):
    values={}
    for action in task['actions']:
        means=[]
        for h in (False,True):
            vals=[r['value'] for r in rows if r['action']==action and bool(r.get('hypothesis_value'))==h]
            means.append(float(np.mean(vals)) if vals else 0.0)
        values[action]=(1-q)*means[0]+q*means[1]
    return values

def values_for(task, rows, method, model=None, q=None):
    if method == 'no_imagination': return {a: 0.0 for a in task['actions']}
    if method == 'flat_rollout': return mixed_values(task, rows, float(q))
    if method in ('independent_trajectory','mean_pooling'):
        return action_means(rows)
    if method == 'exact_dedup': return mixed_values(task, dedup(rows,lambda r:r['text']), float(q))
    if method == 'semantic_dedup': return mixed_values(task, dedup(rows,semantic_signature), float(q))
    if method == 'source_grouping':
        groups=defaultdict(list)
        for r in dedup(rows,lambda r:r['sample_id']): groups[(r['source_id'],r['hypothesis_value'],r['action'])].append(r['value'])
        by_action=defaultdict(lambda: {False:[],True:[]})
        for (source,h,action),v in groups.items(): by_action[action][bool(h)].append(np.mean(v))
        return {a:(1-q)*float(np.mean(v[False])) + q*float(np.mean(v[True])) for a,v in by_action.items()}
    if method in ('bayesian_mixing','oracle_provenance','provenance_preserving'):
        # All generated rollouts share the same premise. The recognized source
        # graph therefore averages sample descendants once before action choice.
        return mixed_values(task, dedup(rows,lambda r:r['sample_id']), float(q))
    if method == 'risk_classifier':
        if model is None: return action_means(rows)
        q=float(task['root_premises'][0]['prior']); feature=[]
        for action in task['actions']:
            vals=[r['value'] for r in rows if r['action']==action]
            feature.append([q,float(np.mean(vals) if vals else 0),float(np.std(vals) if vals else 0),len(vals)])
        probs=model.predict_proba(np.asarray(feature))[:,1]
        return {a:float(p) for a,p in zip(task['actions'],probs)}
    raise ValueError(method)

def fit_risk(train_tasks):
    X=[]; y=[]
    for task in train_tasks:
        best=true_action(task)
        rows=task['rollout_lineage']
        q=float(task['root_premises'][0]['prior'])
        for action in task['actions']:
            vals=[r['value'] for r in rows if r['action']==action]
            X.append([q,float(np.mean(vals)),float(np.std(vals)),len(vals)])
            y.append(int(action==best))
    model=LogisticRegression(max_iter=1000,class_weight='balanced')
    model.fit(np.asarray(X),np.asarray(y)); return model

def evaluate(tasks, train_tasks, conditions=('pure_duplication','independent_rollout'), budgets=BUDGETS):
    risk=fit_risk(train_tasks); result=[]
    for task in tasks:
        oracle=true_action(task)
        true_h=bool(task['hidden_state']['value'])
        true_action_values={a:[r['value'] for r in task['rollout_lineage'] if r['action']==a and bool(r.get('hypothesis_value'))==true_h] for a in task['actions']}
        base_score=max(np.mean(v) for v in true_action_values.values())
        for condition in conditions:
            for budget in budgets:
                rows=rows_for(task,condition,budget)
                for method in METHODS:
                    conf=confidence(task,rows,method,condition,budget)
                    values=values_for(task,rows,method,risk,conf)
                    if method=='no_imagination': action=select_no_imagination(task)
                    else: action=max(values,key=values.get)
                    action_values=[r['value'] for r in rows if r['action']==action]
                    true_h=bool(task['hidden_state']['value'])
                    true_values=[r['value'] for r in task['rollout_lineage'] if r['action']==action and bool(r.get('hypothesis_value'))==true_h]
                    optimal_values=[r['value'] for r in task['rollout_lineage'] if r['action']==oracle and bool(r.get('hypothesis_value'))==true_h]
                    reward=float(np.mean(true_values)) if true_values else -1.0
                    optimal_reward=float(np.mean(optimal_values)) if optimal_values else base_score
                    result.append({'task_id':task['task_id'],'family':task['family'],'difficulty':task['difficulty'],'split':task['split'],
                        'condition':condition,'budget':budget,'method':method,'root_correct':bool(task['hidden_state']['value']),
                        'root_confidence':conf,'action':action,'oracle_action':oracle,'action_correct':int(action==oracle),
                        'reward':reward,'oracle_reward':optimal_reward,'regret':optimal_reward-reward,
                        'wrong_premise_confidence':conf if not task['hidden_state']['value'] else 1-conf,
                        'duplicate_excess':sum(max(0,n-1) for n in Counter(r['sample_id'] for r in rows).values()),
                        'rollout_count':len(rows),'unique_sample_count':len({r['sample_id'] for r in rows})})
    return pd.DataFrame(result)

def summarize(result):
    rows=[]
    group=['family','difficulty','condition','budget','method']
    for keys,g in result.groupby(group,sort=True):
        row=dict(zip(group,keys)); row['n']=len(g)
        for m in ('root_confidence','wrong_premise_confidence','reward','regret','action_correct'):
            row[m+'_mean']=float(g[m].mean()); row[m+'_std']=float(g[m].std(ddof=1)) if len(g)>1 else 0
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',default='results_submission/benchmark');p.add_argument('--output',default='results_submission/baselines');p.add_argument('--split',default='test');p.add_argument('--budgets',default='1,2,4,8,16');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    train=list(load(Path(a.benchmark)/'train.jsonl')); test=list(load(Path(a.benchmark)/(a.split+'.jsonl')))
    budgets=tuple(int(x) for x in a.budgets.split(',')); result=evaluate(test,train,budgets=budgets)
    result.to_csv(out/'baseline_raw.csv',index=False); summarize(result).to_csv(out/'baseline_summary.csv',index=False)
    meta={'methods':METHODS,'conditions':['pure_duplication','independent_rollout'],'budgets':budgets,'train':len(train),'test':len(test),'rows':len(result),'hidden_labels_used_for':'evaluation_reward_and_audit_only','flat_note':'stipulated duplicate-count pseudo-likelihood'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
