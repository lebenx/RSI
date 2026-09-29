"""Train and evaluate a leakage-free provenance edge extractor.

The extractor only sees rollout text and action metadata. Hidden premise/source
IDs are labels in train/dev and are never features. A small TF-IDF pair classifier
is intentionally transparent; its test precision/recall and downstream reward
are the evidence, not the oracle grouping result.
"""
from __future__ import annotations
import argparse, json, math, random, re, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def load(path):
    with Path(path).open() as f:
        for line in f: yield json.loads(line)

def pair_feature(a,b,vec=None,cache=None):
    # Sparse transparent text features. The extractor does not receive the
    # hidden premise/source ID; it learns lexical/structural similarity on train.
    ta=set(a['text'].lower().split()); tb=set(b['text'].lower().split())
    shared=len(ta & tb); union=max(1,len(ta|tb))
    return [shared/union, shared, float(a['action_index']==b['action_index']),
            abs(float(a['value'])-float(b['value'])), float(a['draw_id']==b['draw_id'])]

def make_pairs(tasks, vec, seed, per_task=20):
    rng=random.Random(seed); X=[]; y=[]
    for task in tasks:
        rows=task['rollout_lineage']
        cache={}
        # Within-task positives and cross-task negatives prevent task-ID leakage.
        positives=[]; negatives=[]
        for _ in range(per_task):
            a,b=rng.sample(rows,2); positives.append((a,b,int(a['premise_ids']==b['premise_ids'])))
            other=rng.choice(tasks)
            if other['task_id']==task['task_id']: other=rng.choice(tasks)
            c=rng.choice(other['rollout_lineage']); negatives.append((a,c,0))
        for a,b,label in positives+negatives:
            X.append(pair_feature(a,b,vec,cache)); y.append(label)
    return np.asarray(X),np.asarray(y)


def make_source_pairs(tasks, vec, seed, per_task=20):
    """Build source-lineage pairs with the same leakage-free protocol."""
    rng=random.Random(seed); X=[]; y=[]
    for task in tasks:
        rows=task['rollout_lineage']
        for _ in range(per_task):
            a,b=rng.sample(rows,2)
            X.append(pair_feature(a,b,vec,{})); y.append(int(a['source_id']==b['source_id']))
            other=rng.choice(tasks)
            if other['task_id']==task['task_id']: other=rng.choice(tasks)
            c=rng.choice(other['rollout_lineage']); X.append(pair_feature(a,c,vec,{})); y.append(0)
    return np.asarray(X),np.asarray(y)


def evaluate_source_edges(test, model, threshold):
    """Evaluate source recovery and its grouped-value decision downstream."""
    edge_rows=[]; downstream=[]
    for task in test:
        rows=task['rollout_lineage']; pairs=[]; truth=[]; features=[]
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                pairs.append((a['sample_id'],b['sample_id'])); truth.append(int(a['source_id']==b['source_id']))
                features.append(pair_feature(a,b,None,{}))
        probs=model.predict_proba(np.asarray(features))[:,1]
        pred=[int(x>=threshold) for x in probs]
        precision,recall,f1,_=precision_recall_fscore_support(truth,pred,average='binary',zero_division=0)
        edge_rows.append({'task_id':task['task_id'],'family':task['family'],'difficulty':task['difficulty'],
                          'precision':precision,'recall':recall,'f1':f1,'threshold':threshold,'pairs':len(pairs)})
        learned=components(rows,[pair for pair,p in zip(pairs,pred) if p])
        oracle=components(rows,[(a['sample_id'],b['sample_id']) for i,a in enumerate(rows) for b in rows[i+1:] if a['source_id']==b['source_id']])
        for label,groups in [('learned_source',learned),('oracle_source',oracle)]:
            vals=grouped_action_values(rows,groups); action=max(vals,key=vals.get); optimal=task['optimal_action']; hidden=bool(task['hidden_state']['value'])
            reward=float(np.mean([r['value'] for r in rows if r['action']==action and bool(r.get('hypothesis_value'))==hidden]))
            best=float(np.mean([r['value'] for r in rows if r['action']==optimal and bool(r.get('hypothesis_value'))==hidden]))
            downstream.append({'task_id':task['task_id'],'family':task['family'],'difficulty':task['difficulty'],
                               'recovery':label,'action':action,'optimal_action':optimal,'action_correct':int(action==optimal),
                               'reward':reward,'regret':best-reward})
    return pd.DataFrame(edge_rows),pd.DataFrame(downstream)

def components(rows, positive_pairs):
    parent={r['sample_id']:r['sample_id'] for r in rows}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for a,b in positive_pairs: union(a,b)
    return {r['sample_id']:find(r['sample_id']) for r in rows}

def grouped_action_values(rows, groups):
    group_values=defaultdict(lambda: defaultdict(list))
    for r in rows: group_values[groups[r['sample_id']]][r['action']].append(r['value'])
    per_action=defaultdict(list)
    for group in group_values.values():
        for action,values in group.items(): per_action[action].append(float(np.mean(values)))
    return {a:float(np.mean(v)) for a,v in per_action.items()}

def generated_groups(rows):
    """A generated-ID baseline using only rollout text/action, not hidden IDs.

    The toy text grammar exposes the condition and action, so this is an
    intentionally optimistic model-generated provenance upper bound. Numeric
    returns are removed before hashing; exact source IDs never enter features.
    """
    out={}
    for r in rows:
        norm=re.sub(r'[+-]?\d+\.\d+', '<value>', r['text'].lower())
        norm=re.sub(r'\s+', ' ', norm).strip()
        out[r['sample_id']]='generated-'+hashlib.sha1(norm.encode()).hexdigest()[:12]
    return out

def evaluate(test, vec, model, threshold):
    edge_rows=[]; downstream=[]
    for task in test:
        rows=task['rollout_lineage']; pairs=[]; truth=[]; pred=[]
        cache={}
        feature_rows=[]
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                feature_rows.append(pair_feature(a,b,vec,cache))
                pairs.append((a['sample_id'],b['sample_id']))
                truth.append(int(a['premise_ids']==b['premise_ids']))
        probabilities=model.predict_proba(np.asarray(feature_rows))[:,1]
        pred=[int(probability>=threshold) for probability in probabilities]
        precision,recall,f1,_=precision_recall_fscore_support(truth,pred,average='binary',zero_division=0)
        edge_rows.append({'task_id':task['task_id'],'family':task['family'],'difficulty':task['difficulty'],
                          'precision':precision,'recall':recall,'f1':f1,'threshold':threshold,'pairs':len(pairs)})
        predicted=components(rows,[pair for pair,p in zip(pairs,pred) if p])
        oracle=components(rows,[(a['sample_id'],b['sample_id']) for i,a in enumerate(rows) for b in rows[i+1:] if a['premise_ids']==b['premise_ids']])
        generated=generated_groups(rows)
        for label,groups in [('learned',predicted),('generated',generated),('oracle',oracle)]:
            vals=grouped_action_values(rows,groups); action=max(vals,key=vals.get); optimal=task['optimal_action']
            hidden=bool(task['hidden_state']['value'])
            reward=float(np.mean([r['value'] for r in rows if r['action']==action and bool(r.get('hypothesis_value'))==hidden]))
            optimal_reward=float(np.mean([r['value'] for r in rows if r['action']==optimal and bool(r.get('hypothesis_value'))==hidden]))
            downstream.append({'task_id':task['task_id'],'family':task['family'],'difficulty':task['difficulty'],
                               'recovery':label,'action':action,'optimal_action':optimal,'action_correct':int(action==optimal),
                               'reward':reward,'regret':optimal_reward-reward})
    return pd.DataFrame(edge_rows),pd.DataFrame(downstream)

def noisy_downstream(test, rates):
    rows=[]
    rng=random.Random(8301)
    for noise in rates:
        for task in test:
            bank=task['rollout_lineage']; groups={r['sample_id']:r['source_id'] for r in bank}
            ids=list(groups)
            for sid in ids:
                if rng.random()<noise: groups[sid]=f'noise-{rng.randrange(10_000)}'
            vals=grouped_action_values(bank,groups); action=max(vals,key=vals.get); optimal=task['optimal_action']
            hidden=bool(task['hidden_state']['value'])
            reward=float(np.mean([r['value'] for r in bank if r['action']==action and bool(r.get('hypothesis_value'))==hidden])); best=float(np.mean([r['value'] for r in bank if r['action']==optimal and bool(r.get('hypothesis_value'))==hidden]))
            rows.append({'noise_rate':noise,'task_id':task['task_id'],'family':task['family'],'action_correct':int(action==optimal),'reward':reward,'regret':best-reward})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',default='results_submission/benchmark');p.add_argument('--output',default='results_submission/provenance');p.add_argument('--threshold',type=float,default=.5);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    train=list(load(Path(a.benchmark)/'train.jsonl')); dev=list(load(Path(a.benchmark)/'dev.jsonl')); test=list(load(Path(a.benchmark)/'test.jsonl'))
    vec=None
    X,y=make_pairs(train,vec,991); model=LogisticRegression(max_iter=1000,class_weight='balanced').fit(X,y)
    X_source,y_source=make_source_pairs(train,vec,992); source_model=LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_source,y_source)
    # A dev threshold is chosen once, then frozen for test. This is the only
    # learned-provenance tuning operation.
    best=(0,.5)
    # Freeze on a deterministic development subset so the full test evaluation
    # remains cheap and the split is still respected.
    for threshold in np.linspace(.1,.9,9):
        ed,_=evaluate(dev[:120],vec,model,float(threshold)); score=float(ed.f1.mean())
        if score>best[0]: best=(score,float(threshold))
    edges,downstream=evaluate(test,vec,model,best[1]); noisy=noisy_downstream(test,(0,.05,.1,.2,.4))
    source_best=(0,.5)
    for threshold in np.linspace(.1,.9,9):
        ed,_=evaluate_source_edges(dev[:120],source_model,float(threshold)); score=float(ed.f1.mean())
        if score>source_best[0]: source_best=(score,float(threshold))
    source_edges,source_downstream=evaluate_source_edges(test,source_model,source_best[1])
    edges.to_csv(out/'test_edge_metrics.csv',index=False); downstream.to_csv(out/'test_downstream.csv',index=False); noisy.to_csv(out/'noisy_downstream.csv',index=False)
    source_edges.to_csv(out/'test_source_edge_metrics.csv',index=False); source_downstream.to_csv(out/'test_source_downstream.csv',index=False)
    summary={'train_tasks':len(train),'dev_tasks':len(dev),'test_tasks':len(test),'pair_train_rows':len(X),'source_pair_train_rows':len(X_source),'frozen_threshold':best[1],'source_frozen_threshold':source_best[1],
             'test_edge_precision':float(edges.precision.mean()),'test_edge_recall':float(edges.recall.mean()),'test_edge_f1':float(edges.f1.mean()),
             'test_source_edge_precision':float(source_edges.precision.mean()),'test_source_edge_recall':float(source_edges.recall.mean()),'test_source_edge_f1':float(source_edges.f1.mean()),
             'downstream':downstream.groupby('recovery')[['action_correct','reward','regret']].mean().to_dict('index'),
             'source_downstream':source_downstream.groupby('recovery')[['action_correct','reward','regret']].mean().to_dict('index'),
             'noisy':noisy.groupby('noise_rate')[['action_correct','reward','regret']].mean().to_dict('index')}
    (out/'metadata.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
