"""Multi-root opaque-text stress test for learned provenance recovery.

The main benchmark intentionally exposes a lexical premise key and has one
root per task, so its extractor F1 is an upper bound. This evaluator composes
two same-family tasks into one multi-root episode, removes premise/source names
from rollout text, randomizes draw metadata, and evaluates pairwise recovery.
It is a recovery diagnostic; downstream planning remains reported by the
original leakage-free evaluator.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, random
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def load(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def opaque_row(row, task_tag, index):
    r=copy.deepcopy(row)
    # Keep action/value information needed by a planner, but hide premise key,
    # source ID, task ID, and the original draw parity from the extractor.
    r['text']=f"A branch report says that option {r['action']} has predicted return {float(r['value']):+.5f}."
    r['sample_id']=f"opaque-{task_tag}-sample-{index}"
    r['source_id']=f"opaque-source-{task_tag}-{index % 2}"
    r['draw_id']=index
    r['action_index']=int(r['action_index'])
    return r


def compose(tasks, seed):
    rng=random.Random(seed); grouped={}
    for task in tasks:
        grouped.setdefault(task['family'],[]).append(task)
    out=[]
    for family, rows in grouped.items():
        rows=list(rows); rng.shuffle(rows)
        for i in range(0,len(rows)-1,2):
            a,b=rows[i],rows[i+1]
            tag=hashlib.sha1(f"{a['task_id']}::{b['task_id']}".encode()).hexdigest()[:12]
            merged=[]; root_tags=[]
            for root_idx,task in enumerate((a,b)):
                # Re-root IDs are deliberately opaque but stable labels for
                # evaluation only; they are never passed to the classifier.
                root=f"opaque-root-{tag}-{root_idx}"
                start=len(merged)
                for j,row in enumerate(task['rollout_lineage']):
                    r=opaque_row(row,f"{tag}-r{root_idx}",start+j)
                    r['premise_ids']=[root]
                    r['component_id']=root_idx
                    r['component_optimal_action']=task['optimal_action']
                    r['component_hidden']=bool(task['hidden_state']['value'])
                    merged.append(r)
                root_tags.extend([root]*len(task['rollout_lineage']))
            out.append({'task_id':f'opaque-{tag}','family':family,'rollout_lineage':merged})
    return out


def pair_feature(a,b):
    ta=set(a['text'].lower().split()); tb=set(b['text'].lower().split())
    shared=len(ta & tb); union=max(1,len(ta|tb))
    return [shared/union,shared,float(a['action_index']==b['action_index']),
            abs(float(a['value'])-float(b['value'])),float(a['draw_id']==b['draw_id'])]


def sample_pairs(tasks, seed, target='premise', per_label=80):
    rng=random.Random(seed); X=[]; y=[]
    for task in tasks:
        rows=task['rollout_lineage']; positives=[]; negatives=[]
        for _ in range(per_label):
            def label(row):
                return row['premise_ids'][0] if target == 'premise' else row['source_id']
            root=label(rng.choice(rows))
            same=[r for r in rows if label(r)==root]
            other=[r for r in rows if label(r)!=root]
            if len(same)>=2: positives.append((rng.choice(same),rng.choice(same)))
            if other: negatives.append((rng.choice(same),rng.choice(other)))
        for a,b in positives:
            X.append(pair_feature(a,b)); y.append(1)
        for a,b in negatives:
            X.append(pair_feature(a,b)); y.append(0)
    return np.asarray(X),np.asarray(y)


def evaluate(tasks, model, threshold, target='premise'):
    prepared=prepare_edges(tasks,target)
    rows=[]; truth_all=[]; pred_all=[]
    for task,truth,feats in prepared:
        probs=model.predict_proba(feats)[:,1]; pred=(probs>=threshold).astype(int)
        p,r,f,_=precision_recall_fscore_support(truth,pred,average='binary',zero_division=0)
        rows.append({'task_id':task['task_id'],'family':task['family'],'target':target,'precision':p,'recall':r,'f1':f,'pairs':len(truth),'positive_rate':float(np.mean(truth))})
        truth_all.extend(truth); pred_all.extend(pred)
    p,r,f,_=precision_recall_fscore_support(truth_all,pred_all,average='binary',zero_division=0)
    return rows,{'target':target,'micro_precision':p,'micro_recall':r,'micro_f1':f,'tasks':len(tasks),'pairs':len(truth_all),'threshold':threshold}


def prepare_edges(tasks, target):
    """Build pair features once; threshold sweeps then only score cached arrays."""
    prepared=[]
    for task in tasks:
        bank=task['rollout_lineage']; truth=[]; feats=[]
        for i,a in enumerate(bank):
            for b in bank[i+1:]:
                truth.append(int((a['premise_ids'][0]==b['premise_ids'][0]) if target=='premise' else (a['source_id']==b['source_id'])))
                feats.append(pair_feature(a,b))
        prepared.append((task,np.asarray(truth),np.asarray(feats)))
    return prepared


def downstream(tasks, model, threshold, target):
    """Measure action loss caused by recovered cross-root grouping."""
    out=[]
    for task in tasks:
        bank=task['rollout_lineage']; parent={i:i for i in range(len(bank))}
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]; i=parent[i]
            return i
        def union(i,j):
            i,j=find(i),find(j)
            if i!=j: parent[j]=i
        feats=[]; pairs=[]
        for i,a in enumerate(bank):
            for j,b in enumerate(bank[i+1:],i+1):
                feats.append(pair_feature(a,b)); pairs.append((i,j))
        pred=(model.predict_proba(np.asarray(feats))[:,1]>=threshold).astype(int)
        for (i,j),positive in zip(pairs,pred):
            if positive: union(i,j)
        for component in sorted({r['component_id'] for r in bank}):
            rows=[r for r in bank if r['component_id']==component]
            hidden=rows[0]['component_hidden']; valid=[r for r in rows if bool(r['hypothesis_value'])==hidden]
            grouped={}
            for idx,r in enumerate(bank):
                if r['component_id']!=component or bool(r['hypothesis_value'])!=hidden: continue
                grouped.setdefault(find(bank.index(r)),[]).append(r)
            action_values={}
            for group in grouped.values():
                action_values.setdefault(group[0]['action'],[]).append(float(np.mean([r['value'] for r in group])))
            means={a:float(np.mean(v)) for a,v in action_values.items()}
            action=max(means,key=means.get); optimal=rows[0]['component_optimal_action']
            reward=float(np.mean([r['value'] for r in valid if r['action']==action]))
            best=float(np.mean([r['value'] for r in valid if r['action']==optimal]))
            true_action_means={a:float(np.mean([r['value'] for r in valid if r['action']==a])) for a in sorted({r['action'] for r in valid})}
            ordered=sorted(true_action_means.values(), reverse=True)
            margin=float(ordered[0]-ordered[1]) if len(ordered)>1 else float('inf')
            component_indices=[i for i,r in enumerate(bank) if r['component_id']==component]
            cross_merge=sum(any(bank[j]['component_id']!=component for j in range(len(bank)) if find(j)==find(i)) for i in component_indices)/max(1,len(component_indices))
            out.append({'task_id':task['task_id'],'component':component,'target':target,'recovery':'learned','action':action,'optimal_action':optimal,'action_correct':int(action==optimal),'reward':reward,'regret':best-reward,'action_margin':margin,'cross_root_merge_rate':cross_merge})
            # Oracle keeps each true premise/source component separate.
            out.append({'task_id':task['task_id'],'component':component,'target':target,'recovery':'oracle','action':optimal,'optimal_action':optimal,'action_correct':1,'reward':best,'regret':0.0,'action_margin':margin,'cross_root_merge_rate':0.0})
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmark',default='results_submission/scaling_benchmark'); ap.add_argument('--output',default='results_submission/provenance_semantic_hard'); ap.add_argument('--seed',type=int,default=20260928); ap.add_argument('--max-tasks-per-split',type=int,default=None); ap.add_argument('--train-tasks',type=int,default=328); ap.add_argument('--dev-tasks',type=int,default=88); ap.add_argument('--test-tasks',type=int,default=76); ap.add_argument('--rollouts-per-root',type=int,default=16); args=ap.parse_args()
    caps={'train':args.train_tasks,'dev':args.dev_tasks,'test':args.test_tasks}
    if args.max_tasks_per_split is not None:
        caps={s:args.max_tasks_per_split for s in ('train','dev','test')}
    base={s:list(load(Path(args.benchmark)/f'{s}.jsonl'))[:caps[s]] for s in ('train','dev','test')}
    data={s:compose(base[s],args.seed+len(s)) for s in base}
    for split in data:
        for task in data[split]:
            by_root={}
            for row in task['rollout_lineage']:
                by_root.setdefault(row['premise_ids'][0],[]).append(row)
            selected=[]
            for root in sorted(by_root):
                actions=sorted({row['action'] for row in by_root[root]})
                per_group=max(1,args.rollouts_per_root//max(1,2*len(actions)))
                for h in (False,True):
                    for action in actions:
                        group=[row for row in by_root[root] if bool(row['hypothesis_value'])==h and row['action']==action]
                        selected.extend(group[:per_group])
            task['rollout_lineage']=selected
    X,y=sample_pairs(data['train'],args.seed,'premise'); premise_model=LogisticRegression(max_iter=200,solver='liblinear',class_weight='balanced').fit(X,y)
    X_source,y_source=sample_pairs(data['train'],args.seed+7919,'source'); source_model=LogisticRegression(max_iter=200,solver='liblinear',class_weight='balanced').fit(X_source,y_source)
    models={'premise':premise_model,'source':source_model}
    best={'premise':(0,.5),'source':(0,.5)}
    prepared_dev={target:prepare_edges(data['dev'],target) for target in best}
    prepared_test={target:prepare_edges(data['test'],target) for target in best}
    for target in best:
        for threshold in np.linspace(.1,.9,17):
            rows=[]; truth_all=[]; pred_all=[]
            for task,truth,feats in prepared_dev[target]:
                pred=(models[target].predict_proba(feats)[:,1]>=threshold).astype(int); truth_all.extend(truth); pred_all.extend(pred)
            _,_,f,_=precision_recall_fscore_support(truth_all,pred_all,average='binary',zero_division=0)
            summary={'micro_f1':f}
            if summary['micro_f1']>best[target][0]: best[target]=(summary['micro_f1'],float(threshold))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); details=[]; summaries=[]
    downstream_rows=[]
    identifiability_rows=[]
    for target in ('premise','source'):
        detail,summary=evaluate(data['test'],models[target],best[target][1],target); details.extend(detail); summaries.append(summary)
        downstream_rows.extend(downstream(data['test'],models[target],best[target][1],target))
        truth_all=[]; pred_all=[]
        for task,truth,feats in prepared_test[target]:
            truth_all.extend(truth.tolist())
            pred_all.extend((models[target].predict_proba(feats)[:,1]>=best[target][1]).astype(int).tolist())
        truth_all=np.asarray(truth_all,dtype=int); pred_all=np.asarray(pred_all,dtype=int)
        actual_f1=float(precision_recall_fscore_support(truth_all,pred_all,average='binary',zero_division=0)[2])
        rng=np.random.default_rng(args.seed + (0 if target == 'premise' else 7919))
        null=[]
        for _ in range(2000):
            shuffled=rng.permutation(truth_all)
            null.append(float(precision_recall_fscore_support(shuffled,pred_all,average='binary',zero_division=0)[2]))
        identifiability_rows.append({'target':target,'actual_f1':actual_f1,'test_pairs':int(len(truth_all)),
                                     'test_positive_rate':float(truth_all.mean()),'null_f1_mean':float(np.mean(null)),
                                     'null_f1_ci95_low':float(np.quantile(null,.025)),
                                     'null_f1_ci95_high':float(np.quantile(null,.975)),
                                     'permutation_draws':len(null),
                                     'interpretation':'label-permutation null; not a semantic oracle'})
    downstream_summary={}
    for target in ('premise','source'):
        subset=[r for r in downstream_rows if r['target']==target]
        downstream_summary[target]={recovery:{k:float(np.mean([r[k] for r in subset if r['recovery']==recovery])) for k in ('action_correct','reward','regret')} for recovery in ('learned','oracle')}
    downstream_frame=np.asarray(downstream_rows, dtype=object)
    margin_rows=[]
    if downstream_rows:
        import pandas as pd
        df=pd.DataFrame(downstream_rows)
        bins=[-1e-12,0.05,0.10,0.25,0.50,float('inf')]
        labels=['[0,.05)','[.05,.10)','[.10,.25)','[.25,.50)','[.50,+inf)']
        df['margin_bin']=pd.cut(df['action_margin'].replace(float('inf'), bins[-1]), bins=bins, labels=labels, right=False, include_lowest=True)
        margin_rows=df.groupby(['target','recovery','margin_bin'], observed=False).agg(
            n=('action_correct','size'), action_correct=('action_correct','mean'),
            reward=('reward','mean'), regret=('regret','mean'),
            cross_root_merge_rate=('cross_root_merge_rate','mean'),
            action_margin_mean=('action_margin','mean')).reset_index()
        margin_rows.to_csv(out/'margin_analysis.csv',index=False)
    import pandas as pd
    pd.DataFrame(identifiability_rows).to_csv(out/'identifiability_null.csv',index=False)
    (out/'test_edge_metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in details)+'\n')
    (out/'downstream.jsonl').write_text('\n'.join(json.dumps(x) for x in downstream_rows)+'\n')
    (out/'summary.json').write_text(json.dumps({'train_composites':len(data['train']),'dev_composites':len(data['dev']),'test_composites':len(data['test']),'pair_train_rows':len(X),'source_pair_train_rows':len(X_source),'summaries':summaries,'thresholds':{k:v[1] for k,v in best.items()},'downstream':downstream_summary,'margin_analysis':'margin_analysis.csv','identifiability_null':'identifiability_null.csv','text_contract':'opaque branch report; premise/source names and original draw parity removed; target-specific train models'},indent=2))
    print(json.dumps(json.loads((out/'summary.json').read_text()),indent=2))


if __name__=='__main__': main()
