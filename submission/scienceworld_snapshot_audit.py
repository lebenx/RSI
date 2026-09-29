"""Replay archived ScienceWorld snapshots with simulator state aligned.

The ordinary closed-loop runner regenerates candidates at every duplication
budget. This audit reuses archived visible/candidate/readout decisions,
replays the original prefix in the simulator, and executes each recorded
action from the matching state. It isolates the aggregator presentation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from scienceworld import ScienceWorldEnv
from submission.scienceworld_runner import VERSION


def archived_rows(trace_root: Path, max_snapshots: int):
    rows=[]
    for trace_path in sorted(trace_root.glob('*-flat-m1-trace.jsonl')):
        trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
        if not trace:
            continue
        name=trace_path.name
        task=name.split('-v',1)[0]
        variation=int(name.split('-v',1)[1].split('-',1)[0])
        for row in trace[:max_snapshots]:
            rows.append({'task':task,'variation':variation,'trace':trace,
                         'snapshot_id':row['snapshot_id'],'step':int(row['outcome']['step']),
                         'visible':row['visible'],'candidate':row['candidate']})
    return rows


def replay_action(task, variation, prefix, action, simplification, steps):
    """Replay archived prefix, then execute one action from that state."""
    env=ScienceWorldEnv(envStepLimit=steps+5)
    try:
        env.load(task, variation, simplification, generateGoldPath=False)
        observation, info=env.reset()
        for old in prefix:
            observation, _, done, info=env.step(old['action'])
            if done:
                raise RuntimeError('archived prefix terminated before snapshot')
        before=int(info['score'])
        observation, reward, done, after=env.step(action)
        return {'score_before':before,'score_after':int(after['score']),
                'immediate_reward':float(reward),'done':int(done)}
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace-root',default='results_submission/scienceworld_v3_m1/episodes')
    p.add_argument('--source-raw',default='results_submission/scienceworld_snapshot_audit_v4/snapshot_raw.csv')
    p.add_argument('--output',default='results_submission/scienceworld_snapshot_audit_v5')
    p.add_argument('--max-snapshots-per-trace',type=int,default=4)
    p.add_argument('--simplification',default='easy')
    p.add_argument('--steps',type=int,default=10)
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(args.source_raw)
    source_rows={(r.task,int(r.variation),r.snapshot_id,int(r.step),int(r.duplication),r.method):r
                 for r in source.itertuples(index=False)}
    archives=archived_rows(Path(args.trace_root),args.max_snapshots_per_trace)
    rows=[]
    for snap in archives:
        key_prefix=(snap['task'],snap['variation'],snap['snapshot_id'],snap['step'])
        subset=[(key,row) for key,row in source_rows.items() if key[:4]==key_prefix]
        if not subset:
            continue
        prefix=snap['trace'][:snap['step']]
        for key,old in subset:
            result=replay_action(snap['task'],snap['variation'],prefix,old.action,args.simplification,args.steps)
            rows.append({'task':snap['task'],'variation':snap['variation'],
                         'snapshot_id':snap['snapshot_id'],'step':snap['step'],
                         'duplication':int(old.duplication),'method':old.method,
                         'action':old.action,'p_true':float(old.p_true),
                         'success_probability':float(old.success_probability),
                         **result,'tokens':int(old.tokens)})
    df=pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('no matching archived rows')
    base_dup=int(df.duplication.min())
    ref=df[df.duplication==base_dup].set_index(['task','variation','snapshot_id','step','method']).action.rename('action_m1')
    df=df.join(ref,on=['task','variation','snapshot_id','step','method'])
    df['action_flip_from_m1']=(df.action!=df.action_m1).astype(int)
    df.to_csv(out/'snapshot_raw.csv',index=False)
    summary=df.groupby(['duplication','method']).agg(
        n=('action','size'),root_confidence=('p_true','mean'),
        action_flip=('action_flip_from_m1','mean'),
        immediate_reward=('immediate_reward','mean'),score_after=('score_after','mean'),
        tokens=('tokens','mean')).reset_index()
    summary.to_csv(out/'snapshot_summary.csv',index=False)
    meta={'version':VERSION,'trace_root':args.trace_root,'source_raw':args.source_raw,
          'replayed_prefix':True,'frozen_fields':['visible','candidate','archived readout action'],
          'changed_field':'stipulated true-branch multiplicity in archived readout',
          'uses_gold_path':False,'uses_hidden_state':False,
          'note':'Simulator state is reconstructed by replaying the original flat m1 prefix before each one-step action.'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('wrote',len(df),'rows to',out)


if __name__=='__main__':
    main()
