"""Summarize the balanced local-Qwen full-episode pilot.

This is an exploratory second-model agent check: two plant variations have
flat/provenance-success at m=1 and m=4, plus one held-out variation at both
budgets. It does not pool with the DeepSeek planner ledger.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

RUNS = [
    # Keep every comparison on the current runner/model version.  The m=1
    # directory contains all three variations, so filter it into the balanced
    # grid and held-out strata below instead of mixing in the older v8 archive.
    ('grid', 1, Path('results_submission/scienceworld_local_qwen_value_grid_v9_m1'), {152, 158}),
    ('grid', 4, Path('results_submission/scienceworld_local_qwen_value_grid_m4'), {152, 158}),
    ('v161', 1, Path('results_submission/scienceworld_local_qwen_value_grid_v9_m1'), {161}),
    ('v161', 4, Path('results_submission/scienceworld_local_qwen_value161_v9_m4'), {161}),
]

def load():
    rows=[]
    for label, duplication, directory, variations in RUNS:
        data=json.loads((directory/'summaries.json').read_text())
        for row in data:
            if row.get('error') is not None: continue
            if int(row.get('variation')) not in variations: continue
            row=dict(row); row['duplication']=duplication; row['pilot_stratum']=label; row['source_directory']=str(directory)
            rows.append(row)
    return pd.DataFrame(rows)

def bootstrap(values, seed=20260929, draws=20000):
    values=np.asarray(values,float)
    if not len(values): return (float('nan'),)*3
    rng=np.random.default_rng(seed); d=values[rng.integers(0,len(values),size=(draws,len(values)))].mean(1)
    return float(values.mean()),float(np.quantile(d,.025)),float(np.quantile(d,.975))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('results_submission/report/local_qwen_episode_pilot')); a=p.parse_args()
    frame=load(); a.output.mkdir(parents=True,exist_ok=True); frame.to_csv(a.output/'rows.csv',index=False)
    frame['initial_brier']=(frame['initial_success_probability']-frame['success'])**2
    summary=frame.groupby(['pilot_stratum','duplication','method'],as_index=False).agg(n=('variation','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),initial_brier=('initial_brier','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'))
    summary.to_csv(a.output/'summary.csv',index=False)
    wide=frame.pivot_table(index=['task','variation'],columns=['method','duplication'],values=['success','final_score','reward','steps','charged_tokens'],aggfunc='first')
    paired=[]
    for (task,variation), row in wide.iterrows():
        if ('success','flat',1) not in row.index or ('success','flat',4) not in row.index: continue
        rec={'task':task,'variation':int(variation)}
        for metric in ('success','final_score','reward','steps','charged_tokens'):
            rec[f'flat_m1_{metric}']=row.get((metric,'flat',1)); rec[f'flat_m4_{metric}']=row.get((metric,'flat',4)); rec[f'provenance_success_m1_{metric}']=row.get((metric,'provenance_success',1)); rec[f'provenance_success_m4_{metric}']=row.get((metric,'provenance_success',4))
            rec[f'provenance_success_minus_flat_m1_{metric}']=rec[f'provenance_success_m1_{metric}']-rec[f'flat_m1_{metric}']
            rec[f'provenance_success_minus_flat_m4_{metric}']=rec[f'provenance_success_m4_{metric}']-rec[f'flat_m4_{metric}']
            rec[f'flat_m4_minus_flat_m1_{metric}']=rec[f'flat_m4_{metric}']-rec[f'flat_m1_{metric}']
        paired.append(rec)
    paired=pd.DataFrame(paired); paired.to_csv(a.output/'paired.csv',index=False)
    contrasts=[]
    for metric in ('success','reward','steps'):
        for dup in (1,4):
            col=f'provenance_success_minus_flat_m{dup}_{metric}'; vals=paired[col].dropna().to_numpy(float); est,lo,hi=bootstrap(vals)
            contrasts.append({'duplication':dup,'metric':metric,'n_episodes':len(vals),'estimate':est,'ci95_low':lo,'ci95_high':hi,'bootstrap_draws':20000})
    pd.DataFrame(contrasts).to_csv(a.output/'bootstrap.csv',index=False)
    probe=Path('results_submission/report/local_qwen_gpu_probe.json'); probe_data=json.loads(probe.read_text()) if probe.exists() else {}
    metadata={'protocol':'local-qwen-full-episode-pilot-v1','model':'Qwen2.5-Coder-3B-Instruct','task':'find-plant','variations':[152,158,161],'duplications':[1,4],'methods':['flat','provenance_success'],'valid_rows':int(len(frame)),'error_rows':0,'paired_episodes':int(len(paired)),'step_limit':8,'episode_success_claim':False,'planner_claim':False,'calibration_metric':'initial Brier=(initial_success_probability-success)^2','gpu_probe':probe_data,'interpretation':'small second-model interactive pilot; not pooled with DeepSeek and not a confirmatory success estimate','cuda_run_policy':'LOCAL_QWEN_DEVICE=cuda for every completed runner invocation; CUDA is required and CPU fallback is disabled','runner_version':'scienceworld_runner v9 for all included rows'}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)); print(json.dumps(metadata,indent=2)); print(summary.to_string(index=False)); print(paired.to_string(index=False))
if __name__=='__main__': main()
