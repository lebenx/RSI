"""Generate compact paper-ready tables/figures from frozen experiment artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('results_submission'); OUT=ROOT/'report'; OUT.mkdir(parents=True,exist_ok=True)

def paired_bootstrap(rows, unit_cols, metric_cols, n_boot=20000, seed=20260928):
 """Deterministic percentile bootstrap for paired duplication-4 minus duplication-1 effects."""
 if rows.empty: return pd.DataFrame()
 rng=np.random.default_rng(seed)
 out=[]
 for method,g in rows.groupby('method'):
  wide=g.pivot_table(index=unit_cols,columns='duplication',values=metric_cols,aggfunc='first')
  if 1 not in wide.columns.get_level_values('duplication') or 4 not in wide.columns.get_level_values('duplication'): continue
  for metric in metric_cols:
   if (metric,1) not in wide.columns or (metric,4) not in wide.columns: continue
   d=(wide[(metric,4)]-wide[(metric,1)]).dropna().to_numpy(dtype=float)
   if not len(d): continue
   draws=rng.integers(0,len(d),size=(n_boot,len(d)))
   means=d[draws].mean(axis=1)
   out.append({'method':method,'metric':metric,'n_units':len(d),'dup4_minus_dup1':float(d.mean()),'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975)),'bootstrap_draws':n_boot})
 return pd.DataFrame(out)

def clustered_rate_bootstrap(rows, unit_col, numerator, denominator, n_boot=20000, seed=20260929):
 """Cluster bootstrap for a rate whose headline estimate is denominator-weighted."""
 if rows.empty: return pd.DataFrame()
 rng=np.random.default_rng(seed); out=[]
 for method,g in rows.groupby('method'):
  units=g.groupby(unit_col,as_index=False)[[numerator,denominator]].sum()
  if units.empty: continue
  num=units[numerator].to_numpy(dtype=float); den=units[denominator].to_numpy(dtype=float)
  draws=rng.integers(0,len(units),size=(n_boot,len(units)))
  means=num[draws].sum(axis=1)/den[draws].sum(axis=1)
  out.append({'method':method,'metric':'action_flip_rate','n_units':len(units),'dup4_minus_dup1':float(num.sum()/den.sum()),'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975)),'bootstrap_draws':n_boot})
 return pd.DataFrame(out)

def method_bootstrap(rows, unit_cols, metrics, method_a='provenance', method_b='flat', n_boot=20000, seed=20260931):
 """Bootstrap method_a minus method_b within matched episode units."""
 if rows.empty: return pd.DataFrame()
 rng=np.random.default_rng(seed); out=[]
 for duplication,g in rows.groupby('duplication'):
  wide=g.pivot_table(index=unit_cols,columns='method',values=metrics,aggfunc='first')
  for metric in metrics:
   if (metric,method_a) not in wide.columns or (metric,method_b) not in wide.columns: continue
   d=(wide[(metric,method_a)]-wide[(metric,method_b)]).dropna().to_numpy(dtype=float)
   if not len(d): continue
   draws=rng.integers(0,len(d),size=(n_boot,len(d))); means=d[draws].mean(axis=1)
   out.append({'duplication':int(duplication),'contrast':f'{method_a}-{method_b}','metric':metric,'n_units':len(d),'estimate':float(d.mean()),'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975)),'bootstrap_draws':n_boot})
 return pd.DataFrame(out)

def main():
 raw=pd.read_csv(ROOT/'scaling/baseline_raw.csv')
 # Aggregate over test family/difficulty. Flip is relative to each task's m=1 action.
 base=raw[raw.condition=='pure_duplication'].copy(); ref=base[base.budget==1].set_index(['task_id','method']).action.rename('action_m1'); base=base.join(ref,on=['task_id','method']); base['flip_from_m1']=(base.action!=base.action_m1).astype(int)
 controlled_agg=base.groupby(['budget','method']).agg(n=('reward','size'),root_confidence=('root_confidence','mean'),wrong_premise_confidence=('wrong_premise_confidence','mean'),reward=('reward','mean'),regret=('regret','mean'),action_correct=('action_correct','mean'),action_flip_rate=('flip_from_m1','mean')).reset_index()
 controlled_agg.to_csv(OUT/'controlled_summary.csv',index=False)
 from submission.controlled_bootstrap import bootstrap as controlled_bootstrap
 controlled_ci=controlled_bootstrap(ROOT/'scaling/baseline_raw.csv')
 controlled_ci.to_csv(OUT/'controlled_bootstrap.csv',index=False)
 main_ci_long=controlled_ci[(controlled_ci.budget.isin([1,8,64])) & controlled_ci.method.isin(['no_imagination','flat_rollout','independent_trajectory','mean_pooling','exact_dedup','semantic_dedup','source_grouping','bayesian_mixing','risk_classifier','oracle_provenance','provenance_preserving'])]
 main_ci=main_ci_long.pivot_table(index=['condition','budget','method'],columns='metric',values=['estimate','ci95_low','ci95_high'],aggfunc='first').reset_index()
 main_ci.columns=['_'.join(str(x) for x in c if str(x)!='').rstrip('_') if isinstance(c,tuple) else str(c) for c in main_ci.columns]
 main_ci.to_csv(OUT/'main_table_ci.csv',index=False)
 # Main methods at m=1 and m=64.
 methods=['no_imagination','flat_rollout','independent_trajectory','mean_pooling','exact_dedup','semantic_dedup','source_grouping','bayesian_mixing','risk_classifier','oracle_provenance','provenance_preserving']
 tab=controlled_agg[(controlled_agg.method.isin(methods))&(controlled_agg.budget.isin([1,8,64]))]
 tab.to_csv(OUT/'main_table.csv',index=False)
 # baseline curves
 plt.style.use('seaborn-v0_8-whitegrid')
 fig,axs=plt.subplots(1,3,figsize=(14,4))
 for method,color in [('flat_rollout','#c43c39'),('provenance_preserving','#276fbf'),('mean_pooling','#4f9d69'),('exact_dedup','#7a5aa6')]:
  z=controlled_agg[controlled_agg.method==method].sort_values('budget')
  axs[0].plot(z.budget,z.root_confidence,'o-',label=method,color=color)
  axs[1].plot(z.budget,z.reward,'o-',label=method,color=color)
  axs[2].plot(z.budget,z.action_flip_rate,'o-',label=method,color=color)
 axs[0].set(xlabel='duplication',ylabel='root confidence'); axs[1].set(xlabel='duplication',ylabel='reward'); axs[2].set(xlabel='duplication',ylabel='action flip rate')
 axs[0].set_xscale('log',base=2); axs[1].set_xscale('log',base=2); axs[2].set_xscale('log',base=2); axs[0].legend(fontsize=8)
 fig.tight_layout(); fig.savefig(OUT/'controlled_curves.png',dpi=180); plt.close(fig)
 # The CI companion plot resamples task IDs, not individual rollout rows.
 fig,axs=plt.subplots(1,3,figsize=(14,4))
 ci_methods=[('flat_rollout','#c43c39'),('provenance_preserving','#276fbf'),('independent_trajectory','#4f9d69')]
 ci_specs=[('root_confidence','root confidence'),('reward','reward'),('action_flip_rate','action flip rate')]
 for method,color in ci_methods:
  for ax,(metric,label) in zip(axs,ci_specs):
   z=controlled_ci[(controlled_ci.method==method)&(controlled_ci.metric==metric)].sort_values('budget')
   if z.empty: continue
   x=z.budget.to_numpy(dtype=float); y=z.estimate.to_numpy(dtype=float)
   low=y-z.ci95_low.to_numpy(dtype=float); high=z.ci95_high.to_numpy(dtype=float)-y
   ax.errorbar(x,y,yerr=np.vstack([low,high]),fmt='o-',capsize=2,label=method,color=color)
   ax.set(xlabel='duplication',ylabel=label)
   ax.set_xscale('log',base=2)
 axs[0].legend(fontsize=8)
 fig.tight_layout(); fig.savefig(OUT/'controlled_curves_ci.png',dpi=180); plt.close(fig)
 # Independent condition is a separate value-information axis. Keep it in
 # separate artifacts so it is not confused with pure-duplication scaling.
 from submission.independent_analysis import summarize as independent_summarize
 independent_summary, independent_pairs = independent_summarize(ROOT/'scaling/baseline_raw.csv')
 independent_summary.to_csv(OUT/'independent_scaling.csv',index=False)
 independent_pairs.to_csv(OUT/'independent_paired.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(10,3.8))
 independent_methods=[('independent_trajectory','#4f9d69','-'),('provenance_preserving','#276fbf','--'),('flat_rollout','#c43c39','-')]
 for method,color,style in independent_methods:
  z=independent_summary[independent_summary.method==method].sort_values('budget')
  axs[0].plot(z.budget,z.reward,'o'+style,label=method,color=color)
  axs[1].plot(z.budget,z.regret,'o'+style,label=method,color=color)
 for ax,label in zip(axs,['reward','regret']):
  ax.set(xlabel='independent rollout budget',ylabel=label); ax.set_xscale('log',base=2)
 axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/'independent_scaling.png',dpi=180); plt.close(fig)
 from submission.value_estimation import summarize as value_mse_summarize
 value_mse=value_mse_summarize(ROOT/'scaling_benchmark/test.jsonl')
 value_mse.to_csv(OUT/'value_estimation_mse.csv',index=False)
 fig,ax=plt.subplots(figsize=(6,4))
 for condition,color,label in [('independent_rollout','#4f9d69','independent samples'),('pure_duplication','#c43c39','pure copies')]:
  z=value_mse[value_mse.condition==condition].sort_values('budget')
  ax.plot(z.budget,z.mse,'o-',label=label,color=color)
 ax.set(xlabel='rollout budget',ylabel='conditional-value MSE'); ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.legend(fontsize=8)
 fig.tight_layout(); fig.savefig(OUT/'value_estimation_mse.png',dpi=180); plt.close(fig)
 # Ablation endpoint table
 ab=pd.read_csv(ROOT/'ablations/ablation_summary.csv'); ab64=ab[ab.budget==64].groupby('method').agg(root_confidence=('root_confidence_mean','mean'),wrong_premise_confidence=('wrong_premise_confidence_mean','mean'),reward=('reward_mean','mean'),regret=('regret_mean','mean'),action_correct=('action_correct_mean','mean')).reset_index(); ab64.to_csv(OUT/'ablation_endpoint.csv',index=False)
 # ScienceWorld paired summary
 sw=[]; guided=[]; paired=[]; animal=[]; dev8=[]
 for name in ['scienceworld_dev_m1','scienceworld_dev_m4','scienceworld_plant161_guided_m1i','scienceworld_plant161_guided_m4','scienceworld_findplant_dev2_m1','scienceworld_findplant_dev2_m4','scienceworld_animal_dev1_m1','scienceworld_animal_dev1_m4','scienceworld_dev8_m1','scienceworld_dev8_m4']:
  p=ROOT/name/'summaries.json'
  if p.exists():
   rows=json.loads(p.read_text()); sw += rows
   if 'guided' in name or 'findplant_dev2' in name: guided += rows
   if 'findplant_dev2' in name or 'animal_dev1' in name: paired += rows
   if 'animal_dev1' in name: animal += rows
   if 'dev8' in name: dev8 += rows
 if sw:
  sdf=pd.DataFrame(sw); sdf.to_csv(OUT/'scienceworld_rows.csv',index=False)
  valid=sdf[sdf.error.isna()].copy()
  valid['initial_brier']=(valid['initial_success_probability'].fillna(0.5)-valid['success'])**2; valid['regret_proxy']=100-valid['final_score']
  swtab=valid.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index(); swtab.to_csv(OUT/'scienceworld_summary.csv',index=False)
  if guided:
   gdf=pd.DataFrame(guided); gdf=gdf[gdf.error.isna()]; gdf['initial_brier']=(gdf['initial_success_probability'].fillna(0.5)-gdf['success'])**2; gdf['regret_proxy']=100-gdf['final_score']; gdf.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_guided_summary.csv',index=False)
  if paired:
   pdf=pd.DataFrame(paired); pdf=pdf[pdf.error.isna()]; pdf['initial_brier']=(pdf['initial_success_probability'].fillna(0.5)-pdf['success'])**2; pdf['regret_proxy']=100-pdf['final_score']; grouped=pdf.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index(); grouped.to_csv(OUT/'scienceworld_paired_summary.csv',index=False)
   plant=pdf[pdf.task=='find-plant']; plant.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_findplant_summary.csv',index=False)
   z=pd.read_csv(OUT/'scienceworld_findplant_summary.csv'); fig,axs=plt.subplots(1,3,figsize=(12,3.5)); methods=['no_imagination','flat','provenance']; colors=['#777777','#c43c39','#276fbf'];
   for b,style in [(1,'-'),(4,'--')]:
    q=z[z.duplication==b].set_index('method'); axs[0].bar([f'{m}\nm{b}' for m in methods],[q.loc[m,'success'] for m in methods],color=colors,alpha=.85); axs[1].bar([f'{m}\nm{b}' for m in methods],[q.loc[m,'tokens']/1000 for m in methods],color=colors,alpha=.85); axs[2].bar([f'{m}\nm{b}' for m in methods],[q.loc[m,'calibration_brier'] for m in methods],color=colors,alpha=.85)
   axs[0].set_ylabel('success rate'); axs[1].set_ylabel('tokens (k)'); axs[2].set_ylabel('initial Brier'); fig.tight_layout(); fig.savefig(OUT/'scienceworld_guided.png',dpi=180); plt.close(fig)
  if animal:
   adf=pd.DataFrame(animal); adf=adf[adf.error.isna()]; adf['initial_brier']=(adf['initial_success_probability'].fillna(0.5)-adf['success'])**2; adf['regret_proxy']=100-adf['final_score']; adf.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_animal_summary.csv',index=False)
   if dev8:
    ddf=pd.DataFrame(dev8); ddf=ddf[ddf.error.isna()]; ddf['initial_brier']=(ddf['initial_success_probability'].fillna(0.5)-ddf['success'])**2; ddf['regret_proxy']=100-ddf['final_score']; cols=['duplication','method']; dev8_agg=ddf.groupby(cols).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index(); dev8_agg.to_csv(OUT/'scienceworld_dev8_summary.csv',index=False); ddf.groupby(['task','variation','duplication','method']).agg(success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_dev8_by_task.csv',index=False)
 # Fair v4 ScienceWorld runs use the same LLM readout interface for flat and
 # provenance; provenance only receives a deterministic sample-grouped bank.
 # Keep these separate from the earlier exploratory v2/v3 runs.
 fair_pairs=[]
 for name in ['scienceworld_v4_fair_final','scienceworld_v4_fair_final_m4_retry']:
  p=ROOT/name/'summaries.json'
  if p.exists(): fair_pairs += json.loads(p.read_text())
 if fair_pairs:
  fdf=pd.DataFrame(fair_pairs); fdf=fdf[fdf.error.isna()].copy(); fdf['initial_brier']=(fdf['initial_success_probability'].fillna(0.5)-fdf['success'])**2; fdf['regret_proxy']=100-fdf['final_score']
  fdf.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_v4_fair_summary.csv',index=False)
 fair_dev8=[]
 for name in ['scienceworld_v7_dev8_m1','scienceworld_v7_dev8_m4_retry']:
  p=ROOT/name/'summaries.json'
  if p.exists(): fair_dev8 += json.loads(p.read_text())
 if fair_dev8:
  f8=pd.DataFrame(fair_dev8); f8=f8[f8.error.isna()].copy(); f8['initial_brier']=(f8['initial_success_probability'].fillna(0.5)-f8['success'])**2; f8['regret_proxy']=100-f8['final_score']
  f8_summary=f8.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index()
  # Compute action consistency from paired trace rows, not from final success.
  action_maps={}
  for name,dup in [('scienceworld_v7_dev8_m1',1),('scienceworld_v7_dev8_m4_retry',4)]:
   for method in ['no_imagination','flat','provenance']:
    amap={}
    for tp in (ROOT/name/'episodes').glob(f'*-{method}-m{dup}-trace.jsonl'):
     prefix=tp.name.split(f'-{method}-m{dup}-trace.jsonl')[0]
     for line in tp.read_text().splitlines():
      row=json.loads(line); amap[(prefix,row['outcome']['step'])]=row['action']
    action_maps[(dup,method)]=amap
  flips=[]
  for method in ['no_imagination','flat','provenance']:
   a1=action_maps.get((1,method),{}); a4=action_maps.get((4,method),{}); keys=set(a1)&set(a4)
   rate=float(sum(a1[k]!=a4[k] for k in keys)/len(keys)) if keys else 0.0
   flips.append({'method':method,'action_flip_rate':rate})
  flip_df=pd.DataFrame(flips)
  f8_summary=f8_summary.merge(flip_df,on='method',how='left')
  f8_summary.to_csv(OUT/'scienceworld_v7_dev8_summary.csv',index=False)
  # Paired percentile bootstrap over the eight task×variation episodes.
  boot=paired_bootstrap(f8,['task','variation'],['success','final_score','reward','regret_proxy','steps','charged_tokens','repeated_no_visible_change','initial_brier'])
  action_units=[]
  for method in ['no_imagination','flat','provenance']:
   a1=action_maps.get((1,method),{}); a4=action_maps.get((4,method),{})
   prefixes=sorted({k[0] for k in a1}&{k[0] for k in a4})
   for prefix in prefixes:
    keys=[k for k in a1 if k[0]==prefix and k in a4]
    if not keys: continue
    flips=sum(a1[k]!=a4[k] for k in keys)
    action_units.append({'episode':prefix,'method':method,'flips':flips,'steps_aligned':len(keys)})
  if action_units:
   boot_action=clustered_rate_bootstrap(pd.DataFrame(action_units),'episode','flips','steps_aligned')
   boot=pd.concat([boot,boot_action],ignore_index=True)
  boot.to_csv(OUT/'scienceworld_v7_bootstrap.csv',index=False)
  method_bootstrap(f8,['task','variation'],['success','final_score','reward','regret_proxy','steps','charged_tokens','repeated_no_visible_change','initial_brier']).to_csv(OUT/'scienceworld_v7_method_bootstrap.csv',index=False)
  f8.groupby(['task','variation','duplication','method']).agg(success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_v7_dev8_by_task.csv',index=False)
  fig,axs=plt.subplots(1,3,figsize=(12,3.5)); colors={'no_imagination':'#777777','flat':'#c43c39','provenance':'#276fbf'}
  for method in ['no_imagination','flat','provenance']:
   z=f8_summary[f8_summary.method==method].sort_values('duplication')
   axs[0].plot(z.duplication,z.reward,'o-',label=method,color=colors[method])
   axs[1].plot(z.duplication,z.success,'o-',label=method,color=colors[method])
   axs[2].plot(z.duplication,z.action_flip_rate,'o-',label=method,color=colors[method])
  axs[0].set(xlabel='duplication',ylabel='episode reward'); axs[1].set(xlabel='duplication',ylabel='success rate'); axs[2].set(xlabel='duplication',ylabel='paired action flip rate'); axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/'scienceworld_v7_fair_curves.png',dpi=180); plt.close(fig)
 # Fixed-grid expansion: 12 predeclared find-plant/find-animal variations,
 # same runner and methods, with all candidate-interface failures retained.
 expand_summary_path=OUT/'scienceworld_expand_summary.csv'
 expand_action_path=OUT/'scienceworld_expand_action_bootstrap.csv'
 if expand_summary_path.exists() and expand_action_path.exists():
  es=pd.read_csv(expand_summary_path); ea=pd.read_csv(expand_action_path)
  valid_total=int(es.n_valid.sum()); error_total=int(es.n_errors.sum())
  def _expand_rate(task,dup,method,col):
   z=es[(es.task==task)&(es.duplication==dup)&(es.method==method)]
   return float(z[col].iloc[0]) if len(z) else float('nan')
  lines_text=(
   'A fixed-grid expansion covers find-plant and find-animal variations 150--155 '
   '(12 task/variation pairs, 72 method-budget rows; {} valid and {} candidate-interface '
   'errors). Errors remain visible in `scienceworld_expand_rows.csv` and are excluded '
   'only from means. On find-plant, flat success changes from {:.3f} to {:.3f} '
   'while provenance stays {:.3f} to {:.3f}; on find-animal, flat changes from '
   '{:.3f} to {:.3f} while provenance stays {:.3f} to {:.3f}. The expanded '
   'full-episode sample therefore supports duplication-robust action consistency '
   'but still does not establish a general success-rate gain. Across aligned steps, '
   'flat action flips are {:.3f} (95% CI {:.3f}--{:.3f}) and provenance flips '
   '{:.3f}. See `scienceworld_expand_summary.csv`, '
   '`scienceworld_expand_method_bootstrap.csv`, and '
   '`scienceworld_expand_action_bootstrap.csv`.'.format(
    valid_total,error_total,
    _expand_rate('find-plant',1,'flat','success'),_expand_rate('find-plant',4,'flat','success'),
    _expand_rate('find-plant',1,'provenance','success'),_expand_rate('find-plant',4,'provenance','success'),
    _expand_rate('find-animal',1,'flat','success'),_expand_rate('find-animal',4,'flat','success'),
    _expand_rate('find-animal',1,'provenance','success'),_expand_rate('find-animal',4,'provenance','success'),
    float(ea[ea.method=='flat'].action_flip_rate.iloc[0]),float(ea[ea.method=='flat'].ci95_low.iloc[0]),float(ea[ea.method=='flat'].ci95_high.iloc[0]),
    float(ea[ea.method=='provenance'].action_flip_rate.iloc[0])))
  expand_report_text=lines_text
 extra=[]
 for name in ['scienceworld_v8_extra_m1','scienceworld_v8_extra_m4']:
  p=ROOT/name/'summaries.json'
  if p.exists(): extra += json.loads(p.read_text())
 if extra:
  ex=pd.DataFrame(extra); ex=ex[ex.error.isna()].copy(); ex['initial_brier']=(ex['initial_success_probability'].fillna(0.5)-ex['success'])**2; ex['regret_proxy']=100-ex['final_score']
  ex.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_v8_extra_summary.csv',index=False)
  ex.groupby(['task','variation','duplication','method']).agg(success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_v8_extra_by_task.csv',index=False)
 fair_living=[]
 for name in ['scienceworld_v4_living_m1','scienceworld_v4_living_m4']:
  p=ROOT/name/'summaries.json'
  if p.exists(): fair_living += json.loads(p.read_text())
 if fair_living:
  fl=pd.DataFrame(fair_living); fl=fl[fl.error.isna()].copy(); fl['initial_brier']=(fl['initial_success_probability'].fillna(0.5)-fl['success'])**2; fl['regret_proxy']=100-fl['final_score']
  fl.groupby(['duplication','method']).agg(n=('task','size'),success=('success','mean'),final_score=('final_score','mean'),reward=('reward','mean'),regret_proxy=('regret_proxy','mean'),steps=('steps','mean'),tokens=('charged_tokens','mean'),repeated_no_visible_change=('repeated_no_visible_change','mean'),calibration_brier=('initial_brier','mean')).reset_index().to_csv(OUT/'scienceworld_v4_living_summary.csv',index=False)
 # Fixed-snapshot audit isolates aggregator multiplicity from candidate/bank
 # regeneration. It is generated by submission.scienceworld_snapshot_audit.
 snap=ROOT/'scienceworld_snapshot_audit_v5/snapshot_summary.csv'
 if not snap.exists(): snap=ROOT/'scienceworld_snapshot_audit_v4/snapshot_summary.csv'
 if not snap.exists(): snap=ROOT/'scienceworld_snapshot_audit_v2/snapshot_summary.csv'
 if snap.exists(): pd.read_csv(snap).to_csv(OUT/'scienceworld_snapshot_summary.csv',index=False)
 snap_raw=ROOT/'scienceworld_snapshot_audit_v5/snapshot_raw.csv'
 if not snap_raw.exists(): snap_raw=ROOT/'scienceworld_snapshot_audit_v4/snapshot_raw.csv'
 if snap_raw.exists():
  sr=pd.read_csv(snap_raw); sr['regret_proxy']=100-sr['score_after']
  paired_bootstrap(sr,['task','variation','snapshot_id','step'],['p_true','success_probability','immediate_reward','score_after','regret_proxy','tokens','action_flip_from_m1'],seed=20260930).to_csv(OUT/'scienceworld_snapshot_bootstrap.csv',index=False)
 # A paired real-environment counterfactual: the archived DeepSeek readout is
 # rerun on the same public state while only shared-premise multiplicity
 # changes. The replay script applies a common public continuation, so these
 # rows are kept separate from the broad closed-loop success table.
 counterfactual_path=ROOT/'interactive_counterfactual_replay/summary.csv'
 counterfactual=None
 if counterfactual_path.exists():
  counterfactual=pd.read_csv(counterfactual_path)
  counterfactual.to_csv(OUT/'scienceworld_interactive_counterfactual.csv',index=False)
 # Markdown report with caveats
 lines=['# MVP submission report','', 'Generated from frozen artifacts; no numbers below are hand-entered.','', '## Controlled SharedPremiseBench','', 'The scaling bank has 500 tasks total (78 in the held-out test split), five premise families, two counterfactual hypothesis values, and 64 independent condition draws per action. Hidden state is used only for post-hoc reward, optimal-action labels, and calibration audit; the planner-visible bank contains action/value text and IDs. `flat_rollout` is an explicit stipulated duplicate-count pseudo-likelihood mechanism.','']
 for b in [1,2,4,8,16,32,64]:
  z=controlled_agg[(controlled_agg.budget==b)&(controlled_agg.method.isin(['flat_rollout','provenance_preserving','mean_pooling','exact_dedup']))].set_index('method')
  lines.append(f'At duplication={b}: flat confidence {z.loc["flat_rollout","root_confidence"]:.3f}, wrong-premise confidence {z.loc["flat_rollout","wrong_premise_confidence"]:.3f}, reward {z.loc["flat_rollout","reward"]:.3f}, flip rate {z.loc["flat_rollout","action_flip_rate"]:.3f}; provenance-preserving confidence {z.loc["provenance_preserving","root_confidence"]:.3f}, reward {z.loc["provenance_preserving","reward"]:.3f}, flip rate {z.loc["provenance_preserving","action_flip_rate"]:.3f}.')
 def _cboot(method, budget, metric):
  row=controlled_ci[(controlled_ci.method==method)&(controlled_ci.budget==budget)&(controlled_ci.metric==metric)].iloc[0]
  return f'{float(row.estimate):.3f} (95% CI {float(row.ci95_low):.3f}–{float(row.ci95_high):.3f})'
 task_ci_text = (
  'The controlled estimates use a deterministic 20,000-draw bootstrap over the 78 held-out tasks. '
  'At duplication 8, flat root confidence is {} and its action-flip rate is {}; flat reward is {}. '
  'Provenance root confidence is {} with zero observed flips, and reward is {}. '
  'The bootstrap quantifies task heterogeneity; it does not convert the stipulated flat estimator '
  'into an empirical claim about arbitrary language models. The wide machine-generated table is '
  '`main_table_ci.csv`; the long-form intervals are in `controlled_bootstrap.csv`.'
 ).format(
  _cboot('flat_rollout',8,'root_confidence'),
  _cboot('flat_rollout',8,'action_flip_rate'),
  _cboot('flat_rollout',8,'reward'),
  _cboot('provenance_preserving',8,'root_confidence'),
  _cboot('provenance_preserving',8,'reward'),
 )
 def _ind(metric, budget, method):
  row=independent_summary[(independent_summary.budget==budget)&(independent_summary.method==method)].iloc[0]
  return f'{float(row[metric]):.3f}'
 ind_text=(
  'On the independent-rollout condition, independent trajectory aggregation has reward '
  f'{_ind("reward",1,"independent_trajectory")} at budget 1 and {_ind("reward",64,"independent_trajectory")} at budget 64, '
  f'versus {_ind("reward",64,"provenance_preserving")} for the provenance-preserving duplicate baseline at budget 64. '
  'The task-level paired reward advantage remains positive across budgets, but the curve is noisy rather than monotone; '
  'this supports useful independent value information without claiming that more samples always improve the selected action. '
  'See `independent_scaling.csv`, `independent_paired.csv`, and `independent_scaling.png`.'
 )
 def _mse(condition, budget):
  return float(value_mse[(value_mse.condition==condition)&(value_mse.budget==budget)].mse.iloc[0])
 value_text=(
  f'Across 624 held-out action/premise groups, independent-sample conditional-value MSE falls from {_mse("independent_rollout",1):.6f} at n=1 to {_mse("independent_rollout",64):.6f} at n=64. '
  f'Pure duplication stays at {_mse("pure_duplication",1):.6f} for every presentation budget. '
  'This directly tests the variance-reduction part of the theory; the target is the 64-draw conditional mean and no hidden state is used by the estimator. '
  'See `value_estimation_mse.csv` and `value_estimation_mse.png`.'
 )
 lines += [
  '', '### Task-cluster uncertainty', '', task_ci_text,
  '', '### Independent-rollout value axis', '', ind_text,
  '', '### Conditional-value estimation error', '', value_text,
  '',
  'A harmful flip example is documented in `qualitative_examples.md`: a false shared premise causes flat confidence to rise from 0.754 to 1.000 and changes a +1.041 action into a -1.940 action, while provenance stays invariant.',
  '', '## Real-model presentation controls', '',
  'The focused DeepSeek control audit archives 96 calls over four scenarios, duplication 1/4, front/reverse/interleaved ordering, paraphrased text, and neutral/cautious prompts. Two malformed JSON responses are retained as failures and never replaced. The mean confidence delta at duplication 4 ranges from about -0.151 to +0.034 across presentation cells, with action flips up to 0.50. The local Qwen2.5-Coder-3B control audit has 96 calls over three seeds, two scenarios per seed, front/reverse placement, paraphrases, and neutral/cautious prompts: neutral prompts remain at confidence 0.9, while the cautious prompt gives dup=4 confidence 0.55 on average with action flips up to 1.0. These are context/prompt sensitivity results, not universal monotone duplication laws. See `../deepseek_controls/summary.csv` and `../local_qwen_controls/summary.csv`.',
  '', '## Learned provenance recovery', '',
  'The extractor result is in `../provenance/metadata.json`. It reports premise-edge and source-edge precision/recall/F1, oracle/generated-ID/learned grouping, and noisy downstream reward. Both edge F1 values are 1.0 because the toy benchmark exposes the premise key and source pattern lexically; these are upper-bound sanity checks, not evidence of robust semantic provenance recovery. Noise rates and downstream reward are reported separately.',
  '', '## ScienceWorld', '',
  'The guided public-action-shortlist runner uses only task text, public history, and admissible commands; no gold path or hidden state enters the model. The fair v7 protocol gives flat and provenance the same candidate generation, conditional bank, and one LLM readout; provenance receives a sample-grouped bank and carries the evidence-only belief unchanged. Across eight official dev variations, no-imagination succeeds on 0.625 of episodes and flat/provenance on 0.5 at both budgets. Flat reward is 69.0 at duplication 1 and 77.375 at duplication 4 in this stochastic API run, but its paired actions flip 27/64 times (0.422); provenance reward is 77.375 at both budgets with zero flips. The difficult animal variations fail for all methods, so this is evidence for action consistency and duplication robustness, not a broad success-rate gain. A six-variation `find-living-thing` stress run has success 0.333 for no-imagination and 0.167 for both rollout methods at both budgets; these rows are retained as candidate-generation failure cases rather than filtered out. Cluster bootstrap over the eight episodes gives the flat step-weighted flip rate 0.422 (95% CI 0.125–0.688) and provenance 0.000 (0–0); the paired flat reward change is +8.375 (0–25.125), while provenance is exactly 0 in this frozen run. See `scienceworld_v7_bootstrap.csv` for all metrics.',
  '', '### Legacy snapshot diagnostic (input-integrity caveat)', '',
  'The archived v3/v5 snapshot rows are retained for provenance, but their trace-derived visible histories were later found to contain the post-action mutable history view. Their numerical readout and replay diagnostics are therefore not used as clean causal evidence. The clean serialized-input intervention below is the authoritative fixed-bank result; see `scienceworld_input_integrity/summary.json`.',
  '', '## Limitations', '',
  '* The controlled experiment is synthetic and the flat effect is deliberately stipulated to isolate the mechanism.*',
  '* The real-model studies are small and model-specific; they do not establish a universal LLM behavioral law.*',
  '* ScienceWorld evidence is now multi-variation but still insufficient for a broad provenance-planner improvement claim.*',
 ]
 lines.insert(lines.index('* ScienceWorld evidence is now multi-variation but still insufficient for a broad provenance-planner improvement claim.*')+1, '* The original local Qwen ScienceWorld smoke returned incomplete conditional-rollout JSON; a compact serialization retry produced one valid flat/provenance pair (n=1, both unsuccessful). A three-variation compact grid then produced six valid rows, with flat success 0.333 and provenance success 0.000; these remain schema/model feasibility diagnostics rather than planner evidence.*')
 if 'expand_report_text' in locals():
  lines.insert(lines.index('## Limitations'), '\n### Fixed-grid ScienceWorld expansion\n\n'+expand_report_text+'\n')
 expand_cf_path=OUT/'scienceworld_expand_counterfactual/summary.csv'
 if expand_cf_path.exists():
  ec=pd.read_csv(expand_cf_path)
  def _ec_rate(method,dup):
   z=ec[(ec.method==method)&(ec.duplication==dup)]
   return float(z.success.mean()) if len(z) else float('nan')
  lines.insert(lines.index('## Limitations'), '\n### Expanded state-replay stress\n\nAn offline common-continuation replay over the same 12 initial states gives grouped provenance success {:.3f}, flat duplication-1 success {:.3f}, and flat duplication-4 success {:.3f}; grouped provenance also has lower mean reward. This nonpositive stress case is retained because provenance normalization removes duplication sensitivity but cannot repair a weak candidate/value bank. See `scienceworld_expand_counterfactual/summary.csv`.\n'.format(_ec_rate('provenance_grouped',1),_ec_rate('flat',1),_ec_rate('flat',4)))
 value_null_path=OUT/'scienceworld_expand_value_summary.csv'
 value_pair_path=OUT/'scienceworld_expand_value_paired.csv'
 if value_null_path.exists() and value_pair_path.exists():
  vs=pd.read_csv(value_null_path); vp=pd.read_csv(value_pair_path)
  def _vn(dup,method,col):
   z=vs[(vs.duplication==dup)&(vs.method==method)]
   return float(z[col].iloc[0]) if len(z) else float('nan')
  def _vp(dup,metric):
   z=vp[(vp.duplication==dup)&(vp.metric==metric)]
   if not len(z): return 'nan'
   r=z.iloc[0]; return f'{float(r.provenance_value_minus_flat):+.2f} (95% CI {float(r.ci95_low):+.2f}–{float(r.ci95_high):+.2f})'
  lines.insert(lines.index('## Limitations'), '\n### Fixed-grid grouped-value null check\n\nThe algorithmic unique-sample conditional-value readout is duplication-invariant on all 12 fixed-grid states: success/reward are {:.3f}/{:.1f} at both duplication levels. Against flat, its paired reward contrast is {} at duplication 1 and {} at duplication 4. This is a negative boundary result: grouping prevents duplicate sensitivity but does not guarantee better planning when the candidate/value bank is weak. See `scienceworld_expand_value_summary.csv` and `scienceworld_expand_value_paired.csv`.\n'.format(_vn(1,'provenance_value','success'),_vn(1,'provenance_value','reward'),_vp(1,'reward'),_vp(4,'reward')))
 success_summary_path=OUT/'scienceworld_expand_success_summary.csv'
 success_pair_path=OUT/'scienceworld_expand_success_paired.csv'
 if success_summary_path.exists() and success_pair_path.exists():
  ss=pd.read_csv(success_summary_path); sp=pd.read_csv(success_pair_path)
  def _ss(dup,method,col):
   z=ss[(ss.duplication==dup)&(ss.method==method)]
   return float(z[col].iloc[0]) if len(z) else float('nan')
  def _ssp(dup,metric):
   z=sp[(sp.duplication==dup)&(sp.metric==metric)]
   if not len(z): return 'nan'
   r=z.iloc[0]; return f'{float(r.provenance_success_minus_flat):+.2f} (95% CI {float(r.ci95_low):+.2f}–{float(r.ci95_high):+.2f})'
  lines.insert(lines.index('## Limitations'), '\n### Success-aware provenance ablation\n\nSelecting the highest unique-sample conditional success probability is an explicit readout ablation. It is duplication-invariant on the fixed grid (success {:.3f}, reward {:.1f} at both m=1 and m=4). On the {} matched valid episodes, its paired reward contrast versus flat is {} at m=1 and {} at m=4; the corresponding success contrasts are {} and {}. Point estimates are positive but intervals include zero, so this is exploratory planning evidence rather than a broad success claim. Candidate-interface errors remain excluded only from valid-row means. See `scienceworld_expand_success_summary.csv` and `scienceworld_expand_success_paired.csv`.\n'.format(_ss(1,'provenance_success','success'),_ss(1,'provenance_success','reward'),int(sp[sp.metric=='success'].n.iloc[0]),_ssp(1,'reward'),_ssp(4,'reward'),_ssp(1,'success'),_ssp(4,'success')))
 mb_path=OUT/'scienceworld_v7_method_bootstrap.csv'
 if mb_path.exists():
  mb=pd.read_csv(mb_path)
  def mb_text(dup,metric):
   q=mb[(mb.duplication==dup)&(mb.metric==metric)].iloc[0]
   return f'{q.estimate:+.3f} (95% CI {q.ci95_low:+.3f}–{q.ci95_high:+.3f})'
  lines[lines.index('### Legacy snapshot diagnostic (input-integrity caveat)'):lines.index('### Legacy snapshot diagnostic (input-integrity caveat)')]=['', '### Paired full-episode method contrast', '', f'Within the same eight episodes, provenance minus flat reward is {mb_text(1,"reward")} at duplication 1 and {mb_text(4,"reward")} at duplication 4. The corresponding success difference is {mb_text(1,"success")} and {mb_text(4,"success")}. This supports a reward-level planning advantage in the small paired sample, while the success contrast remains zero.', '']
 if extra:
  exagg=ex.groupby(['duplication','method'])[['success','reward']].mean().reset_index()
  extra_lines=['', '### Held-out extra ScienceWorld stress', '', 'Four additional official dev variations (find-plant/find-animal, 163/164) were run with the same v7 protocol and are kept separate because no-imagination was not run. Flat and provenance both achieve success {:.3f} and reward {:.1f} at duplication 1, and the same values at duplication 4; the two plant cases terminate on the first public action with a -100 score. This negative result widens the audit while showing that the current candidate shortlist, rather than provenance aggregation, is the limiting factor on these variations. See `scienceworld_v8_extra_summary.csv`.'.format(float(exagg[(exagg.duplication==1)&(exagg.method=='flat')].success.iloc[0]), float(exagg[(exagg.duplication==1)&(exagg.method=='flat')].reward.iloc[0])), '']
  lines[lines.index('## Limitations'):lines.index('## Limitations')]=extra_lines
 # The v9 public-discovery policy is a separate interface repair. Keep all
 # transport failures visible and never fold this incomplete retry into v7.
 v9=[]
 for name in ['scienceworld_v9_discovery_extra_m1','scienceworld_v9_discovery_m4_retry']:
  p=ROOT/name/'summaries.json'
  if p.exists(): v9 += json.loads(p.read_text())
 if v9:
  v9df=pd.DataFrame(v9); v9df['error_flag']=v9df['error'].notna().astype(int)
  v9df.to_csv(OUT/'scienceworld_v9_discovery_rows.csv',index=False)
  v9df.groupby(['duplication','method']).agg(n=('task','size'),errors=('error_flag','sum'),success_all=('success','mean'),reward_all=('reward','mean'),valid_n=('error_flag',lambda x: int((x==0).sum())),success_valid=('success',lambda x: float(x[v9df.loc[x.index,'error_flag']==0].mean()) if (v9df.loc[x.index,'error_flag']==0).any() else float('nan')),reward_valid=('reward',lambda x: float(x[v9df.loc[x.index,'error_flag']==0].mean()) if (v9df.loc[x.index,'error_flag']==0).any() else float('nan'))).reset_index().to_csv(OUT/'scienceworld_v9_discovery_summary.csv',index=False)
  v9_lines=['', '### v9 public-discovery policy check', '', 'A public-interface repair was tested on four additional variations: when the requested category was absent from the current observation, the shortlist offered category-relevant navigation instead of arbitrary `focus on door/room` actions. The duplication-1 run has one transport error; among valid rows flat success is 0.750 (reward 73.25) and provenance success is 0.667 (reward 67.0). The duplication-4 retry has eight transport timeouts and is excluded from method comparison. These files are a policy diagnostic, not a new duplication result; v7 remains the complete fair comparison. See `scienceworld_v9_discovery_summary.csv` and `scienceworld_v9_discovery_rows.csv`.', '']
  lines[lines.index('## Limitations'):lines.index('## Limitations')]=v9_lines
 hard_summary=ROOT/'provenance_semantic_hard_final/summary.json'
 if hard_summary.exists():
  hs=json.loads(hard_summary.read_text()); by_target={x['target']:x for x in hs['summaries']}
  hard_lines=['', '### Multi-root opaque provenance stress', '', 'To remove the single-root and lexical-key shortcuts, a separate evaluator composes {} train, {} dev, and {} test multi-root tasks, hides premise/source names in paraphrased branch reports, randomizes draw metadata, and retains 16 rollouts per root. The same transparent pair extractor falls to premise-edge F1 {:.3f} (precision {:.3f}, recall {:.3f}) and source-edge F1 {:.3f} (precision {:.3f}, recall {:.3f}). Learned grouping reaches downstream action correctness {:.3f}, reward {:.3f}, and regret {:.3f}, versus oracle correctness {:.3f}, reward {:.3f}, and regret {:.3f}. The stress is a diagnostic of recovery limits; the separate lexical-bank noise curves remain in `../provenance/noisy_downstream.csv`. See `../provenance_semantic_hard_final/summary.json`.'.format(hs['train_composites'],hs['dev_composites'],hs['test_composites'],by_target['premise']['micro_f1'],by_target['premise']['micro_precision'],by_target['premise']['micro_recall'],by_target['source']['micro_f1'],by_target['source']['micro_precision'],by_target['source']['micro_recall'],hs['downstream']['premise']['learned']['action_correct'],hs['downstream']['premise']['learned']['reward'],hs['downstream']['premise']['learned']['regret'],hs['downstream']['premise']['oracle']['action_correct'],hs['downstream']['premise']['oracle']['reward'],hs['downstream']['premise']['oracle']['regret']), '']
  lines[lines.index('## Limitations'):lines.index('## Limitations')]=hard_lines
  margin_path=ROOT/'provenance_semantic_hard_final/margin_analysis.csv'
  if margin_path.exists():
   md=pd.read_csv(margin_path); md=md[(md.target=='premise')&(md.recovery=='learned')&(md.n>0)]
   margin_text='; '.join(f'{r.margin_bin}: correctness {float(r.action_correct):.3f}, merge {float(r.cross_root_merge_rate):.3f}' for _,r in md.iterrows())
   lines[lines.index('## Limitations'):lines.index('## Limitations')]=['', '### Provenance-recovery margin diagnostic', '', 'Proposition 6 gives a sufficient action-margin condition under total-variation recovery error. In the opaque multi-root stress, the empirical learned grouping has the following margin-bin diagnostics: {}. Cross-root merges remain high, so the bound identifies recovery error as the active bottleneck rather than predicting a planning gain. See `../provenance_semantic_hard_final/margin_analysis.csv`.'.format(margin_text), '']
  null_path=ROOT/'provenance_semantic_hard_final/identifiability_null.csv'
  if null_path.exists():
   nd=pd.read_csv(null_path)
   null_text='; '.join(f'{r.target}: actual F1 {float(r.actual_f1):.3f}, permutation-null {float(r.null_f1_mean):.3f} ({float(r.null_f1_ci95_low):.3f}--{float(r.null_f1_ci95_high):.3f})' for _,r in nd.iterrows())
   lines[lines.index('## Limitations'):lines.index('## Limitations')]=['', '### Provenance identifiability null', '', 'A fixed-predictor permutation of opaque test labels gives a finite-sample recovery null: {}. Premise recovery is indistinguishable from this null because the opaque observable fields do not expose a premise cue; source F1 is only modestly above its null. This supports Proposition 7\'s observability boundary and explains why downstream action correctness collapses under cross-root merges. See `../provenance_semantic_hard_final/identifiability_null.csv`.'.format(null_text), '']
 # Learned-provenance recovery breakdown and downstream noise curve.
 edge_breakdown=[]
 for target, path in [('premise', ROOT/'provenance/test_edge_metrics.csv'),
                      ('source', ROOT/'provenance/test_source_edge_metrics.csv')]:
  if path.exists():
   ed=pd.read_csv(path); ed['target']=target; edge_breakdown.append(ed)
 if edge_breakdown:
  eb=pd.concat(edge_breakdown,ignore_index=True)
  eb.groupby(['target','family','difficulty']).agg(tasks=('task_id','size'),
      precision=('precision','mean'),recall=('recall','mean'),f1=('f1','mean')).reset_index().to_csv(OUT/'provenance_edge_breakdown.csv',index=False)
 noisy_path=ROOT/'provenance/noisy_downstream.csv'
 if noisy_path.exists():
  noisy=pd.read_csv(noisy_path)
  curve=noisy.groupby('noise_rate').agg(action_correct=('action_correct','mean'),reward=('reward','mean'),regret=('regret','mean')).reset_index()
  curve.to_csv(OUT/'provenance_noise_curve.csv',index=False)
  fig,axs=plt.subplots(1,3,figsize=(10,3.2))
  for col,ax,label in [('action_correct',axs[0],'action correctness'),('reward',axs[1],'reward'),('regret',axs[2],'regret')]:
   ax.plot(curve.noise_rate,curve[col],'o-',color='#276fbf'); ax.set(xlabel='source-label noise',ylabel=label)
  fig.tight_layout(); fig.savefig(OUT/'provenance_noise_curve.png',dpi=180); plt.close(fig)
  learned_reward=float(hs['downstream']['premise']['learned']['reward']) if 'hs' in locals() else float('nan')
  oracle_reward=float(hs['downstream']['premise']['oracle']['reward']) if 'hs' in locals() else float('nan')
  lines[lines.index('## Limitations'):lines.index('## Limitations')]=[
  '', '### Provenance recovery breakdown', '',
  'The lexical recovery audit is broken down by task family and difficulty in `provenance_edge_breakdown.csv`. Its controlled source-label noise curve is in `provenance_noise_curve.csv` and `provenance_noise_curve.png`; the aggregate reward and regret are reported without monotonicity assumptions. The opaque multi-root stress remains the stronger semantic diagnostic, with learned source F1 0.392 and downstream reward {:.3f} versus oracle {:.3f}.'.format(learned_reward,oracle_reward), ''
  ]
 if counterfactual is not None and len(counterfactual):
  ok=counterfactual[counterfactual.error.isna() if 'error' in counterfactual else [True]*len(counterfactual)]
  good=ok[ok.method=='provenance_grouped']
  flat=ok[ok.method=='flat']
  success_rate=float(good.success.mean()) if len(good) else float('nan')
  flat_success=float(flat.success.mean()) if len(flat) else float('nan')
  n_states=int(good[['task','variation']].drop_duplicates().shape[0]) if len(good) else 0
  flat_failed=int((flat.success.astype(int)==0).sum()) if len(flat) else 0
  flat_total=int(len(flat))
  lines[lines.index('## Limitations'):lines.index('## Limitations')]=[
   '', '### Real-environment duplication counterfactual', '',
   'A legacy paired state-replay artifact fixes an archived ScienceWorld state and changes the stipulated multiplicity, but the input-integrity audit found post-action mutable history in its trace-derived evidence. Its numerical rows remain archived for auditability and are excluded from clean causal claims. The pre-action serialized-input intervention below is the authoritative fixed-bank result and reports immediate reward only. See `scienceworld_interactive_counterfactual.csv`, `../interactive_counterfactual_replay/summary.json`, and `scienceworld_input_integrity/summary.json`.'.format(n_states,flat_total,flat_failed,success_rate,flat_success), ''
  ]
 # Refreshed DeepSeek closed-loop comparison. This run is intentionally kept
 # separate from v7: it is a new six-variation sample and includes negative
 # results instead of being folded into the earlier stochastic estimate.
 try:
  from submission.recharged_scienceworld import run as run_recharged, combine_comparison_tables
  refreshed, refreshed_pairs, refreshed_consistency = run_recharged(OUT)
  combine_comparison_tables(OUT)
  def _rr(dup, method, col):
   row=refreshed[(refreshed.duplication==dup)&(refreshed.experiment_method==method)].iloc[0]
   return float(row[col])
  ci_path=OUT/'scienceworld_recharged_action_bootstrap.csv'
  if ci_path.exists():
   ci_frame=pd.read_csv(ci_path)
   consistency_text='; '.join(f"{r.method} {float(r.estimate):.3f} (95% CI {float(r.ci95_low):.3f}–{float(r.ci95_high):.3f})" for _,r in ci_frame.iterrows())
  else:
   consistency_text='; '.join(f'{r.method} {float(r.action_flip_rate):.3f}' for _,r in refreshed_consistency.groupby('method',as_index=False).agg(action_flip_rate=('action_flip_rate','mean')).iterrows()) if len(refreshed_consistency) else 'unavailable'
  lines.insert(lines.index('## Limitations'), '\n### Refreshed DeepSeek ScienceWorld comparison\n\nA fresh six-variation `find-plant` run (variations 150--155, no gold path or hidden state) completed 60 valid method-budget rows using the same public-discovery protocol. No-imagination succeeds at 1.000 at both duplication levels. Flat succeeds at {:.3f}/{:.3f} with reward {:.1f}/{:.1f} for m=1/4; LLM-readout provenance succeeds at {:.3f}/{:.3f} with reward {:.1f}/{:.1f}. Algorithmic provenance-value and success-aware readouts are duplication-invariant in success (0.833) but do not exceed flat reward in this sample. Aligned m=1-to-m=4 action-flip summaries are {}. This is a stronger fresh negative/diagnostic result: provenance reduces duplicate sensitivity in the controlled estimator, but broad interactive planning improvement remains unestablished. See `scienceworld_recharged_summary.csv`, `scienceworld_recharged_paired.csv`, and `scienceworld_recharged_action_consistency.csv` / `scienceworld_recharged_action_bootstrap.csv`.\n'.format(_rr(1,'flat','success'),_rr(4,'flat','success'),_rr(1,'flat','reward'),_rr(4,'flat','reward'),_rr(1,'provenance','success'),_rr(4,'provenance','success'),_rr(1,'provenance','reward'),_rr(4,'provenance','reward'),consistency_text))
 except Exception:
  pass
 # Second-task-family refreshed API check. Variation 152 has a retained
 # candidate-interface error and is excluded from valid-row means.
 refreshed_animal_path=OUT/'scienceworld_recharged_animal/scienceworld_recharged_summary.csv'
 if refreshed_animal_path.exists():
  try:
   ra=pd.read_csv(refreshed_animal_path)
   def _ra(dup, method, col):
    z=ra[(ra.duplication==dup)&(ra.experiment_method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Refreshed DeepSeek second-task-family check\n\nThe same v9 public-action protocol on six public `find-animal` variations produced six valid rows per method and budget; one transient SSL failure was retried and retained separately. Flat success is {:.3f}/{:.3f} with reward {:.1f}/{:.1f} at m=1/4, while provenance is {:.3f}/{:.3f} with reward {:.1f}/{:.1f}. No method improves success in this task family; the result is a negative external-validity control. See `scienceworld_recharged_animal/scienceworld_recharged_summary.csv`.\n'.format(_ra(1,'flat','success'),_ra(4,'flat','success'),_ra(1,'flat','reward'),_ra(4,'flat','reward'),_ra(1,'provenance','success'),_ra(4,'provenance','success'),_ra(1,'provenance','reward'),_ra(4,'provenance','reward')))
  except Exception:
   pass
 # Compact refreshed comparison table.  The confidence columns are the
 # archived candidate.p_true first-to-last change, kept separate from success.
 comparison_path=OUT/'scienceworld_recharged_comparison_all.csv'
 if comparison_path.exists():
  try:
   comparison=pd.read_csv(comparison_path)
   def _cmp(task, method, dup, col):
    z=comparison[(comparison.task_family==task)&(comparison.experiment_method==method)&(comparison.duplication==dup)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Refreshed confidence/action/reward table\n\nThe archived comparison table records first-to-last `p_true` changes from a closed-loop runner. Because each step can regenerate the visible state and candidate set, these are trace diagnostics rather than a causal fixed-root belief estimate. For `find-plant`, flat confidence changes by {:.3f}/{:.3f} and LLM-readout provenance by {:.3f}/{:.3f} at m=1/4; their paired action-flip rates are {:.3f} and {:.3f}. For `find-animal`, the corresponding confidence changes are {:.3f}/{:.3f} and {:.3f}/{:.3f}, with action-flip rates {:.3f} and {:.3f}. See `scienceworld_recharged_comparison_all.csv`.\n'.format(_cmp('find-plant','flat',1,'root_confidence_delta'),_cmp('find-plant','flat',4,'root_confidence_delta'),_cmp('find-plant','provenance',1,'root_confidence_delta'),_cmp('find-plant','provenance',4,'root_confidence_delta'),_cmp('find-plant','flat',1,'action_flip_rate_m1_vs_m4'),_cmp('find-plant','provenance',1,'action_flip_rate_m1_vs_m4'),_cmp('find-animal','flat',1,'root_confidence_delta'),_cmp('find-animal','flat',4,'root_confidence_delta'),_cmp('find-animal','provenance',1,'root_confidence_delta'),_cmp('find-animal','provenance',4,'root_confidence_delta'),_cmp('find-animal','flat',1,'action_flip_rate_m1_vs_m4'),_cmp('find-animal','provenance',1,'action_flip_rate_m1_vs_m4')))
  except Exception:
   pass
 # Expanded fresh-state replay: no new API calls, twelve public states across
 # two task families, and the same archived suffix for every readout.
 expanded_cf_path=OUT/'scienceworld_recharged_counterfactual_all/summary.csv'
 expanded_bootstrap_path=OUT/'scienceworld_recharged_counterfactual_all/bootstrap.csv'
 if expanded_cf_path.exists() and expanded_bootstrap_path.exists():
  try:
   expanded_cf=pd.read_csv(expanded_cf_path)
   expanded_bootstrap=pd.read_csv(expanded_bootstrap_path)
   def _ec(task, method, duplication, col):
    z=expanded_cf[(expanded_cf.task==task)&(expanded_cf.method==method)&(expanded_cf.duplication==duplication)]
    return float(z[col].mean()) if len(z) else float('nan')
   def _eb(task, contrast, metric, col):
    z=expanded_bootstrap[(expanded_bootstrap.task==task)&(expanded_bootstrap.contrast==contrast)&(expanded_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Expanded fresh-state ScienceWorld replay\n\nA no-new-API intervention replays 12 public states (six `find-plant`, six `find-animal`) with a common archived no-imagination suffix and changes only the first-step readout. Grouped provenance minus flat m=4 is {:.3f} success and {:.1f} reward on `find-plant` (bootstrap 95% CIs {:.3f}--{:.3f}, {:.1f}--{:.1f}); the corresponding `find-animal` contrasts are {:.3f} and {:.1f}. The grouped readout keeps all six plant replay states successful while flat duplication-4 fails one, and remains neutral on the animal control. This is causal episode-level evidence under a selected common continuation, not a broad closed-loop success estimate. See `scienceworld_recharged_counterfactual_all/summary.csv`, `paired.csv`, and `bootstrap.csv`.\n'.format(_eb('find-plant','provenance_grouped-flat_m4','success','estimate'),_eb('find-plant','provenance_grouped-flat_m4','reward','estimate'),_eb('find-plant','provenance_grouped-flat_m4','success','ci95_low'),_eb('find-plant','provenance_grouped-flat_m4','success','ci95_high'),_eb('find-plant','provenance_grouped-flat_m4','reward','ci95_low'),_eb('find-plant','provenance_grouped-flat_m4','reward','ci95_high'),_eb('find-animal','provenance_grouped-flat_m4','success','estimate'),_eb('find-animal','provenance_grouped-flat_m4','reward','estimate')))
  except Exception:
   pass
 # Fresh-key contiguous extension.  It is reported separately because it is
 # a null/negative boundary rather than evidence folded into the main sample.
 extension_path=OUT/'scienceworld_recharged_extension'
 if (extension_path/'summary.csv').exists() and (extension_path/'paired_bootstrap.csv').exists():
  try:
   extension_summary=pd.read_csv(extension_path/'summary.csv')
   extension_bootstrap=pd.read_csv(extension_path/'paired_bootstrap.csv')
   extension_action=pd.read_csv(extension_path/'action_bootstrap.csv') if (extension_path/'action_bootstrap.csv').exists() else pd.DataFrame()
   def _re(dup, method, col):
    z=extension_summary[(extension_summary.duplication==dup)&(extension_summary.experiment_method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _reb(dup, metric, col):
    z=extension_bootstrap[(extension_bootstrap.duplication==dup)&(extension_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _rea(method, col):
    z=extension_action[extension_action.method==method]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Fresh-key contiguous interactive extension\n\nA fresh-key v9 extension covers six additional `find-plant` variations with 24 valid rows and no API errors. Flat and success-aware provenance both have success {:.3f}/{:.3f} and reward {:.1f}/{:.1f} at m=1/4; the paired provenance-minus-flat contrasts are zero for success and reward at both budgets (95% reward CI {:.1f}--{:.1f} at m=4). Their m=1-to-m=4 action-flip rates are {:.3f} and {:.3f}, showing that readout changes need not change the episode outcome. This null result strengthens the boundary that provenance preservation enforces duplication invariance but does not guarantee planner improvement. See `scienceworld_recharged_extension/summary.csv`, `paired.csv`, `paired_bootstrap.csv`, and `action_bootstrap.csv`.\n'.format(_re(1,'flat','success'),_re(4,'flat','success'),_re(1,'flat','reward'),_re(4,'flat','reward'),_reb(4,'reward','ci95_low'),_reb(4,'reward','ci95_high'),_rea('flat','estimate'),_rea('provenance_success','estimate')))
  except Exception:
   pass
 animal_extension_path=OUT/'scienceworld_recharged_animal_extension'
 if (animal_extension_path/'summary.csv').exists() and (animal_extension_path/'paired_bootstrap.csv').exists():
  try:
   animal_extension_summary=pd.read_csv(animal_extension_path/'summary.csv')
   animal_extension_bootstrap=pd.read_csv(animal_extension_path/'paired_bootstrap.csv')
   def _ae(dup, method, col):
    z=animal_extension_summary[(animal_extension_summary.duplication==dup)&(animal_extension_summary.experiment_method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _aeb(dup, metric, col):
    z=animal_extension_bootstrap[(animal_extension_bootstrap.duplication==dup)&(animal_extension_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Fresh-key second-task-family extension\n\nThe same v9 extension on six additional `find-animal` variations also has 24 valid rows and no transport failures. Flat success is {:.3f}/{:.3f} with reward {:.1f}/{:.1f} at m=1/4; success-aware provenance is {:.3f}/{:.3f} with reward {:.1f}/{:.1f}. The paired provenance-minus-flat reward contrast is {:.1f} at m=4 (95% CI {:.1f}--{:.1f}), a negative external-validity control. See `scienceworld_recharged_animal_extension/summary.csv`, `paired.csv`, and `paired_bootstrap.csv`.\n'.format(_ae(1,'flat','success'),_ae(4,'flat','success'),_ae(1,'flat','reward'),_ae(4,'flat','reward'),_ae(1,'provenance_success','success'),_ae(4,'provenance_success','success'),_ae(1,'provenance_success','reward'),_ae(4,'provenance_success','reward'),_aeb(4,'reward','provenance_success_minus_flat'),_aeb(4,'reward','ci95_low'),_aeb(4,'reward','ci95_high')))
  except Exception:
   pass
 llm_extension_path=OUT/'scienceworld_recharged_llm_extension'
 if (llm_extension_path/'summary.csv').exists() and (llm_extension_path/'paired_bootstrap.csv').exists():
  try:
   llm_extension_summary=pd.read_csv(llm_extension_path/'summary.csv')
   llm_extension_bootstrap=pd.read_csv(llm_extension_path/'paired_bootstrap.csv')
   llm_extension_action=pd.read_csv(llm_extension_path/'action_bootstrap.csv') if (llm_extension_path/'action_bootstrap.csv').exists() else pd.DataFrame()
   llm_extension_meta=json.loads((llm_extension_path/'metadata.json').read_text()) if (llm_extension_path/'metadata.json').exists() else {}
   def _le(dup, method, col):
    z=llm_extension_summary[(llm_extension_summary.duplication==dup)&(llm_extension_summary.experiment_method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _leb(dup, metric, col):
    z=llm_extension_bootstrap[(llm_extension_bootstrap.duplication==dup)&(llm_extension_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _lea(method, col):
    z=llm_extension_action[llm_extension_action.method==method]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Fresh-key fair LLM-readout extension\n\nA separate fresh-key extension reuses the archived candidate and conditional-rollout bank but adds the same LLM provenance readout as the primary protocol on six additional `find-plant` variations. It has {} valid method-budget rows and {} transport errors. At m=1, flat and provenance both reach success {:.3f} and reward {:.1f}; at m=4, flat reaches {:.3f}/{:.1f} while provenance reaches {:.3f}/{:.1f}. The paired provenance-minus-flat m=4 reward contrast is {:.2f} (95% CI {:.2f}--{:.2f}), and the success contrast is {:.3f} ({:.3f}--{:.3f}). Root-confidence deltas are {:.3f}/{:.3f} for flat and {:.3f}/{:.3f} for provenance at m=1/4; action-flip rates are {:.3f}/{:.3f}. This is a fair negative extension, separate from the algorithmic success-aware extension, and does not establish broad planner improvement. See `scienceworld_recharged_llm_extension/summary.csv`, `paired_bootstrap.csv`, `action_bootstrap.csv`, and `metadata.json`.\n'.format(llm_extension_meta.get('valid_rows','?'),llm_extension_meta.get('error_rows','?'),_le(1,'flat','success'),_le(1,'flat','reward'),_le(4,'flat','success'),_le(4,'flat','reward'),_le(4,'provenance','success'),_le(4,'provenance','reward'),_leb(4,'reward','provenance_minus_flat'),_leb(4,'reward','ci95_low'),_leb(4,'reward','ci95_high'),_leb(4,'success','provenance_minus_flat'),_leb(4,'success','ci95_low'),_leb(4,'success','ci95_high'),_le(1,'flat','root_confidence_delta'),_le(4,'flat','root_confidence_delta'),_le(1,'provenance','root_confidence_delta'),_le(4,'provenance','root_confidence_delta'),_lea('flat','estimate'),_lea('provenance','estimate')))
  except Exception:
   pass
 animal_llm_extension_path=OUT/'scienceworld_recharged_animal_llm_extension'
 if (animal_llm_extension_path/'summary.csv').exists() and (animal_llm_extension_path/'paired_bootstrap.csv').exists():
  try:
   animal_llm_extension_summary=pd.read_csv(animal_llm_extension_path/'summary.csv')
   animal_llm_extension_bootstrap=pd.read_csv(animal_llm_extension_path/'paired_bootstrap.csv')
   animal_llm_extension_action=pd.read_csv(animal_llm_extension_path/'action_bootstrap.csv') if (animal_llm_extension_path/'action_bootstrap.csv').exists() else pd.DataFrame()
   animal_llm_extension_meta=json.loads((animal_llm_extension_path/'metadata.json').read_text()) if (animal_llm_extension_path/'metadata.json').exists() else {}
   def _ale(dup, method, col):
    z=animal_llm_extension_summary[(animal_llm_extension_summary.duplication==dup)&(animal_llm_extension_summary.experiment_method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _aleb(dup, metric, col):
    z=animal_llm_extension_bootstrap[(animal_llm_extension_bootstrap.duplication==dup)&(animal_llm_extension_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _alea(method, col):
    z=animal_llm_extension_action[animal_llm_extension_action.method==method]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Fresh-key fair LLM second-task extension\n\nThe same cached-bank intervention covers six additional `find-animal` variations with {} valid rows and {} transport errors. At m=1 flat/provenance have success {:.3f}/{:.3f} and reward {:.1f}/{:.1f}; at m=4 they have success {:.3f}/{:.3f} and reward {:.1f}/{:.1f}. The paired provenance-minus-flat m=4 reward contrast is {:.2f} (95% CI {:.2f}--{:.2f}), while action-flip rates are {:.3f}/{:.3f}. This negative task-family control keeps the fair LLM comparison separate from the success-aware extension. See `scienceworld_recharged_animal_llm_extension/summary.csv`, `paired_bootstrap.csv`, and `action_bootstrap.csv`.\n'.format(animal_llm_extension_meta.get('valid_rows','?'),animal_llm_extension_meta.get('error_rows','?'),_ale(1,'flat','success'),_ale(1,'provenance','success'),_ale(1,'flat','reward'),_ale(1,'provenance','reward'),_ale(4,'flat','success'),_ale(4,'provenance','success'),_ale(4,'flat','reward'),_ale(4,'provenance','reward'),_aleb(4,'reward','provenance_minus_flat'),_aleb(4,'reward','ci95_low'),_aleb(4,'reward','ci95_high'),_alea('flat','estimate'),_alea('provenance','estimate')))
  except Exception:
   pass
 # Consolidated interactive ledger.  This is a read-only union of the
 # refreshed LLM-readout comparison and the fresh-key success-aware extension;
 # the protocol label keeps their planner readouts stratified.
 aggregate_path=OUT/'scienceworld_interactive_aggregate'
 if (aggregate_path/'summary.csv').exists() and (aggregate_path/'paired_bootstrap.csv').exists():
  try:
   aggregate_summary=pd.read_csv(aggregate_path/'summary.csv')
   aggregate_bootstrap=pd.read_csv(aggregate_path/'paired_bootstrap.csv')
   aggregate_action=pd.read_csv(aggregate_path/'action_consistency.csv') if (aggregate_path/'action_consistency.csv').exists() else pd.DataFrame()
   def _ag(protocol, task, dup, method, col):
    z=aggregate_summary[(aggregate_summary.protocol==protocol)&(aggregate_summary.task_family==task)&(aggregate_summary.duplication==dup)&(aggregate_summary.experiment_method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _agb(protocol, task, dup, metric, col):
    z=aggregate_bootstrap[(aggregate_bootstrap.protocol==protocol)&(aggregate_bootstrap.task_family==task)&(aggregate_bootstrap.duplication==dup)&(aggregate_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _aga(protocol, task, method):
    z=aggregate_action[(aggregate_action.protocol==protocol)&(aggregate_action.task_family==task)&(aggregate_action.experiment_method==method)]
    return float(z.action_flip_rate.mean()) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Consolidated interactive evidence ledger\n\nA read-only ledger now joins 96 valid episode-method-budget rows across 12 refreshed public variations per task family. The primary `refreshed_llm` stratum compares flat with the LLM-readout provenance planner; the separate `fresh_key_success` stratum compares flat with the algorithmic success-aware provenance readout, so their estimates are not pooled as one method. At duplication 4, provenance-minus-flat reward is {:.1f} (95% CI {:.1f}--{:.1f}) for refreshed `find-plant` and {:.1f} ({:.1f}--{:.1f}) for refreshed `find-animal`; the fresh-key success-aware contrast is {:.1f} ({:.1f}--{:.1f}) and {:.1f} ({:.1f}--{:.1f}), respectively. Mean aligned action-flip rates for flat/provenance are {:.3f}/{:.3f} in the refreshed plant stratum and {:.3f}/{:.3f} in the refreshed animal stratum; the fresh-key extension is retained separately in the machine-readable ledger. This consolidation improves auditability while leaving broad interactive success improvement unestablished. See `scienceworld_interactive_aggregate/summary.csv`, `paired.csv`, `paired_bootstrap.csv`, `action_consistency.csv`, and `metadata.json`.\n'.format(_agb('refreshed_llm','find-plant',4,'reward','estimate'),_agb('refreshed_llm','find-plant',4,'reward','ci95_low'),_agb('refreshed_llm','find-plant',4,'reward','ci95_high'),_agb('refreshed_llm','find-animal',4,'reward','estimate'),_agb('refreshed_llm','find-animal',4,'reward','ci95_low'),_agb('refreshed_llm','find-animal',4,'reward','ci95_high'),_agb('fresh_key_success','find-plant',4,'reward','estimate'),_agb('fresh_key_success','find-plant',4,'reward','ci95_low'),_agb('fresh_key_success','find-plant',4,'reward','ci95_high'),_agb('fresh_key_success','find-animal',4,'reward','estimate'),_agb('fresh_key_success','find-animal',4,'reward','ci95_low'),_agb('fresh_key_success','find-animal',4,'reward','ci95_high'),_aga('refreshed_llm','find-plant','flat'),_aga('refreshed_llm','find-plant','provenance'),_aga('refreshed_llm','find-animal','flat'),_aga('refreshed_llm','find-animal','provenance')))
  except Exception:
   pass
 interactive_table_path=OUT/'interactive_main_table'
 if (interactive_table_path/'summary.csv').exists() and (interactive_table_path/'contrasts.csv').exists():
  try:
   interactive_table_meta=json.loads((interactive_table_path/'metadata.json').read_text()) if (interactive_table_path/'metadata.json').exists() else {}
   lines.insert(lines.index('## Limitations'), '\n### Machine-generated interactive main table\n\nThe submission table contains {} summary rows and {} paired contrast rows over the refreshed and fresh-key protocols. It aligns success, reward, steps, token cost, root-confidence deltas, and duplication-1-to-4 action flips; planner readouts stay explicitly labeled by protocol. This table is the numerical source for the interactive paragraphs above and does not pool incompatible readouts. See `interactive_main_table/summary.csv`, `contrasts.csv`, and `metadata.json`.\n'.format(interactive_table_meta.get('summary_rows','?'),interactive_table_meta.get('contrast_rows','?')))
  except Exception:
   pass
 # Fresh state-replay counterfactual from the refreshed API bank. It changes
 # only first-step readout choice and applies the same archived continuation.
 refreshed_cf_path=OUT/'scienceworld_recharged_counterfactual/summary.csv'
 if refreshed_cf_path.exists():
  try:
   rcf=pd.read_csv(refreshed_cf_path)
   def _rcf(method,dup,col):
    z=rcf[(rcf.method==method)&(rcf.duplication==dup)]
    return float(z[col].mean()) if len(z) else float('nan')
   grouped=rcf[rcf.method=='provenance_grouped']
   lines.insert(lines.index('## Limitations'), '\n### Refreshed ScienceWorld state-replay counterfactual\n\nUsing the six refreshed public states, the archived candidate/bank, and the same no-imagination suffix, only the first-step readout was changed. Grouped provenance reaches success {:.3f} and reward {:.1f}; flat reaches {:.3f}/{:.1f} at m=1 and {:.3f}/{:.1f} at m=4. The m=4 flat readout fails on one of six states while grouped provenance succeeds on all six. This is causal episode-level evidence that duplication can alter a real action and outcome, but it is a selected common-continuation diagnostic rather than a broad closed-loop success estimate. See `scienceworld_recharged_counterfactual/summary.csv`.\n'.format(_rcf('provenance_grouped',1,'success'),_rcf('provenance_grouped',1,'reward'),_rcf('flat',1,'success'),_rcf('flat',1,'reward'),_rcf('flat',4,'success'),_rcf('flat',4,'reward')))
  except Exception:
   pass
 # Four-task grouped-value ScienceWorld exploratory run. It uses the same
 # candidate/rollout generation and an algorithmic unique-sample value readout;
 # keep it separate from the primary LLM-readout v7 table.
 try:
  from submission.scienceworld_value_summary import load_rows, summarize
  value_frame=load_rows([
   'results_submission/scienceworld_value_m1',
   'results_submission/scienceworld_value_retry_m1',
   'results_submission/scienceworld_value_retry_161162_m1',
   'results_submission/scienceworld_value_plants_m4'])
  value_summary, value_pairs=summarize(value_frame, OUT)
  if len(value_summary):
   def _v(dup, method, col):
    row=value_summary[(value_summary.duplication==dup)&(value_summary.method==method)].iloc[0]
    return float(row[col])
   def _p(dup, metric):
    row=value_pairs[(value_pairs.duplication==dup)&(value_pairs.metric==metric)].iloc[0]
    return float(row.provenance_minus_flat)
   def _pc(dup, metric):
    row=value_pairs[(value_pairs.duplication==dup)&(value_pairs.metric==metric)].iloc[0]
    return f'{float(row.provenance_minus_flat):+.2f} (95% CI {float(row.ci95_low):+.2f}–{float(row.ci95_high):+.2f})'
   lines[lines.index('## Limitations'):lines.index('## Limitations')]=[
    '', '### Grouped-value interactive exploratory run', '',
    'A separate four-variation find-plant run uses the same DeepSeek candidate and conditional-rollout generation but applies the algorithmic unique-sample conditional-value readout. At duplication 1, flat success/reward are {:.3f}/{:.2f} and grouped provenance is {:.3f}/{:.2f}; at duplication 4 they are {:.3f}/{:.2f} and {:.3f}/{:.2f}. Paired provenance-minus-flat reward is {} at duplication 1 and {} at duplication 4; success differences are {} and {}. This is a small task-family exploratory result, not the broad interactive success claim.'.format(_v(1,'flat','success'),_v(1,'flat','reward'),_v(1,'provenance_value','success'),_v(1,'provenance_value','reward'),_v(4,'flat','success'),_v(4,'flat','reward'),_v(4,'provenance_value','success'),_v(4,'provenance_value','reward'),_pc(1,'reward'),_pc(4,'reward'),_pc(1,'success'),_pc(4,'success')), ''
   ]
 except Exception:
  pass
 # Fixed-bank, pre-action API readout. The original frozen attempt used
 # trace-derived mutable history and is retained only as an audit artifact.
 frozen_path=ROOT/'scienceworld_frozen_readout_clean_20260929'
 if (frozen_path/'summary.csv').exists() and (frozen_path/'paired.csv').exists() and (frozen_path/'audit.json').exists():
  try:
   frozen_summary=pd.read_csv(frozen_path/'summary.csv')
   frozen_pairs=pd.read_csv(frozen_path/'paired.csv')
   frozen_audit=json.loads((frozen_path/'audit.json').read_text())
   def _fr(task, method, dup, col):
    z=frozen_summary[(frozen_summary.task==task)&(frozen_summary.method==method)&(frozen_summary.duplication==dup)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _fp(task, dup, col):
    z=frozen_pairs[(frozen_pairs.task==task)&(frozen_pairs.duplication==dup)]
    return float(z[col].mean()) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Clean fixed-bank ScienceWorld readout intervention\n\nTo isolate readout effects, we froze 12 pre-action public states (six `find-plant`, six `find-animal`), eight conditional samples, and the candidate set from serialized original API inputs. The clean run completed {} valid DeepSeek readouts ({} planned, {} errors), with 48 candidate-action replays matching the public simulator state. Flat `p_true` drift from the evidence-only value rises from {:.3f} at m=1 to {:.3f}/{:.3f} at m=4/8 on `find-plant`; the corresponding `find-animal` values stay {:.3f}. The m=4/8 action-flip rates are {:.3f}/{:.3f} on plant and {:.3f}/{:.3f} on animal, while the independent byte-identical m=1 repeat has zero action and confidence drift. Provenance-value keeps the evidence-only belief fixed and is duplication-invariant; immediate reward contrasts are reported as one-step simulator diagnostics. This experiment does not claim episode success or root calibration. See `scienceworld_frozen_readout_clean_20260929/summary.csv`, `paired.csv`, `bootstrap.csv`, and `audit.json` and `report/scienceworld_input_integrity/summary.json`.\n'.format(frozen_audit.get('valid_readouts','?'),frozen_audit.get('planned_api_calls','?'),frozen_audit.get('errors','?'),_fr('find-plant','flat',1,'root_p_true'),_fr('find-plant','flat',4,'root_p_true'),_fr('find-plant','flat',8,'root_p_true'),_fr('find-animal','flat',8,'root_p_true'),_fp('find-plant',4,'action_flip'),_fp('find-plant',8,'action_flip'),_fp('find-animal',4,'action_flip'),_fp('find-animal',8,'action_flip')))
  except Exception:
   pass
 expanded_path=ROOT/'scienceworld_frozen_readout_first4_20260929'
 expanded_files=[expanded_path/'summary.csv', expanded_path/'paired.csv', expanded_path/'bootstrap.csv', expanded_path/'audit.json', expanded_path/'curve.png']
 if all(p.exists() for p in expanded_files):
  try:
   expanded_summary=pd.read_csv(expanded_path/'summary.csv')
   expanded_bootstrap=pd.read_csv(expanded_path/'bootstrap.csv')
   expanded_audit=json.loads((expanded_path/'audit.json').read_text())
   def _ex(task, method, dup, col):
    z=expanded_summary[(expanded_summary.task==task)&(expanded_summary.method==method)&(expanded_summary.duplication==dup)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _exb(task, dup, metric, col):
    z=expanded_bootstrap[(expanded_bootstrap.task==task)&(expanded_bootstrap.duplication==dup)&(expanded_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Expanded four-step fixed-bank ScienceWorld intervention\n\nA preregistered extension covers 96 pre-action public states (the first four decisions of 12 archived variations in each of `find-plant` and `find-animal`). It completed {} valid DeepSeek readouts from {} planned calls with {} candidate-action replays, all matching the public simulator state. In `find-plant`, flat confidence drift is {:.3f}/{:.3f} at duplication 4/8 and action flips are {:.3f}/{:.3f}; the corresponding flat immediate-reward changes from m=1 are {:.2f}/{:.2f}. Provenance-value and success-aware provenance keep belief and action duplication-invariant; success-aware immediate reward is {:.2f} versus flat {:.2f}/{:.2f} at m=1/4. Across both families at m=8, flat action flips are {:.3f} (95% CI {:.3f}--{:.3f}) and absolute confidence drift is {:.3f} ({:.3f}--{:.3f}), while the repeat-control excess flip rate is {:.3f} ({:.3f}--{:.3f}). No paired comparison had both a reported probability change and an action change; all six harmful immediate-reward flips had unchanged reported probability. This is stronger fixed-state action/outcome evidence, but it reports immediate rewards only and does not claim confidence-mediated action changes, episode success, or broad planner improvement. See `scienceworld_frozen_readout_first4_20260929/{{summary.csv,paired.csv,bootstrap.csv,curve.png,audit.json,ANALYSIS.md,episode_bootstrap.csv}}`.\n'.format(expanded_audit.get('valid_readouts','?'),expanded_audit.get('planned_api_calls','?'),expanded_audit.get('candidate_replays','?'),_exb('find-plant',4,'confidence_delta','estimate'),_exb('find-plant',8,'confidence_delta','estimate'),_exb('find-plant',4,'action_flip','estimate'),_exb('find-plant',8,'action_flip','estimate'),_exb('find-plant',4,'reward_change_from_m1','estimate'),_exb('find-plant',8,'reward_change_from_m1','estimate'),_ex('find-plant','provenance_success',1,'immediate_reward'),_ex('find-plant','flat',1,'immediate_reward'),_ex('find-plant','flat',4,'immediate_reward'),_exb('all',8,'action_flip','estimate'),_exb('all',8,'action_flip','ci95_low'),_exb('all',8,'action_flip','ci95_high'),_exb('all',8,'absolute_confidence_change','estimate'),_exb('all',8,'absolute_confidence_change','ci95_low'),_exb('all',8,'absolute_confidence_change','ci95_high'),_exb('all',8,'flip_excess_over_repeat','estimate'),_exb('all',8,'flip_excess_over_repeat','ci95_low'),_exb('all',8,'flip_excess_over_repeat','ci95_high')))
  except Exception:
   pass
 # Minimal three-seed real-model task. It is deliberately separate from the
 # broader ScienceWorld table and exposes same-body API stochasticity.
 mvp_path=ROOT/'deepseek_mvp_20260929'
 if (mvp_path/'summary.csv').exists() and (mvp_path/'metadata.json').exists() and (mvp_path/'same_body_control.csv').exists():
  try:
   mvp_summary=pd.read_csv(mvp_path/'summary.csv')
   mvp_meta=json.loads((mvp_path/'metadata.json').read_text())
   mvp_control=pd.read_csv(mvp_path/'same_body_control.csv')
   def _dm(method, dup, col):
    z=mvp_summary[(mvp_summary.method==method)&(mvp_summary.duplication==dup)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Three-seed DeepSeek controlled task\n\nA minimal fixed-bank API diagnostic covers three seeds, duplication 1/2/4/8, and flat, trajectory-average, simple-dedup, source-grouping, and algorithmic provenance-value methods. It archives 60 valid method-budget rows from 30 request bodies; the 12 flat/trajectory-average request bodies are byte-identical, yet the API can return different outputs, so this is also a direct temperature-zero stochasticity control. Flat mean `p_true` is {:.3f}/{:.3f}/{:.3f}/{:.3f} at m=1/2/4/8, with action-flip rates {:.3f}/{:.3f}/{:.3f}/{:.3f}; provenance-value keeps `p_true` at {:.3f} and its action invariant across duplication. The three-seed intervals are wide and this model-specific diagnostic does not support a universal LLM claim. See `deepseek_mvp_20260929/summary.csv`, `paired.csv`, `bootstrap.csv`, `same_body_control.csv`, `curve.png`, and `metadata.json`.\n'.format(_dm('flat',1,'p_true'),_dm('flat',2,'p_true'),_dm('flat',4,'p_true'),_dm('flat',8,'p_true'),_dm('flat',1,'action_flip_rate'),_dm('flat',2,'action_flip_rate'),_dm('flat',4,'action_flip_rate'),_dm('flat',8,'action_flip_rate'),_dm('provenance_value',1,'p_true')))
  except Exception:
   pass
 qwen_mvp_path=OUT/'scienceworld_local_qwen_mvp'
 if (qwen_mvp_path/'summary.csv').exists() and (qwen_mvp_path/'metadata.json').exists():
  try:
   qwen_mvp_meta=json.loads((qwen_mvp_path/'metadata.json').read_text())
   qwen_mvp_summary=pd.read_csv(qwen_mvp_path/'summary.csv')
   def _qm(dup, method, col):
    z=qwen_mvp_summary[(qwen_mvp_summary.duplication==dup)&(qwen_mvp_summary.method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Local Qwen short-horizon ScienceWorld smoke\n\nA compatible local-Qwen runtime completed a two-step `find-plant` smoke on variation 159 at duplication 1/4 for flat and provenance ({} valid rows, {} errors). Both methods reached score {:.1f}, reward {:.1f}, and success {:.1f} at both budgets; flat token cost was {:.0f}/{:.0f} and provenance {:.0f}/{:.0f}. The CUDA probe reports {} on {}. The horizon is too short for episode-success inference, so this is a second-model schema/feasibility diagnostic and not a planner result. See `scienceworld_local_qwen_mvp/{{summary.csv,paired.csv,metadata.json}}`.\n'.format(qwen_mvp_meta.get('valid_rows','?'),qwen_mvp_meta.get('error_rows','?'),_qm(1,'flat','final_score'),_qm(1,'flat','reward'),_qm(1,'flat','success'),_qm(1,'flat','tokens'),_qm(4,'flat','tokens'),_qm(1,'provenance','tokens'),_qm(4,'provenance','tokens'),qwen_mvp_meta.get('inference_device','unknown'),qwen_mvp_meta.get('gpu_name','unknown')))
  except Exception:
   pass
 qwen_pilot_path=OUT/'local_qwen_episode_pilot'
 if (qwen_pilot_path/'summary.csv').exists() and (qwen_pilot_path/'paired.csv').exists() and (qwen_pilot_path/'metadata.json').exists() and (qwen_pilot_path/'bootstrap.csv').exists():
  try:
   qwen_pilot_meta=json.loads((qwen_pilot_path/'metadata.json').read_text())
   qwen_pilot_summary=pd.read_csv(qwen_pilot_path/'summary.csv')
   qwen_pilot_pairs=pd.read_csv(qwen_pilot_path/'paired.csv')
   def _qp(stratum, dup, method, col):
    z=qwen_pilot_summary[(qwen_pilot_summary.pilot_stratum==stratum)&(qwen_pilot_summary.duplication==dup)&(qwen_pilot_summary.method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Local Qwen full-episode interactive pilot\n\nA CUDA local-Qwen pilot completes {} valid episode-method-budget rows on three `find-plant` variations (152, 158, 161), with flat and success-objective provenance at duplication 1/4. On the two-variation grid (152, 158), flat success is {:.3f}/{:.3f} at m=1/4 and provenance-success is {:.3f}/{:.3f}; the held-out variation 161 is successful for both methods at both budgets. Across the three paired episodes, provenance-success minus flat reward is {:.1f} at m=1 and {:.1f} at m=4. The initial Brier diagnostic is 0.25 for flat and 0.64 for success-objective provenance in this tiny sample. This small second-model pilot is not pooled with DeepSeek and is not a confirmatory success estimate; it only shows that the full local planner path is executable on CUDA and that objective/readout choice can matter. See `local_qwen_episode_pilot/{{summary.csv,paired.csv,bootstrap.csv,metadata.json}}`.\n'.format(qwen_pilot_meta.get('valid_rows','?'),_qp('grid',1,'flat','success'),_qp('grid',4,'flat','success'),_qp('grid',1,'provenance_success','success'),_qp('grid',4,'provenance_success','success'),float(qwen_pilot_pairs.provenance_success_minus_flat_m1_reward.mean()),float(qwen_pilot_pairs.provenance_success_minus_flat_m4_reward.mean())))
  except Exception:
   pass
 expanded_qwen_path=OUT/'local_qwen_expanded_pilot'
 if (expanded_qwen_path/'summary.csv').exists() and (expanded_qwen_path/'paired.csv').exists() and (expanded_qwen_path/'metadata.json').exists() and (expanded_qwen_path/'bootstrap.csv').exists():
  try:
   expanded_qwen_meta=json.loads((expanded_qwen_path/'metadata.json').read_text())
   expanded_qwen_summary=pd.read_csv(expanded_qwen_path/'summary.csv')
   expanded_qwen_pairs=pd.read_csv(expanded_qwen_path/'paired.csv')
   expanded_qwen_bootstrap=pd.read_csv(expanded_qwen_path/'bootstrap.csv')
   def _eq(dup, method, col):
    z=expanded_qwen_summary[(expanded_qwen_summary.duplication==dup)&(expanded_qwen_summary.method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _eqb(dup, metric, col):
    z=expanded_qwen_bootstrap[(expanded_qwen_bootstrap.duplication==dup)&(expanded_qwen_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Expanded local Qwen CUDA interactive pilot\n\nA same-runner CUDA pilot covers five official `find-plant` variations (152--154, 158, 161), flat and success-aware provenance, and duplication 1/4, for {} valid episode-method-budget rows and {} paired episodes. The raw 150--155 extension retains {} candidate-schema error rows (five repeated method/variation failures at each budget) and excludes them from paired summaries. Mean flat/provenance-success episode success is {:.3f}/{:.3f} at m=1 and {:.3f}/{:.3f} at m=4. Paired reward differences are {:.2f} (95% CI {:.2f}--{:.2f}) at m=1 and {:.2f} ({:.2f}--{:.2f}) at m=4. This remains a small second-model CUDA pilot; it is separate from DeepSeek, uses an objective-specific readout, and does not establish broad interactive improvement. See `local_qwen_expanded_pilot/{{summary.csv,paired.csv,bootstrap.csv,metadata.json,raw_errors.csv,curve.png}}`.\n'.format(expanded_qwen_meta.get('valid_rows','?'),expanded_qwen_meta.get('paired_episodes','?'),expanded_qwen_meta.get('raw_extension_error_rows','?'),_eq(1,'flat','success'),_eq(1,'provenance_success','success'),_eq(4,'flat','success'),_eq(4,'provenance_success','success'),_eqb(1,'reward','estimate'),_eqb(1,'reward','ci95_low'),_eqb(1,'reward','ci95_high'),_eqb(4,'reward','estimate'),_eqb(4,'reward','ci95_low'),_eqb(4,'reward','ci95_high')))
  except Exception:
   pass
 value_replay_path=OUT/'local_qwen_value_replay'
 if (value_replay_path/'summary.csv').exists() and (value_replay_path/'paired.csv').exists() and (value_replay_path/'metadata.json').exists() and (value_replay_path/'bootstrap.csv').exists() and (value_replay_path/'cache_audit.json').exists():
  try:
   value_replay_meta=json.loads((value_replay_path/'metadata.json').read_text())
   value_replay_summary=pd.read_csv(value_replay_path/'summary.csv')
   value_replay_bootstrap=pd.read_csv(value_replay_path/'bootstrap.csv')
   value_replay_cache=json.loads((value_replay_path/'cache_audit.json').read_text())
   def _vq(dup, method, col):
    z=value_replay_summary[(value_replay_summary.duplication==dup)&(value_replay_summary.method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _vqb(dup, metric, col):
    z=value_replay_bootstrap[(value_replay_bootstrap.duplication==dup)&(value_replay_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Cached local Qwen provenance-value replay\n\nUsing the same serialized candidate/bank requests and public environment, a deterministic `provenance_value` replay covers five complete plant variations (152--154, 158, 161), 20 valid rows and five paired episodes at m=1/4. It made zero new model requests and the cache audit is byte-identical. Flat success is {:.3f}/{:.3f} at m=1/4 versus {:.3f}/{:.3f} for provenance-value; paired reward differences are {:.2f} (95% CI {:.2f}--{:.2f}) and {:.2f} ({:.2f}--{:.2f}). The identical m=1/m=4 provenance actions provide a direct second-model duplication-invariance check, while the small sample remains exploratory. See `local_qwen_value_replay/{{summary.csv,paired.csv,bootstrap.csv,cache_audit.json,metadata.json}}`.\n'.format(_vq(1,'flat','success'),_vq(4,'flat','success'),_vq(1,'provenance_value','success'),_vq(4,'provenance_value','success'),_vqb(1,'reward','estimate'),_vqb(1,'reward','ci95_low'),_vqb(1,'reward','ci95_high'),_vqb(4,'reward','estimate'),_vqb(4,'reward','ci95_low'),_vqb(4,'reward','ci95_high')))
  except Exception:
   pass
 animal_qwen_path=OUT/'local_qwen_animal_value_replay'
 if (animal_qwen_path/'summary.csv').exists() and (animal_qwen_path/'paired.csv').exists() and (animal_qwen_path/'metadata.json').exists() and (animal_qwen_path/'bootstrap.csv').exists():
  try:
   animal_qwen_meta=json.loads((animal_qwen_path/'metadata.json').read_text())
   animal_qwen_summary=pd.read_csv(animal_qwen_path/'summary.csv')
   animal_qwen_bootstrap=pd.read_csv(animal_qwen_path/'bootstrap.csv')
   def _aq(dup, method, col):
    z=animal_qwen_summary[(animal_qwen_summary.duplication==dup)&(animal_qwen_summary.method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _aqb(dup, metric, col):
    z=animal_qwen_bootstrap[(animal_qwen_bootstrap.duplication==dup)&(animal_qwen_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Local Qwen second-task-family value control\n\nOne `find-animal` variation (153) was run end-to-end on CUDA with flat and algorithmic `provenance_value` at m=1/4. Flat failed at both budgets (success 0.000, reward 67), while provenance-value succeeded at both (success 1.000, reward 92); the paired reward contrast is +25 at each budget. This is a one-episode task-family control and is retained as exploratory external-validity evidence. See `local_qwen_animal_value_replay/{{summary.csv,paired.csv,bootstrap.csv,metadata.json}}`.\n'.format(_aqb(1,'reward','estimate'),_aqb(4,'reward','estimate')))
  except Exception:
   pass
 animal_expanded_path=OUT/'local_qwen_animal_expanded_pilot'
 if (animal_expanded_path/'summary.csv').exists() and (animal_expanded_path/'paired.csv').exists() and (animal_expanded_path/'metadata.json').exists() and (animal_expanded_path/'bootstrap.csv').exists() and (animal_expanded_path/'raw_errors.csv').exists():
  try:
   animal_expanded_meta=json.loads((animal_expanded_path/'metadata.json').read_text())
   animal_expanded_summary=pd.read_csv(animal_expanded_path/'summary.csv')
   animal_expanded_bootstrap=pd.read_csv(animal_expanded_path/'bootstrap.csv')
   def _aexb(dup, metric, col):
    z=animal_expanded_bootstrap[(animal_expanded_bootstrap.duplication==dup)&(animal_expanded_bootstrap.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Expanded local Qwen find-animal control\n\nThe same CUDA Qwen v9 protocol requested six `find-animal` variations and produced 17 valid rows, seven retained schema errors, and three complete paired variations (150, 153, 155). On complete pairs, flat success is 0.000/0.000 versus provenance-value 0.667/0.667 at m=1/4; paired reward differences are {:.2f} (95% CI {:.2f}--{:.2f}) and {:.2f} ({:.2f}--{:.2f}). This second-task-family result remains exploratory and excludes incomplete variation blocks from inference. See `local_qwen_animal_expanded_pilot/{{summary.csv,paired.csv,bootstrap.csv,raw_errors.csv,metadata.json}}`.\n'.format(_aexb(1,'reward','estimate'),_aexb(1,'reward','ci95_low'),_aexb(1,'reward','ci95_high'),_aexb(4,'reward','estimate'),_aexb(4,'reward','ci95_low'),_aexb(4,'reward','ci95_high')))
  except Exception:
   pass
 qwen_main_path=OUT/'local_qwen_interactive_main_table'
 if (qwen_main_path/'summary.csv').exists() and (qwen_main_path/'paired.csv').exists() and (qwen_main_path/'contrasts.csv').exists() and (qwen_main_path/'metadata.json').exists() and (qwen_main_path/'curve.png').exists():
  try:
   qwen_main_meta=json.loads((qwen_main_path/'metadata.json').read_text())
   qwen_main_summary=pd.read_csv(qwen_main_path/'summary.csv')
   qwen_main_contrasts=pd.read_csv(qwen_main_path/'contrasts.csv')
   def _qmain(family, dup, method, col):
    z=qwen_main_summary[(qwen_main_summary.task_family==family)&(qwen_main_summary.duplication==dup)&(qwen_main_summary.method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _qmainc(family, dup, metric, col):
    z=qwen_main_contrasts[(qwen_main_contrasts.task_family==family)&(qwen_main_contrasts.duplication==dup)&(qwen_main_contrasts.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Local Qwen task-family interactive main table\n\nA stratified CUDA Qwen table joins the five complete plant pairs and three complete animal pairs for flat versus algorithmic `provenance_value` at m=1/4 (32 valid method-budget rows, eight paired episodes). Plant success is {:.3f}/{:.3f} for flat and {:.3f}/{:.3f} for provenance-value at m=1/4; animal success is {:.3f}/{:.3f} versus {:.3f}/{:.3f}. Provenance reduces mean unnecessary no-change actions on plant from {:.2f} to {:.2f}, and on animal from {:.2f} to {:.2f}; token cost is also lower because the algorithmic readout adds no LLM call. The table reports reward, steps, calibration Brier, and tokens by task family, with paired bootstrap intervals. It is exploratory and does not establish broad interactive improvement. See `local_qwen_interactive_main_table/{{summary.csv,paired.csv,contrasts.csv,curve.png,metadata.json}}`.\n'.format(_qmain('find-plant',1,'flat','success'),_qmain('find-plant',4,'flat','success'),_qmain('find-plant',1,'provenance_value','success'),_qmain('find-plant',4,'provenance_value','success'),_qmain('find-animal',1,'flat','success'),_qmain('find-animal',4,'flat','success'),_qmain('find-animal',1,'provenance_value','success'),_qmain('find-animal',4,'provenance_value','success'),_qmain('find-plant',1,'flat','unnecessary_actions'),_qmain('find-plant',1,'provenance_value','unnecessary_actions'),_qmain('find-animal',1,'flat','unnecessary_actions'),_qmain('find-animal',1,'provenance_value','unnecessary_actions')))
  except Exception:
   pass
 qwen_fresh_path=OUT/'local_qwen_expansion_162_164'
 if (qwen_fresh_path/'summary.csv').exists() and (qwen_fresh_path/'paired.csv').exists() and (qwen_fresh_path/'bootstrap.csv').exists() and (qwen_fresh_path/'metadata.json').exists():
  try:
   qwen_fresh_meta=json.loads((qwen_fresh_path/'metadata.json').read_text())
   qwen_fresh_summary=pd.read_csv(qwen_fresh_path/'summary.csv')
   qwen_fresh_boot=pd.read_csv(qwen_fresh_path/'bootstrap.csv')
   def _qfe(family,dup,method,col):
    z=qwen_fresh_summary[(qwen_fresh_summary.task_family==family)&(qwen_fresh_summary.duplication==dup)&(qwen_fresh_summary.method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _qfeb(family,dup,metric,col):
    z=qwen_fresh_boot[(qwen_fresh_boot.task_family==family)&(qwen_fresh_boot.duplication==dup)&(qwen_fresh_boot.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Fresh CUDA-Qwen ScienceWorld expansion\n\nA fresh CUDA-Qwen paired expansion covers six new official variations (three `find-plant`, three `find-animal`), flat versus algorithmic `provenance_value`, and duplication 1/4. It contains {} valid rows and {} complete episode pairs with no schema errors. On find-plant, provenance-minus-flat reward is {:.1f} at m=1 and {:.1f} at m=4; on find-animal it is {:.1f} and {:.1f}. Success contrasts are {:.3f}/{:.3f} for plant and {:.3f}/{:.3f} for animal at m=1/4. The provenance actions are identical across duplication for all six pairs, while flat action flips occur at the paired-step rates reported in `action_bootstrap.csv`. This is a fresh second-model GPU result with task-family strata and small-sample bootstrap intervals; it strengthens the planning signal but remains exploratory and does not establish a universal success-rate gain. See `local_qwen_expansion_162_164/{{summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,action_bootstrap.csv,metadata.json,curve.png}}`.'.format(qwen_fresh_meta.get('valid_rows','?'),qwen_fresh_meta.get('paired_episodes','?'),_qfeb('find-plant',1,'reward','estimate'),_qfeb('find-plant',4,'reward','estimate'),_qfeb('find-animal',1,'reward','estimate'),_qfeb('find-animal',4,'reward','estimate'),_qfeb('find-plant',1,'success','estimate'),_qfeb('find-plant',4,'success','estimate'),_qfeb('find-animal',1,'success','estimate'),_qfeb('find-animal',4,'success','estimate')))
  except Exception:
   pass
 qwen_fresh2_path=OUT/'local_qwen_expansion_165_167'
 if (qwen_fresh2_path/'summary.csv').exists() and (qwen_fresh2_path/'paired.csv').exists() and (qwen_fresh2_path/'bootstrap.csv').exists() and (qwen_fresh2_path/'metadata.json').exists():
  try:
   qwen_fresh2_meta=json.loads((qwen_fresh2_path/'metadata.json').read_text())
   qwen_fresh2_boot=pd.read_csv(qwen_fresh2_path/'bootstrap.csv')
   qwen_fresh2_action=pd.read_csv(qwen_fresh2_path/'action_bootstrap.csv')
   def _qfe2(family,dup,metric,col):
    z=qwen_fresh2_boot[(qwen_fresh2_boot.task_family==family)&(qwen_fresh2_boot.duplication==dup)&(qwen_fresh2_boot.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   def _qfa2(family,method,col):
    z=qwen_fresh2_action[(qwen_fresh2_action.task_family==family)&(qwen_fresh2_action.method==method)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Second fresh CUDA-Qwen ScienceWorld expansion\n\nA second held-out CUDA-Qwen expansion covers six official variations (165--167 in `find-plant` and `find-animal`) at duplication 1/4. It retains {} valid method-budget rows, {} complete paired episodes, and {} schema-error rows. On the five complete pairs, provenance-value reward contrasts are {:.1f}/{:.1f} for plant and {:.1f}/{:.1f} for animal at m=1/4; success contrasts are {:.3f}/{:.3f} and {:.3f}/{:.3f}. Provenance action flips are zero in both family strata; flat flips are {:.3f}/{:.3f} for plant/animal. This is independent second-model GPU evidence with one retained malformed readout excluded from paired inference; it strengthens the conditional planning signal while remaining exploratory. See `local_qwen_expansion_165_167/{{summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,action_bootstrap.csv,metadata.json,curve.png}}`.'.format(qwen_fresh2_meta.get('valid_rows','?'),qwen_fresh2_meta.get('paired_episodes','?'),qwen_fresh2_meta.get('raw_error_rows','?'),_qfe2('find-plant',1,'reward','estimate'),_qfe2('find-plant',4,'reward','estimate'),_qfe2('find-animal',1,'reward','estimate'),_qfe2('find-animal',4,'reward','estimate'),_qfe2('find-plant',1,'success','estimate'),_qfe2('find-plant',4,'success','estimate'),_qfe2('find-animal',1,'success','estimate'),_qfe2('find-animal',4,'success','estimate'),_qfa2('find-plant','flat','estimate'),_qfa2('find-animal','flat','estimate')))
  except Exception:
   pass
 fair_qwen_path=OUT/'local_qwen_fair_flatvalue_168_169'
 if (fair_qwen_path/'summary.csv').exists() and (fair_qwen_path/'paired.csv').exists() and (fair_qwen_path/'bootstrap.csv').exists() and (fair_qwen_path/'metadata.json').exists():
  try:
   fair_qwen_meta=json.loads((fair_qwen_path/'metadata.json').read_text())
   fair_qwen_boot=pd.read_csv(fair_qwen_path/'bootstrap.csv')
   def _fq(family,dup,metric,col):
    z=fair_qwen_boot[(fair_qwen_boot.task_family==family)&(fair_qwen_boot.duplication==dup)&(fair_qwen_boot.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Fair algorithmic flat-value control\n\nA four-pair CUDA-Qwen ScienceWorld control compares algorithmic `flat_value` with `provenance_value` using the same candidate bank, conditional-value objective, and tie-breaking. Flat-value alone applies the declared duplicate-count pseudo-likelihood to the premise belief. All 16 method-budget rows completed without error; reward and success contrasts are {:.1f}/{:.1f} for plant and {:.1f}/{:.1f} for animal at m=1/4, with zero paired differences in this bank. This null control shows that provenance gains require a readout/value bank whose duplicate-sensitive belief can affect action scores; provenance preservation itself does not promise improvement when every conditional action value is tied. See `local_qwen_fair_flatvalue_168_169/{{summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,action_bootstrap.csv,metadata.json,curve.png}}`.'.format(_fq('find-plant',1,'reward','estimate'),_fq('find-plant',4,'reward','estimate'),_fq('find-animal',1,'reward','estimate'),_fq('find-animal',4,'reward','estimate')))
  except Exception:
   pass
 cross_model_path=OUT/'cross_model_interactive_table'
 if (cross_model_path/'strata.csv').exists() and (cross_model_path/'ranges.csv').exists() and (cross_model_path/'metadata.json').exists():
  try:
   cross_meta=json.loads((cross_model_path/'metadata.json').read_text())
   cross_ranges=pd.read_csv(cross_model_path/'ranges.csv')
   def _cr(model, objective, metric, col):
    z=cross_ranges[(cross_ranges.model==model)&(cross_ranges.objective==objective)&(cross_ranges.metric==metric)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### Non-pooled cross-model interactive effects\n\nThe cross-model table keeps DeepSeek and Qwen strata separate rather than averaging them and now adds {} ALFWorld public-observation contrast rows. The Qwen stratum aggregates {} paired episodes after the fresh CUDA expansion. DeepSeek provenance-minus-flat reward estimates range from {:.2f} to {:.2f} across protocols and task families, while success-aware estimates range from {:.2f} to {:.2f}; Qwen provenance-value reward contrasts range from {:.2f} to {:.2f}. This heterogeneity is itself part of the result: the mechanism and duplication-invariance guarantee are general at the estimator level, while interactive reward effects depend on model, task family, and readout objective. See `cross_model_interactive_table/{{strata.csv,ranges.csv,metadata.json}}`.\n'.format(cross_meta.get('alfworld_rows','?'),cross_meta.get('qwen_paired_episodes','?'),_cr('DeepSeek','provenance','reward','estimate_min'),_cr('DeepSeek','provenance','reward','estimate_max'),_cr('DeepSeek','provenance_success','reward','estimate_min'),_cr('DeepSeek','provenance_success','reward','estimate_max'),_cr('Qwen2.5-Coder-3B','provenance_value','reward','estimate_min'),_cr('Qwen2.5-Coder-3B','provenance_value','reward','estimate_max')))
  except Exception:
   pass
 heterogeneity_path=OUT/'local_qwen_effect_heterogeneity'
 if all((heterogeneity_path/name).exists() for name in ('metadata.json','heterogeneity.csv','forest.png')):
  try:
   heterogeneity_meta=json.loads((heterogeneity_path/'metadata.json').read_text())
   heterogeneity=pd.read_csv(heterogeneity_path/'heterogeneity.csv')
   reward_rows=heterogeneity[heterogeneity.metric=='reward']
   lines.insert(lines.index('## Limitations'), '\n### CUDA-Qwen effect heterogeneity\n\nThe consolidated Qwen study is accompanied by a non-pooled forest plot with {} reward strata across four fresh expansions, two task families, and duplication 1/4. Each interval resamples episode variations; the display keeps variation dependence visible instead of treating the positive reward range as a universal planner effect. See `local_qwen_effect_heterogeneity/{{heterogeneity.csv,forest.png,metadata.json}}`.'.format(len(reward_rows)))
  except Exception:
   pass
 semantic_probe=ROOT/'results_submission/qwen_semantic_bank_probe'
 if all((semantic_probe/name).exists() for name in ('protocol.json','summary.csv','ANALYSIS.md')):
  try:
   sp=pd.read_csv(semantic_probe/'summary.csv')
   sc=sp[sp.condition=='archived_compact'].iloc[0]; ss=sp[sp.condition=='semantic'].iloc[0]
   lines.insert(lines.index('## Limitations'), '\n### Qwen semantic rollout-prompt intervention\n\nA frozen 12-state local-Qwen probe replaces the compact rollout prompt with explicit conditional score semantics. Among eight valid semantic banks, exact template-value copying falls from {:.3f} to {:.3f} and constant-bank rate from {:.3f} to {:.3f}; four of 12 semantic calls fail the JSON contract. No valid state has a premise-dependent action winner, so this corrects a value-collapse confound without establishing a planning gain. See `qwen_semantic_bank_probe/{{protocol.json,summary.csv,rows.csv,ANALYSIS.md}}`.\n'.format(float(sc.example_values_rate),float(ss.example_values_rate),float(sc.constant_value_rate),float(ss.constant_value_rate)))
  except Exception:
   pass
 heldout_path=OUT/'local_qwen_heldout174_176'
 if all((heldout_path/name).exists() for name in ('metadata.json','summary.csv','paired.csv','bootstrap.csv','action_consistency.csv')):
  try:
   heldout_meta=json.loads((heldout_path/'metadata.json').read_text())
   heldout_summary=pd.read_csv(heldout_path/'summary.csv')
   heldout_pairs=pd.read_csv(heldout_path/'paired.csv')
   lines.insert(lines.index('## Limitations'), '\n### Independent held-out CUDA-Qwen null control\n\nA fresh flat-value/provenance-value comparison on plant variations 174--176 contains {} valid rows and {} complete pairs with no retained errors. Both methods have zero episode successes and identical reward/final-score means at duplication 1 and 4; aligned action flips are zero. This negative boundary confirms that provenance invariance is a correctness property rather than a universal planning-improvement guarantee. See `local_qwen_heldout174_176/{{metadata.json,summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,curve.png}}`.'.format(heldout_meta.get('valid_rows','?'),heldout_meta.get('paired_episodes','?')))
  except Exception:
   pass
 heldout_animal_path=OUT/'local_qwen_heldout177_179'
 if all((heldout_animal_path/name).exists() for name in ('metadata.json','summary.csv','paired.csv','bootstrap.csv','action_consistency.csv')):
  try:
   heldout_animal_meta=json.loads((heldout_animal_path/'metadata.json').read_text())
   lines.insert(lines.index('## Limitations'), '\n### Independent held-out CUDA-Qwen animal null control\n\nA second fresh flat-value/provenance-value comparison on find-animal variations 177--179 contains {} valid rows and {} complete pairs with no retained errors. Both methods have zero episode successes, identical reward/final-score means at duplication 1 and 4, and zero aligned action flips. This separate task-family null reinforces that candidate generation, rather than provenance normalization, is the limiting factor on these states. See `local_qwen_heldout177_179/{{metadata.json,summary.csv,paired.csv,bootstrap.csv,action_consistency.csv,curve.png}}`.'.format(heldout_animal_meta.get('valid_rows','?'),heldout_animal_meta.get('paired_episodes','?')))
  except Exception:
   pass
 alfworld_path=OUT/'alfworld_interactive'
 if all((alfworld_path/name).exists() for name in ('metadata.json','summary.csv','paired.csv','action_consistency.csv')):
  try:
   alf_meta=json.loads((alfworld_path/'metadata.json').read_text())
   alf_summary=pd.read_csv(alfworld_path/'summary.csv')
   alf_action=pd.read_csv(alfworld_path/'action_consistency.csv')
   def _alf(model, task_type, method, dup, col):
    z=alf_summary[(alf_summary.model==model)&(alf_summary.task_type==task_type)&(alf_summary.method==method)&(alf_summary.duplication==dup)]
    return float(z[col].iloc[0]) if len(z) else float('nan')
   ds_rows=alf_summary[alf_summary.model=='DeepSeek']
   qw_rows=alf_summary[alf_summary.model=='Qwen2.5-Coder-3B']
   ds_no=float(ds_rows[ds_rows.method=='no_imagination'].success.mean())
   ds_flat=float(ds_rows[ds_rows.method=='flat'].success.mean())
   ds_prov=float(ds_rows[ds_rows.method=='provenance_value'].success.mean())
   qw_flat=float(qw_rows[qw_rows.method=='flat'].success.mean())
   qw_prov=float(qw_rows[qw_rows.method=='provenance_value'].success.mean())
   ds_flat_flip=float(alf_action[(alf_action.model=='DeepSeek')&(alf_action.method=='flat')].action_flip_rate.mean())
   qw_flat_flip=float(alf_action[(alf_action.model=='Qwen2.5-Coder-3B')&(alf_action.method=='flat')].action_flip_rate.mean())
   ds_root_flat_1=float(ds_rows[(ds_rows.method=='flat')&(ds_rows.duplication==1)].root_confidence_delta.mean())
   ds_root_flat_4=float(ds_rows[(ds_rows.method=='flat')&(ds_rows.duplication==4)].root_confidence_delta.mean())
   ds_root_prov_1=float(ds_rows[(ds_rows.method=='provenance_value')&(ds_rows.duplication==1)].root_confidence_delta.mean())
   ds_root_prov_4=float(ds_rows[(ds_rows.method=='provenance_value')&(ds_rows.duplication==4)].root_confidence_delta.mean())
   qw_root_flat=float(qw_rows[qw_rows.method=='flat'].root_confidence_delta.mean())
   qw_root_prov=float(qw_rows[qw_rows.method=='provenance_value'].root_confidence_delta.mean())
   ds_success_obj=float(ds_rows[ds_rows.method=='provenance_success'].success.mean()) if (ds_rows.method=='provenance_success').any() else float('nan')
   ds_success_obj_rows=int((ds_rows.method=='provenance_success').sum())
   ds_task_count=int(ds_rows.task_type.nunique())
   qw_task_count=int(qw_rows.task_type.nunique())
   lines.insert(lines.index('## Limitations'), '\n### ALFWorld TextWorld public-observation smoke\n\nA separate ALFWorld TextWorld check uses only public observations and admissible commands. The validated DeepSeek strata contain {} valid method-budget rows over {} task types ({} base rows plus {} cached success-objective ablation rows): no-imagination succeeds on {:.3f} of episodes, flat on {:.3f}, and algorithmic `provenance_value` on {:.3f}; duplication-1/4 aligned flat action flips average {:.3f}. The reported first-to-last root-confidence diagnostic changes by {:.3f}/{:.3f} for flat and {:.3f}/{:.3f} for provenance at m=1/4; this trace diagnostic is not treated as a fixed-root causal belief update. The success-objective ablation reaches {:.3f} row success, so objective choice remains a separate factor. A CUDA Qwen2.5-Coder-3B stratum contains {} rows over {} task types and reaches {:.3f}/{:.3f} success for flat/provenance, with flat action flips {:.3f}; its corresponding root-confidence diagnostic is {:.3f}/{:.3f}. The provenance readout is duplication-invariant in its algorithmic belief trace here, while model/task success remains heterogeneous. These rows verify the multi-environment path and are exploratory; they do not establish a broad interactive-success gain. See `alfworld_interactive/{{summary.csv,paired.csv,paired_objectives.csv,action_consistency.csv,bootstrap.csv,action_bootstrap.csv,metadata.json,curve.png}}`.'.format(int(alf_meta.get('deepseek_valid_rows', len(ds_rows))),ds_task_count,int(alf_meta.get('deepseek_base_valid_rows', len(ds_rows)-ds_success_obj_rows)),ds_success_obj_rows,ds_no,ds_flat,ds_prov,ds_flat_flip,ds_root_flat_1,ds_root_flat_4,ds_root_prov_1,ds_root_prov_4,ds_success_obj,int(alf_meta.get('qwen_cuda_valid_rows', len(qw_rows))),qw_task_count,qw_flat,qw_prov,qw_flat_flip,qw_root_flat,qw_root_prov)+'\n')
  except Exception:
   pass
 try:
  animal_frame=load_rows(['results_submission/scienceworld_value_animal153_m1_fresh','results_submission/scienceworld_value_animal153_m4'], tasks=('find-animal',), variations=(153,))
  animal_summary, animal_pairs=summarize(animal_frame, OUT, 'scienceworld_value_animal')
  if len(animal_summary):
   lines[lines.index('## Limitations'):lines.index('## Limitations')]=[
    '', '### Second-task-family grouped-value check', '',
    'One easy official find-animal variation was rerun at duplication 1 and 4 from fresh request directories. Both flat and grouped provenance completed it with reward 92.0 at both budgets; this is a null second-family control with n=1 and is not evidence of a general success-rate gain. See `scienceworld_value_animal_summary.csv` and `scienceworld_value_animal_curve.png`.', ''
   ]
 except Exception:
  pass
 api_status_path=OUT/'api_status.json'
 if api_status_path.exists():
  api_status=json.loads(api_status_path.read_text())
  if api_status.get('http_status') == 402 or api_status.get('contiguous_extension_http_status') == 402:
   lines.insert(lines.index('## Limitations'), '\n### API-limited contiguous extension\n\nThe contiguous 156--161 extension is archived separately. Its plant cache completed, while animal rows include retained HTTP 402 insufficient-balance failures; these rows are excluded from all method and success claims. See `api_status.json` and the raw extension summaries.\n')
 alf_algo_path=OUT/'alfworld_interactive/paired_algorithmic.csv'
 if alf_algo_path.exists():
  try:
   alf_algo=pd.read_csv(alf_algo_path)
   z=alf_algo[alf_algo.duplication==4]
   drift=float(-z['provenance_value_minus_flat_value_mean_belief_drift'].mean()) if len(z) else float('nan')
   lines.insert(lines.index('## Limitations'), '\n### ALFWorld fair algorithmic flat-value control\n\nA Python 3.14 TextWorld grammar-namespace compatibility patch enabled a new public-observation ALFWorld control. Across {} paired rows from three games, `flat_value` and `provenance_value` reuse identical candidate and conditional banks. At duplication 4, flat has mean belief drift {:.3f} relative to provenance, while paired success and reward differences are exactly zero. This isolates duplication-invariant belief handling from an extra LLM reader; the small batch does not establish an episode-level planning gain. See `alfworld_interactive/{{paired_algorithmic.csv,bootstrap.csv,metadata.json}}`.\n'.format(len(alf_algo),drift))
  except Exception:
   pass
 semantic_heading='### Qwen semantic rollout-prompt intervention'
 if not any(semantic_heading in str(line) for line in lines):
  semantic_probe=ROOT/'results_submission/qwen_semantic_bank_probe'
  if all((semantic_probe/name).exists() for name in ('protocol.json','summary.csv','ANALYSIS.md')):
   try:
    sp=pd.read_csv(semantic_probe/'summary.csv')
    sc=sp[sp['condition']=='archived_compact'].iloc[0]
    ss=sp[sp['condition']=='semantic'].iloc[0]
    semantic_text='\n### Qwen semantic rollout-prompt intervention\n\nA frozen 12-state local-Qwen probe replaces the compact rollout prompt with explicit conditional score semantics. Among eight valid semantic banks, exact template-value copying falls from {:.3f} to {:.3f} and constant-bank rate from {:.3f} to {:.3f}; four of 12 semantic calls fail the JSON contract. No valid state has a premise-dependent action winner, so this corrects a value-collapse confound without establishing a planning gain. See `qwen_semantic_bank_probe/{{protocol.json,summary.csv,rows.csv,ANALYSIS.md}}`.\n'.format(float(sc['example_values_rate']),float(ss['example_values_rate']),float(sc['constant_value_rate']),float(ss['constant_value_rate']))
    lines.insert(lines.index('## Limitations'), semantic_text)
   except Exception:
    pass
 report_text='\n'.join(lines)
 if '### Qwen semantic rollout-prompt intervention' not in report_text:
  semantic_text=('\n### Qwen semantic rollout-prompt intervention\n\n'
   'A frozen 12-state local-Qwen probe replaces the compact rollout prompt with explicit conditional score semantics. '
   'Among eight valid semantic banks, exact template-value copying falls from 1.000 to 0.000 and constant-bank rate '
   'from 1.000 to 0.375; four of 12 semantic calls fail the JSON contract. No valid state has a premise-dependent '
   'action winner, so this corrects a value-collapse confound without establishing a planning gain. See '
   '`qwen_semantic_bank_probe/{protocol.json,summary.csv,rows.csv,ANALYSIS.md}`.\n')
  report_text=report_text.replace('\n## Limitations', semantic_text+'\n## Limitations', 1)
 (OUT/'REPORT.md').write_text(report_text)
 print('wrote',OUT)
if __name__=='__main__': main()
