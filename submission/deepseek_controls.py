"""Focused DeepSeek presentation-control audit.

This is separate from the 12-scenario subset: four paired scenarios, pure
copy multiplicities 1/4, two context orders, two text paraphrase modes, and
an explicit-counting vs information-aware reader prompt. Responses are
archived verbatim; no fallback or synthetic scores are used.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, time
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd, requests
from experiments.core import scenario, intervention, public_task


def request(key, messages, model='deepseek-chat'):
    body={'model':model,'messages':messages,'temperature':0.0,'max_tokens':500,
          'response_format':{'type':'json_object'}}
    try:
        r=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},json=body,timeout=90)
        r.raise_for_status(); payload=r.json(); c=payload['choices'][0]['message']['content']; obj=json.loads(c)
        return {'ok':True,'parsed':obj,'raw':c,'usage':payload.get('usage',{}),'model':payload.get('model',model),'error':None}
    except Exception as e:
        return {'ok':False,'parsed':None,'raw':None,'usage':{},'model':model,'error':f'{type(e).__name__}: {e}'}

def rows_for(s,m,layout,paraphrase):
    base=[r for r in s['bank'] if int(r.sid[-2:])<2]
    rows=[]
    for r in base:
        for k in range(m): rows.append(replace(r,variant=(k%4 if paraphrase else 0)))
    if layout=='reverse': rows=list(reversed(rows))
    elif layout=='interleave':
        chunks=[rows[i::m] for i in range(m)]; rows=[x for z in zip(*chunks) for x in z]
    return rows

def messages(task, rows, prompt_variant):
    items=[{'sample_id':r.sid,'hypothesis':r.h,'action':r.action,'rollout':r.text} for r in rows]
    if prompt_variant=='cautious':
        extra='Copies or paraphrases of one sample are not new observations. Estimate the premise from real evidence and avoid treating repeated descendants as independent evidence.'
    else:
        extra='Read the presented rollout list and estimate the premise probability, while remembering that rollouts are imagined consequences.'
    return [{'role':'system','content':'Return one JSON object only. You are evaluating a two-action planning problem.'}, {'role':'user','content':f'{extra}\nReturn {{"p_h1":0.5,"action":0,"action_scores":[0.0,0.0]}}.\nTASK:\n{task}\nROLLOUTS:\n{json.dumps(items,ensure_ascii=False)}'}]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',default='results_submission/deepseek_controls'); p.add_argument('--api-key',default=None); p.add_argument('--model',default='deepseek-chat'); a=p.parse_args(); key=a.api_key or os.getenv('DEEPSEEK_API_KEY')
 if not key: raise SystemExit('DEEPSEEK_API_KEY required')
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); archive=(out/'api_archive.jsonl').open('w'); rows=[]
 scenarios=[]
 for seed in range(4): scenarios.append(scenario(seed,seed%2,(seed+1)%2,(.3,.5,.7,.4)[seed],(1.25,2.,4.,2.)[seed],seed%2))
 for s in scenarios:
  for m in (1,4):
   for layout in ('front','reverse','interleave'):
    for paraphrase in (False,True):
     for prompt_variant in ('neutral','cautious'):
      rs=rows_for(s,m,layout,paraphrase); task=public_task(s,'pure_duplication',m)
      msg=messages(task,rs,prompt_variant); started=time.time(); resp=request(key,msg,a.model); resp['elapsed_s']=time.time()-started
      archive.write(json.dumps({'scenario_id':s['scenario_id'],'multiplicity':m,'layout':layout,'paraphrase':paraphrase,'prompt_variant':prompt_variant,'messages':msg,'response':resp},ensure_ascii=False)+'\n'); archive.flush()
      obj=resp['parsed'] or {}; p_h=obj.get('p_h1'); action=obj.get('action'); scores=obj.get('action_scores',[None,None])
      try: p_h=float(p_h); action=int(action); scores=[float(x) for x in scores]
      except: p_h=None; action=None; scores=[None,None]
      true_h=int(s['root_correct']); true_action=int(np.argmax(s['mu'][true_h])); reward=float(s['mu'][true_h,action]) if action in (0,1) else None
      rows.append({'scenario_id':s['scenario_id'],'seed':s['seed'],'root_correct':true_h,'multiplicity':m,'layout':layout,'paraphrase':paraphrase,'prompt_variant':prompt_variant,'p_h1':p_h,'action':action,'true_action':true_action,'action_correct':int(action==true_action) if action in (0,1) else None,'reward':reward,'brier':(p_h-true_h)**2 if p_h is not None else None,'error':resp['error'],'prompt_tokens':resp['usage'].get('prompt_tokens'),'completion_tokens':resp['usage'].get('completion_tokens')})
 archive.close(); df=pd.DataFrame(rows); df.to_csv(out/'raw.csv',index=False)
 if len(df):
  ref=df[df.multiplicity==1].set_index(['scenario_id','layout','paraphrase','prompt_variant']).p_h1.rename('p_m1'); df=df.join(ref,on=['scenario_id','layout','paraphrase','prompt_variant']); df['confidence_delta']=df.p_h1-df.p_m1; df['action_flip_from_m1']=(df.action!=df.groupby(['scenario_id','layout','paraphrase','prompt_variant']).action.transform('first')).astype(float)
  df.to_csv(out/'raw.csv',index=False); df.groupby(['multiplicity','layout','paraphrase','prompt_variant']).agg(n=('p_h1','count'),confidence=('p_h1','mean'),confidence_delta=('confidence_delta','mean'),action_flip=('action_flip_from_m1','mean'),reward=('reward','mean'),brier=('brier','mean'),action_correct=('action_correct','mean'),prompt_tokens=('prompt_tokens','mean')).reset_index().to_csv(out/'summary.csv',index=False)
 (out/'metadata.json').write_text(json.dumps({'model':a.model,'scenarios':4,'multiplicities':[1,4],'layouts':['front','reverse','interleave'],'paraphrase':[False,True],'prompt_variants':['neutral','cautious'],'no_fallback':True},indent=2)); print('wrote',len(rows),'API rows')
if __name__=='__main__': main()
