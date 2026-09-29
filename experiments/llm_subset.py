"""Real DeepSeek subset with archived prompts/responses and no synthetic fallback.

The numerical grid is the primary mechanism audit. This file is intentionally a
smaller API experiment: the model reads the same public evidence and controlled
rollout records, and its returned scores are used for the derived aggregators.
Failed or unparsable API calls are retained as failures and never replaced by
hand-written scores.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, math, os, random, time, threading
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy import stats
from experiments.core import scenario, intervention, METHODS, MULTIPLICITIES, public_task

SEEDS = tuple(range(12))
CONDITIONS = ('pure_duplication','paraphrase_duplication','independent_rollout')
MODEL = 'deepseek-chat'

def request_json(api_key, model, messages, max_tokens=1200):
    body={'model':model,'messages':messages,'temperature':0.2,'max_tokens':max_tokens,
          'response_format':{'type':'json_object'}}
    try:
        r=requests.post('https://api.deepseek.com/chat/completions',
                        headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'},
                        json=body,timeout=90)
        r.raise_for_status(); payload=r.json(); content=payload['choices'][0]['message']['content']
        obj=json.loads(content) if isinstance(content,str) else content
        usage=payload.get('usage',{})
        return {'ok':True,'object':obj,'raw_content':content,'usage':usage,
                'model':payload.get('model',model),'error':None}
    except Exception as exc:
        return {'ok':False,'object':None,'raw_content':None,'usage':{},'model':model,
                'error':f'{type(exc).__name__}: {exc}'}

def item_messages(task, rows):
    items=[{'sample_id':r.sid,'hypothesis':r.h,'action':r.action,
            'semantic_key':list(r.semantic_key),'rollout':r.text} for r in rows]
    system=('You are a calibrated evaluator in a controlled planning study. '
            'Return JSON only. A rollout is an imagined conditional consequence, '
            'not a new real observation. Do not infer hidden physical state. '
            'Preserve every sample_id exactly.')
    user=('Given the task and rollout list below, score each rollout independently. '
          'For each item return root_probability (P(H=1) from the real evidence '
          'and the item, between 0 and 1) and predicted_return (a numeric estimate '
          'for that listed action). Do not count other list items as evidence for H. '
          'JSON schema: {"items":[{"sample_id":"...","root_probability":0.5,"predicted_return":0.0}]}\n\n'
          f'TASK:\n{task}\nROLLOUTS:\n{json.dumps(items,ensure_ascii=False)}')
    return [{'role':'system','content':system},{'role':'user','content':user}]

def evidence_messages(task):
    return [
      {'role':'system','content':'Return valid JSON only. Do not use imagined rollouts as observations.'},
      {'role':'user','content':f'Estimate P(H=1) using only the real evidence in this task. JSON schema: {{"p_h1":0.5}}\n\n{task}'}]

def flat_messages(task, rows):
    items=[{'sample_id':r.sid,'hypothesis':r.h,'action':r.action,'rollout':r.text} for r in rows]
    return [
      {'role':'system','content':'Return valid JSON only. This is a flat baseline; read the entire presented list, but do not claim an imagined consequence is a real observation.'},
      {'role':'user','content':('Estimate root probability and choose an action from the complete presented rollout list. '
        'Return JSON: {"p_h1":0.5,"action":0,"action_scores":[0.0,0.0]}. '
        f'\nTASK:\n{task}\nPRESENTED ROLLOUTS:\n{json.dumps(items,ensure_ascii=False)}')}]

def parse_evidence(resp):
    if not resp['ok']: raise ValueError(resp['error'])
    p=float(resp['object']['p_h1'])
    if not 0<=p<=1: raise ValueError('p_h1 out of range')
    return p

def parse_items(resp, expected):
    if not resp['ok']: raise ValueError(resp['error'])
    items=resp['object']['items']; got={str(x['sample_id']):x for x in items}
    if set(got)!=set(expected): raise ValueError(f'missing/extra sample ids: {len(got)} vs {len(expected)}')
    out={}
    for sid in expected:
        p=float(got[sid]['root_probability']); v=float(got[sid]['predicted_return'])
        if not 0<=p<=1 or not math.isfinite(v): raise ValueError('invalid item score')
        out[sid]={'root_probability':p,'predicted_return':v}
    return out

def parse_flat(resp):
    if not resp['ok']: raise ValueError(resp['error'])
    p=float(resp['object']['p_h1']); action=int(resp['object']['action'])
    scores=[float(x) for x in resp['object'].get('action_scores',[0.,0.])]
    if not 0<=p<=1 or action not in (0,1) or len(scores)!=2: raise ValueError('invalid flat output')
    return p,action,scores

def derived(method, q, rows, scores):
    # Preserve actual sample identities. Exact/semantic variants only alter the
    # evidence presentation; all derived methods consume the same per-item scores.
    if method=='exact_dedup': key=lambda r:r.text
    elif method=='semantic_dedup': key=lambda r:r.semantic_key
    else: key=lambda r:r.sid
    selected={}
    for r in rows: selected.setdefault(key(r),r)
    selected=list(selected.values())
    if method=='trajectory_average':
        p=float(np.mean([scores[r.sid]['root_probability'] for r in rows]))
    elif method in ('source_average','belief_mixing','provenance'):
        p=q
    else:
        p=float(np.mean([scores[r.sid]['root_probability'] for r in selected]))
    vals=[]
    for a in (0,1):
        vals.append(float(np.mean([scores[r.sid]['predicted_return'] for r in selected if r.action==a])))
    if method=='source_average':
        # One mean per (h, action) family, then explicit q mixture.
        vals=[]
        for a in (0,1):
            fam=[]
            for h in (0,1):
                x=[scores[r.sid]['predicted_return'] for r in selected if r.action==a and r.h==h]
                fam.append(float(np.mean(x)))
            vals.append((1-q)*fam[0]+q*fam[1])
    if method in ('belief_mixing','provenance'):
        vals=[]
        for a in (0,1):
            fam=[]
            for h in (0,1):
                x=[scores[r.sid]['predicted_return'] for r in selected if r.action==a and r.h==h]
                fam.append(float(np.mean(x)))
            vals.append((1-q)*fam[0]+q*fam[1])
    return p,int(np.argmax(vals)),vals,len(selected)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results_paper/llm_subset');ap.add_argument('--api-key',default=None);ap.add_argument('--model',default=MODEL)
    a=ap.parse_args(); key=a.api_key or os.getenv('DEEPSEEK_API_KEY')
    if not key: raise SystemExit('DEEPSEEK_API_KEY is required; no fallback is used')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rawlog=(out/'api_archive.jsonl').open('w'); rawlog_lock=threading.Lock(); rows=[]; scenario_meta=[]
    # Balanced small subset: both directions/states and varied prior/strength.
    scenarios=[]
    for seed in SEEDS:
        state=seed%2; direction=(seed//2)%2; prior=(.3,.5,.7)[seed%3]; strength=(1.25,2.,4.)[seed%3]; family=seed%2
        scenarios.append(scenario(seed,state,direction,prior,strength,family))

    def call(kind, scenario_id, condition, m, messages, max_tokens):
        started=time.time(); resp=request_json(key,a.model,messages,max_tokens); resp['elapsed_s']=time.time()-started
        with rawlog_lock:
            rawlog.write(json.dumps({'kind':kind,'scenario_id':scenario_id,'condition':condition,'multiplicity':m,'messages':messages,'response':resp},ensure_ascii=False)+'\n');rawlog.flush()
        return resp

    # Evidence calls are independent of imagined rollouts and cached per scenario.
    evidence={}
    for s in scenarios:
        task=public_task(s,'pure_duplication',1)
        resp=call('evidence',s['scenario_id'],'base',1,evidence_messages(task),180)
        try: evidence[s['scenario_id']]=parse_evidence(resp)
        except Exception as exc: evidence[s['scenario_id']]=None; print('evidence failure',s['scenario_id'],exc)

    jobs=[]
    for s in scenarios:
      q=evidence[s['scenario_id']]
      if q is None: continue
      for condition in CONDITIONS:
       for m in MULTIPLICITIES[:4]:
        task=public_task(s,condition,m); _,rollouts=intervention(s,condition,m)
        order_seed=int(hashlib.sha256(f'{s["scenario_id"]}|{condition}|{m}'.encode()).hexdigest()[:8],16)
        order_rng=random.Random(order_seed); order_rng.shuffle(rollouts)
        jobs.append((s,condition,m,q,task,rollouts))

    def run_job(job):
        s,condition,m,q,task,rollouts=job; sid=s['scenario_id']
        item=call('item_scores',sid,condition,m,item_messages(task,rollouts),2600)
        flat=call('flat',sid,condition,m,flat_messages(task,rollouts),700)
        try: scores=parse_items(item,[r.sid for r in rollouts]); item_error=None
        except Exception as exc: scores=None; item_error=str(exc)
        try: flat_out=parse_flat(flat); flat_error=None
        except Exception as exc: flat_out=None; flat_error=str(exc)
        outrows=[]
        if scores is not None:
            for method in METHODS:
                if method=='flat':
                    if flat_out is None: continue
                    p,action,vals=flat_out; nsel=len(rollouts)
                else:
                    p,action,vals,nsel=derived(method,q,rollouts,scores)
                true_h=int(s['root_correct']); true_action=int(np.argmax(s['mu'][true_h]))
                outrows.append({'scenario_id':sid,'seed':s['seed'],'physical_state':s['physical_state'],'hypothesis_direction':s['hypothesis_direction'],'root_correct':bool(s['root_correct']),'prior':s['prior'],'strength':s['strength'],'family':s['family'],'condition':condition,'multiplicity':m,'method':method,'q_evidence':q,'root_confidence':p,'action':action,'true_action':true_action,'expected_reward':float(s['mu'][true_h,action]),'oracle_expected_reward':float(np.max(s['mu'][true_h])),'decision_regret':float(np.max(s['mu'][true_h])-s['mu'][true_h,action]),'value_spread':float(np.ptp(vals)),'selected_items':nsel,'rollout_count':len(rollouts),'equal_text_length':len({len(r.text) for r in rollouts})==1,'item_parse_error':item_error,'flat_parse_error':flat_error,'api_item_ok':item['ok'],'api_flat_ok':flat['ok']})
        return outrows

    # API calls are independent and are limited to eight concurrent requests.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for i,outrows in enumerate(ex.map(run_job,jobs),1):
            rows.extend(outrows)
            if i%8==0: print('jobs',i,'rows',len(rows),flush=True)
    rawlog.close()
    result=pd.DataFrame(rows)
    result.to_csv(out/'llm_raw.csv',index=False)
    meta={'model':a.model,'scenarios':len(scenarios),'jobs':len(jobs),'rows':len(result),'api_archive':'api_archive.jsonl','no_fallback':True,'conditions':CONDITIONS,'multiplicities':list(MULTIPLICITIES[:4]),'methods':METHODS,'api_success':int(result.api_item_ok.all() and result.api_flat_ok.all()) if len(result) else 0}
    (out/'llm_metadata.json').write_text(json.dumps(meta,indent=2))
    if len(result):
        result['baseline_action']=result.groupby(['scenario_id','condition','method']).action.transform('first')
        result['action_flip']=result.action!=result.baseline_action
        group_cols=['root_correct','condition','multiplicity','method']; summary_rows=[]
        for keys,g in result.groupby(group_cols,sort=True):
            if not isinstance(keys,tuple): keys=(keys,)
            row=dict(zip(group_cols,keys)); row['n']=len(g)
            for metric in ['root_confidence','action_flip','expected_reward','decision_regret']:
                x=g[metric].to_numpy(dtype=float); mean=float(x.mean()); sd=float(x.std(ddof=1)) if len(x)>1 else 0.0
                margin=float(stats.t.ppf(.975,len(x)-1)*sd/math.sqrt(len(x))) if len(x)>1 else float('nan')
                row[metric+'_mean']=mean;row[metric+'_std']=sd;row[metric+'_ci95_low']=mean-margin;row[metric+'_ci95_high']=mean+margin
            summary_rows.append(row)
        summary=pd.DataFrame(summary_rows)
        summary.to_csv(out/'llm_summary.csv',index=False)
    print(json.dumps(meta))

if __name__=='__main__': main()
