"""Small local-Qwen diagnostic for duplicate-rollout sensitivity.

This is deliberately a diagnostic, not a benchmark: three random scenario seeds,
4 duplication levels, two reader prompts. The hidden truth is used only for
post-hoc reward/action-flip summaries.
"""
from __future__ import annotations
import argparse,json,re,random,time,os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DEFAULT='/home/xhj/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5'

def parse_obj(s):
    for cand in [s, *re.findall(r'```(?:json)?\s*(.*?)```',s,re.S), *re.findall(r'\{.*?\}',s,re.S)]:
        try:
            x=json.loads(cand)
            if isinstance(x,dict): return x
        except: pass
    return {'parse_error':s[-500:]}

def scenarios(seed):
    rng=random.Random(seed); out=[]
    for i in range(3):
        truth=bool((seed+i)%2); prior=0.45 + 0.1*((seed+i)%3)
        # A and B are counterfactual actions under one shared premise.
        values_false={'safe':0.8,'commit':-1.4}; values_true={'safe':0.7,'commit':1.8}
        rollouts=[]
        for h in (False,True):
            for action in ('safe','commit'):
                v=(values_true if h else values_false)[action]
                for j in range(2):
                    rollouts.append({'source_id':f's{j%2}','sample_id':f'h{int(h)}-{action}-{j}','premise':'H','hypothesis':h,'action':action,'value':v})
        out.append({'scenario_id':f'seed{seed}-{i}','truth':truth,'prior':prior,'values_false':values_false,'values_true':values_true,'rollouts':rollouts})
    return out

def load_model(path):
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float16)
    requested_device=os.environ.get('LOCAL_QWEN_DEVICE','cuda')
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('LOCAL_QWEN_DEVICE=cuda requested but CUDA is unavailable')
    model=model.to(requested_device).eval()
    return tok,model

def call(tok,model,prompt,max_new_tokens=80):
    msgs=[{'role':'system','content':'You are a calibrated decision reader. Return only one JSON object with keys confidence (number 0 to 1), action (string), and rationale (short string).'}, {'role':'user','content':prompt}]
    x=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors='pt'); x=x['input_ids'] if hasattr(x,'keys') else x
    device=next(model.parameters()).device; x=x.to(device)
    with torch.inference_mode(): y=model.generate(x,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(y[0,x.shape[-1]:],skip_special_tokens=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='results_submission/local_qwen'); p.add_argument('--model',default=MODEL_DEFAULT); p.add_argument('--seeds',default='11,29,47'); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); tok,model=load_model(a.model)
    rows=[]
    for seed in [int(x) for x in a.seeds.split(',')]:
      for sc in scenarios(seed):
       for dup in (1,2,4,8):
        base='\n'.join(f"[{r['source_id']}] premise H={str(r['hypothesis']).lower()}, action={r['action']}, predicted_value={r['value']:+.1f}" for r in sc['rollouts'])
        text='\n'.join([base]*dup)
        for method in ('flat','provenance'):
          if method=='flat':
            extra='Treat every listed rollout as an independent piece of evidence. Estimate P(H=true) and select the higher expected-value action.'
          else:
            extra='Rollouts with the same source_id and premise are descendants of one sample. Count each source/premise group once; repeated copies add no information. Estimate P(H=true) and select the higher expected-value action.'
          prompt=f"Prior P(H=true)={sc['prior']:.2f}. {extra}\nRollout bank:\n{text}\nReturn JSON now."
          raw=call(tok,model,prompt); obj=parse_obj(raw)
          try: conf=float(obj.get('confidence')); conf=max(0,min(1,conf))
          except: conf=None
          rows.append({'seed':seed,'scenario_id':sc['scenario_id'],'truth':sc['truth'],'duplication':dup,'method':method,'confidence':conf,'action':str(obj.get('action','')),'raw':raw})
          print(seed,sc['scenario_id'],dup,method,obj.get('confidence'),obj.get('action'))
    (out/'raw.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    # summary/action flips/reward: reward is the table value under hidden truth.
    for r in rows:
      sc=next(s for seed in [int(x) for x in a.seeds.split(',')] for s in scenarios(seed) for s in [s] if s['scenario_id']==r['scenario_id'])
      r['reward']=sc['values_true' if r['truth'] else 'values_false'].get(r['action'].strip().lower(),-2.0)
    import pandas as pd
    df=pd.DataFrame(rows); df['baseline_action']=df.groupby(['seed','scenario_id','method'])['action'].transform(lambda x:x.iloc[0]); df['flip_from_dup1']=(df['action']!=df['baseline_action']).astype(int)
    summary=df.groupby(['duplication','method'],dropna=False).agg(n=('action','size'),confidence_mean=('confidence','mean'),confidence_std=('confidence','std'),reward_mean=('reward','mean'),flip_rate=('flip_from_dup1','mean')).reset_index()
    df.to_csv(out/'raw.csv',index=False); summary.to_csv(out/'summary.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'model':a.model,'seeds':[int(x) for x in a.seeds.split(',')],'duplications':[1,2,4,8],'note':'diagnostic local Qwen; hidden truth only used for post-hoc reward'},indent=2))
    print(summary.to_string(index=False))
if __name__=='__main__': main()
