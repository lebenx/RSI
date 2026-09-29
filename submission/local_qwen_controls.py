"""Local Qwen presentation-control audit, no API fallback."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import pandas as pd
from submission.local_llm_subset import MODEL_DEFAULT, scenarios, load_model, call, parse_obj

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='results_submission/local_qwen_controls');p.add_argument('--model',default=MODEL_DEFAULT);p.add_argument('--seeds',default='11,29,47');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);tok,model=load_model(a.model); rows=[]
 for seed in [int(x) for x in a.seeds.split(',')]:
  for sc in scenarios(seed)[:2]:
   for dup in (1,4):
    base=[f"[{r['source_id']}] premise H={str(r['hypothesis']).lower()}, action={r['action']}, predicted_value={r['value']:+.1f}" for r in sc['rollouts']]
    for layout in ('front','reverse'):
     for paraphrase in (False,True):
      text_rows=base if not paraphrase else [x.replace('premise H=','under shared condition H=') for x in base]
      text=list(text_rows)*dup
      if layout=='reverse': text=list(reversed(text))
      for prompt_variant in ('neutral','cautious'):
       extra='Treat copies as independent evidence.' if prompt_variant=='neutral' else 'Copies with the same source_id add no new information; count each source once.'
       prompt=f"Prior P(H=true)={sc['prior']:.2f}. {extra} Estimate P(H=true) and choose safe or commit. Return JSON keys confidence (0-1), action, rationale.\nRollouts:\n"+'\n'.join(text)
       obj=parse_obj(call(tok,model,prompt));
       try: conf=float(obj.get('confidence')); action=str(obj.get('action'))
       except: conf=None;action=''
       truth_values=sc['values_true' if sc['truth'] else 'values_false']; reward=truth_values.get(action.strip().lower(),None)
       rows.append({'seed':seed,'scenario_id':sc['scenario_id'],'truth':sc['truth'],'duplication':dup,'layout':layout,'paraphrase':paraphrase,'prompt_variant':prompt_variant,'confidence':conf,'action':action,'reward':reward,'brier':(conf-float(sc['truth']))**2 if conf is not None else None})
       print(seed,sc['scenario_id'],dup,layout,paraphrase,prompt_variant,conf,action,flush=True)
 df=pd.DataFrame(rows); ref=df[df.duplication==1].set_index(['seed','scenario_id','layout','paraphrase','prompt_variant']).confidence.rename('confidence_m1'); df=df.join(ref,on=['seed','scenario_id','layout','paraphrase','prompt_variant']); df['confidence_delta']=df.confidence-df.confidence_m1; df['action_flip_from_m1']=(df.action!=df.groupby(['seed','scenario_id','layout','paraphrase','prompt_variant']).action.transform('first')).astype(int)
 df.to_csv(out/'raw.csv',index=False); df.groupby(['duplication','layout','paraphrase','prompt_variant']).agg(n=('confidence','count'),confidence=('confidence','mean'),confidence_delta=('confidence_delta','mean'),action_flip=('action_flip_from_m1','mean'),reward=('reward','mean'),brier=('brier','mean')).reset_index().to_csv(out/'summary.csv',index=False)
 (out/'metadata.json').write_text(json.dumps({'model':a.model,'seeds':[int(x) for x in a.seeds.split(',')],'scenario_count':6,'duplications':[1,4],'layouts':['front','reverse'],'paraphrase':[False,True],'prompt_variants':['neutral','cautious'],'fallback':False},indent=2))
 print('wrote',len(df))
if __name__=='__main__': main()
