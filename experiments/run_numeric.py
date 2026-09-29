"""Reproduce the analytical grid without API access."""
import argparse, csv, json, itertools, time, hashlib
from pathlib import Path
import numpy as np
from experiments.core import *

def run(out, seeds=30):
    out.mkdir(parents=True,exist_ok=True)
    n=0; started=time.time(); audit=[]
    with (out/'numeric_raw.csv').open('w') as f, (out/'scenario_manifest.jsonl').open('w') as manifest:
        writer=None
        for seed,state,direction,prior,strength,family in itertools.product(
                range(seeds),(0,1),(0,1),(.1,.3,.5,.7,.9),(1.25,2.,4.),(0,1)):
            s=scenario(seed,state,direction,prior,strength,family)
            manifest.write(json.dumps({**public_metadata(s),'sensors':s['sensors'].tolist(),
                'mu':s['mu'].tolist(),'execution_noise':s['execution_noise'].tolist(),
                'bank':[asdict(x) for x in s['bank']]})+'\n')
            for condition in CONDITIONS:
                baseline={}
                for m in MULTIPLICITIES:
                    q,rows=intervention(s,condition,m)
                    order_seed=int(hashlib.sha256(f'{s["scenario_id"]}|{condition}|{m}'.encode()).hexdigest()[:8],16)
                    order_rng=np.random.default_rng(order_seed)
                    rows=list(rows); order_rng.shuffle(rows)
                    for method in METHODS:
                        p,values=predict(q,rows,method,strength)
                        met=metrics(s,q,p,values)
                        if m==1: baseline[method]=met
                        b=baseline[method]
                        row={**public_metadata(s),'condition':condition,'multiplicity':m,'method':method,
                             'q_evidence':q,'unique_samples':len({r.sid for r in rows}),
                             'presented_samples':len(rows),**met,
                             'order_seed':order_seed,'order_randomized':True,
                             'equal_text_length':len({len(r.text) for r in rows})==1,
                             'confidence_delta':p-b['root_confidence'],
                             'abs_confidence_delta':abs(p-b['root_confidence']),
                             'action_flip':int(met['action']!=b['action']),
                             'reward_delta':met['reward']-b['reward'],
                             'expected_reward_delta':met['expected_reward']-b['expected_reward'],
                             'harmful_flip':int(met['action']!=b['action'] and met['reward']<b['reward']),
                             'helpful_flip':int(met['action']!=b['action'] and met['reward']>b['reward'])}
                        if writer is None:
                            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row); n+=1
            if seed%5==0 and state==direction==family==0 and prior==.1 and strength==1.25:
                print(f'seed={seed} rows={n}',flush=True)
    meta={'seeds':seeds,'scenarios':seeds*120,'rows':n,'seconds':time.time()-started,
          'api_calls':0,'protocol':'experiments/protocol.md','source_mode':'oracle_generation_ids',
          'methods':METHODS,'conditions':CONDITIONS,'multiplicities':MULTIPLICITIES}
    (out/'numeric_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results_paper');p.add_argument('--seeds',type=int,default=30)
    a=p.parse_args();run(Path(a.output),a.seeds)
