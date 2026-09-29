"""Summarize the GPU local-Qwen find-animal value-aggregation control."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(directory: Path, duplication: int) -> pd.DataFrame:
    rows = []
    for row in json.loads((directory / "summaries.json").read_text()):
        if row.get("error") is None:
            row = dict(row); row["duplication"] = duplication; rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(values, seed=20260929, draws=20000):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def main() -> None:
    out = Path("results_submission/report/local_qwen_animal_value_replay"); out.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([load(Path("results_submission/scienceworld_local_qwen_animal153_v9_m1"), 1), load(Path("results_submission/scienceworld_local_qwen_animal153_v9_m4"), 4)], ignore_index=True)
    frame.to_csv(out / "rows.csv", index=False)
    summary = frame.groupby(["duplication", "method"], as_index=False).agg(n=("variation", "size"), success=("success", "mean"), final_score=("final_score", "mean"), reward=("reward", "mean"), steps=("steps", "mean"), tokens=("charged_tokens", "mean"))
    summary.to_csv(out / "summary.csv", index=False)
    wide = frame.pivot_table(index=["task", "variation"], columns=["method", "duplication"], values=["success", "final_score", "reward", "steps", "charged_tokens"], aggfunc="first")
    paired=[]
    for (task,variation), row in wide.iterrows():
        rec={"task":task,"variation":int(variation)}
        for metric in ("success","final_score","reward","steps","charged_tokens"):
            for dup in (1,4):
                rec[f"flat_m{dup}_{metric}"]=row.get((metric,"flat",dup)); rec[f"provenance_value_m{dup}_{metric}"]=row.get((metric,"provenance_value",dup)); rec[f"provenance_value_minus_flat_m{dup}_{metric}"]=rec[f"provenance_value_m{dup}_{metric}"]-rec[f"flat_m{dup}_{metric}"]
        paired.append(rec)
    paired=pd.DataFrame(paired); paired.to_csv(out/'paired.csv',index=False)
    contrasts=[]
    for metric in ('success','reward','steps'):
        for dup in (1,4):
            est,lo,hi=bootstrap(paired[f'provenance_value_minus_flat_m{dup}_{metric}'].dropna().to_numpy(float)); contrasts.append({'duplication':dup,'metric':metric,'n_episodes':len(paired),'estimate':est,'ci95_low':lo,'ci95_high':hi,'bootstrap_draws':20000})
    pd.DataFrame(contrasts).to_csv(out/'bootstrap.csv',index=False)
    request_files=sum(len(list((Path(p)/'requests').glob('*.json'))) for p in ('results_submission/scienceworld_local_qwen_animal153_v9_m1','results_submission/scienceworld_local_qwen_animal153_v9_m4'))
    meta={'protocol':'local-qwen-find-animal-value-v1','model':'Qwen2.5-Coder-3B-Instruct','task':'find-animal','variation':153,'duplications':[1,4],'methods':['flat','provenance_value'],'valid_rows':int(len(frame)),'paired_episodes':int(len(paired)),'error_rows':0,'new_request_files':request_files,'planner_claim':False,'cuda_run_policy':'LOCAL_QWEN_DEVICE=cuda; RTX 4090 FP16','interpretation':'single second-task-family control; exploratory and separate from the plant cache replay'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)); print(json.dumps(meta,indent=2)); print(summary.to_string(index=False)); print(paired.to_string(index=False))


if __name__=='__main__': main()
