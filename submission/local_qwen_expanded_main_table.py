"""Merge the original CUDA-Qwen ScienceWorld table with the fresh expansion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def boot(x, seed=20260929, draws=20000):
    x = np.asarray(x, dtype=float)
    if not len(x): return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    y = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(y, .025)), float(np.quantile(y, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("results_submission/report/local_qwen_interactive_expanded_table"))
    ap.add_argument("--expansion", type=Path, nargs="+", default=[
        Path("results_submission/report/local_qwen_expansion_162_164"),
        Path("results_submission/report/local_qwen_expansion_165_167"),
    ])
    args = ap.parse_args()
    old = pd.read_csv("results_submission/report/local_qwen_interactive_main_table/rows.csv")
    expansions = [pd.read_csv(path / "rows.csv") for path in args.expansion]
    new = pd.concat(expansions, ignore_index=True, sort=False)
    # The two reports use the same runner schema; normalize the family field.
    old["task_family"] = old["task"].map(lambda x: "find-plant" if "plant" in str(x) else "find-animal")
    new["task_family"] = new["task"].map(lambda x: "find-plant" if "plant" in str(x) else "find-animal")
    frame = pd.concat([old, new], ignore_index=True, sort=False)
    frame["unnecessary_actions"] = frame["repeated_no_visible_change"].astype(float)
    frame["calibration_brier"] = (frame["initial_success_probability"] - frame["success"]) ** 2
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "rows.csv", index=False)
    summary = frame.groupby(["task_family", "duplication", "method"], as_index=False).agg(
        n=("variation", "size"), success=("success", "mean"), reward=("reward", "mean"),
        final_score=("final_score", "mean"), steps=("steps", "mean"),
        unnecessary_actions=("unnecessary_actions", "mean"), calibration_brier=("calibration_brier", "mean"),
        tokens=("charged_tokens", "mean"))
    summary.to_csv(args.output / "summary.csv", index=False)
    paired=[]
    for (family,var), g in frame.groupby(["task_family","variation"]):
        req={("flat",1),("flat",4),("provenance_value",1),("provenance_value",4)}
        if not req <= set(zip(g.method,g.duplication)): continue
        w=g.set_index(["method","duplication"]); r={"task_family":family,"variation":int(var)}
        for metric in ("success","reward","final_score","steps","unnecessary_actions","calibration_brier","charged_tokens"):
            for d in (1,4):
                r[f"provenance_value_minus_flat_m{d}_{metric}"]=float(w.loc[("provenance_value",d),metric]-w.loc[("flat",d),metric])
                r[f"flat_m{d}_{metric}"]=float(w.loc[("flat",d),metric]); r[f"provenance_value_m{d}_{metric}"]=float(w.loc[("provenance_value",d),metric])
        paired.append(r)
    paired=pd.DataFrame(paired); paired.to_csv(args.output/"paired.csv",index=False)
    contrasts=[]
    for family,g in paired.groupby("task_family"):
        for d in (1,4):
            for metric in ("success","reward","final_score","steps","unnecessary_actions","calibration_brier","charged_tokens"):
                e,l,h=boot(g[f"provenance_value_minus_flat_m{d}_{metric}"].dropna().to_numpy(),20260929+d)
                contrasts.append({"task_family":family,"duplication":d,"metric":metric,"n_episodes":int(g.shape[0]),"estimate":e,"ci95_low":l,"ci95_high":h,"bootstrap_draws":20000})
    pd.DataFrame(contrasts).to_csv(args.output/"contrasts.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    for ax,metric,title in zip(axes,("success","reward"),("Episode success","Mean reward")):
        z=summary.pivot_table(index=["task_family","duplication"],columns="method",values=metric)
        labels=[f"{f.replace('find-','')}\nm={d}" for f,d in z.index]; x=np.arange(len(labels)); width=.36
        for i,m in enumerate(["flat","provenance_value"]): ax.bar(x+(i-.5)*width,z[m].to_numpy(),width,label=m)
        ax.set_xticks(x,labels); ax.set_title(title); ax.grid(axis='y',alpha=.25)
    axes[1].legend(fontsize=8); fig.savefig(args.output/"curve.png",dpi=160); plt.close(fig)
    metadata={"protocol":"local-qwen-interactive-expanded-table-v2","model":"Qwen2.5-Coder-3B-Instruct","task_families":sorted(frame.task_family.unique()),"methods":["flat","provenance_value"],"duplications":[1,4],"valid_rows":int(len(frame)),"paired_episodes":int(len(paired)),"plant_paired_episodes":int((paired.task_family=='find-plant').sum()),"animal_paired_episodes":int((paired.task_family=='find-animal').sum()),"fresh_expansions":[str(path) for path in args.expansion],"planner_claim":False,"interpretation":"stratified exploratory CUDA-Qwen table; legacy and fresh paired episodes remain task-family strata"}
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)); print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
