"""Machine-readable audit of the current paper evidence boundary."""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_submission/report/evidence_audit.json'

def file_ok(path): return (ROOT/path).exists()

def main():
    checks=[]
    def add(name,status,evidence,note=''):
        checks.append({'requirement':name,'status':status,'evidence':evidence,'note':note})
    theory=(ROOT/'paper/theory.md').read_text()
    props=all(f'Proposition {i}' in theory for i in range(1,7)) and 'Corollary' in theory and 'Provenance-preserving aggregation algorithm' in theory
    props = props and 'Proposition 7 (observability limit for provenance recovery)' in theory
    add('theory propositions', 'verified' if props else 'missing', 'paper/theory.md', 'seven propositions, aggregation algorithm, corollary, and observability limit present')
    stats=json.loads((ROOT/'results_submission/benchmark/stats.json').read_text())
    add('benchmark scale/splits', 'verified' if stats['instances']>=5000 and set(stats['split_counts'])=={'train','dev','test'} else 'missing', 'results_submission/benchmark/stats.json', json.dumps(stats['split_counts']))
    required_families={'hidden_state_dependency','permission_tool_availability','object_property_uncertainty','resource_constraint','planning_failure'}
    add('benchmark task families', 'verified' if set(stats['family_counts'])==required_families else 'missing', 'results_submission/benchmark/stats.json')
    add('benchmark lineage visualization', 'verified' if file_ok(Path('results_submission/benchmark/lineage_example.svg')) else 'missing', 'results_submission/benchmark/lineage_example.svg')
    full_rows=0; lineage_ok=True; public_ok=True
    for split in ('train','dev','test'):
        full_path=ROOT/f'results_submission/benchmark/{split}.jsonl'; public_path=ROOT/f'results_submission/benchmark/{split}_public.jsonl'
        if not (full_path.exists() and public_path.exists()):
            lineage_ok=False; public_ok=False; continue
        with full_path.open() as f, public_path.open() as pf:
            for line, public_line in zip(f, pf):
                row=json.loads(line); public_row=json.loads(public_line); full_rows+=1
                lineage_ok = lineage_ok and bool(row.get('provenance_graph')) and bool(row.get('rollout_lineage')) and all(k in row for k in ('hidden_state','root_premises'))
                public_ok = public_ok and not any(k in public_row for k in ('hidden_state','provenance_graph','rollout_lineage','root_premises')) and bool(public_row.get('public_rollouts'))
    add('benchmark graph and public lineage contract', 'verified' if lineage_ok and public_ok and full_rows>=5000 else 'partial', 'results_submission/benchmark/{train,dev,test}{,_public}.jsonl', f'full_rows={full_rows}; hidden graph private={lineage_ok}; public_view_filtered={public_ok}')
    from submission.benchmark_eval import METHODS
    expected={'no_imagination','flat_rollout','independent_trajectory','mean_pooling','exact_dedup','semantic_dedup','source_grouping','bayesian_mixing','risk_classifier','oracle_provenance','provenance_preserving'}
    add('eleven baselines', 'verified' if set(METHODS)==expected else 'missing', 'submission/benchmark_eval.py', ','.join(METHODS))
    deepseek_path=Path('results_submission/deepseek_controls/summary.csv')
    if file_ok(deepseek_path):
        deepseek=pd.read_csv(ROOT/deepseek_path)
        deepseek_ok=set(['confidence','action_flip','reward','brier','prompt_tokens']).issubset(deepseek.columns) and len(deepseek)>0
        add('DeepSeek audit', 'verified' if deepseek_ok else 'partial', str(deepseek_path), f'rows={len(deepseek)}; metrics=confidence/action_flip/reward/brier/tokens')
    else:
        add('DeepSeek audit', 'missing', str(deepseek_path))
    mvp_dir=ROOT/'results_submission/deepseek_mvp_20260929'
    mvp_paths=[mvp_dir/'metadata.json',mvp_dir/'rows.csv',mvp_dir/'summary.csv',mvp_dir/'paired.csv',mvp_dir/'bootstrap.csv',mvp_dir/'same_body_control.csv']
    if all(p.exists() for p in mvp_paths):
        mvp_meta=json.loads((mvp_dir/'metadata.json').read_text()); mvp_rows=pd.read_csv(mvp_dir/'rows.csv'); mvp_control=pd.read_csv(mvp_dir/'same_body_control.csv')
        mvp_ok=(mvp_meta.get('seeds')==[0,1,2] and mvp_meta.get('duplications')==[1,2,4,8] and mvp_meta.get('valid_rows')==60 and mvp_meta.get('archived_request_files')==30 and mvp_meta.get('same_body_control_all_identical') is True and len(mvp_rows)==60 and len(mvp_control)==12)
        add('DeepSeek three-seed MVP', 'verified' if mvp_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in mvp_paths), f"valid_rows={mvp_meta.get('valid_rows')}; request_files={mvp_meta.get('archived_request_files')}; same_body_identical={mvp_meta.get('same_body_control_all_identical')}")
    else:
        add('DeepSeek three-seed MVP', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in mvp_paths))
    qwen_path=Path('results_submission/local_qwen_controls/summary.csv')
    if file_ok(qwen_path):
        qwen=pd.read_csv(ROOT/qwen_path)
        qwen_ok=set(['confidence','action_flip','reward','brier']).issubset(qwen.columns) and len(qwen)>0
        add('local Qwen audit', 'verified' if qwen_ok else 'partial', str(qwen_path), f'rows={len(qwen)}; metrics=confidence/action_flip/reward/brier')
    else:
        add('local Qwen audit', 'missing', str(qwen_path))
    scale=pd.read_csv(ROOT/'results_submission/scaling/baseline_summary.csv')
    budgets=set(scale['budget'].astype(int))
    add('scaling 1..64', 'verified' if budgets=={1,2,4,8,16,32,64} else 'missing', 'results_submission/scaling/baseline_summary.csv', str(sorted(budgets)))
    controlled_bootstrap_path=Path('results_submission/report/controlled_bootstrap.csv')
    if file_ok(controlled_bootstrap_path):
        cb=pd.read_csv(ROOT/controlled_bootstrap_path)
        cb_ok=(set(cb['budget'].astype(int))=={1,2,4,8,16,32,64} and set(cb['metric'])=={'root_confidence','reward','regret','action_flip_rate'} and cb['n_tasks'].nunique()==1 and int(cb['n_tasks'].iloc[0])==78 and cb['bootstrap_draws'].nunique()==1 and int(cb['bootstrap_draws'].iloc[0])==20000)
        add('controlled task-cluster uncertainty', 'verified' if cb_ok else 'partial', str(controlled_bootstrap_path), f"rows={len(cb)}; tasks={cb['n_tasks'].iloc[0] if len(cb) else 0}; draws={cb['bootstrap_draws'].iloc[0] if len(cb) else 0}")
    else:
        add('controlled task-cluster uncertainty', 'missing', str(controlled_bootstrap_path))
    add('controlled uncertainty figure', 'verified' if file_ok(Path('results_submission/report/controlled_curves_ci.png')) else 'missing', 'results_submission/report/controlled_curves_ci.png')
    main_ci_path=Path('results_submission/report/main_table_ci.csv')
    if file_ok(main_ci_path):
        main_ci=pd.read_csv(ROOT/main_ci_path)
        main_ci_ok={'condition','budget','method','estimate_action_flip_rate','ci95_low_reward','ci95_high_reward'}.issubset(main_ci.columns) and set(main_ci.budget.astype(int))=={1,8,64}
        add('main table with confidence intervals', 'verified' if main_ci_ok else 'partial', str(main_ci_path), f"rows={len(main_ci)}; budgets={sorted(main_ci.budget.astype(int).unique()) if len(main_ci) else []}")
    else:
        add('main table with confidence intervals', 'missing', str(main_ci_path))
    independent_paths=[Path('results_submission/report/independent_scaling.csv'),Path('results_submission/report/independent_paired.csv'),Path('results_submission/report/independent_scaling.png')]
    if all(file_ok(p) for p in independent_paths):
        independent=pd.read_csv(ROOT/independent_paths[0]); paired_ind=pd.read_csv(ROOT/independent_paths[1])
        independent_ok=(set(independent['budget'].astype(int))=={1,2,4,8,16,32,64} and {'independent_trajectory','provenance_preserving','flat_rollout'}.issubset(set(independent['method'])) and len(paired_ind)>0)
        add('independent-rollout value axis', 'verified' if independent_ok else 'partial', ';'.join(str(p) for p in independent_paths), f"summary_rows={len(independent)}; paired_rows={len(paired_ind)}")
    else:
        add('independent-rollout value axis', 'missing', ';'.join(str(p) for p in independent_paths))
    value_mse_paths=[Path('results_submission/report/value_estimation_mse.csv'),Path('results_submission/report/value_estimation_mse.png')]
    if all(file_ok(p) for p in value_mse_paths):
        value_mse=pd.read_csv(ROOT/value_mse_paths[0])
        mse_ok=(set(value_mse['condition'])=={'independent_rollout','pure_duplication'} and set(value_mse['budget'].astype(int))=={1,2,4,8,16,32,64} and set(value_mse['groups'].astype(int))=={624})
        add('conditional-value variance reduction', 'verified' if mse_ok else 'partial', ';'.join(str(p) for p in value_mse_paths), f"groups={value_mse['groups'].iloc[0] if len(value_mse) else 0}; conditions={','.join(sorted(value_mse.condition.unique())) if len(value_mse) else ''}")
    else:
        add('conditional-value variance reduction', 'missing', ';'.join(str(p) for p in value_mse_paths))
    ab=pd.read_csv(ROOT/'results_submission/ablations/ablation_summary.csv')
    ab_methods=set(ab['method'])
    add('ablation methods', 'verified' if len(ab_methods)>=7 else 'partial', 'results_submission/ablations/ablation_summary.csv', ','.join(sorted(ab_methods)))
    prov=json.loads((ROOT/'results_submission/provenance/metadata.json').read_text())
    hard=json.loads((ROOT/'results_submission/provenance_semantic_hard_final/summary.json').read_text())
    add('learned provenance lexical audit', 'verified' if prov['test_edge_f1']==1.0 and prov['test_source_edge_f1']==1.0 else 'partial', 'results_submission/provenance/metadata.json', 'lexical upper bound')
    add('learned provenance semantic-hard audit', 'verified' if hard['summaries'][0]['micro_f1']<1.0 and hard['summaries'][1]['micro_f1']<1.0 and 'downstream' in hard else 'missing', 'results_submission/provenance_semantic_hard_final/summary.json', f"premise_f1={hard['summaries'][0]['micro_f1']:.3f}; source_f1={hard['summaries'][1]['micro_f1']:.3f}")
    null_path=Path('results_submission/provenance_semantic_hard_final/identifiability_null.csv')
    if file_ok(null_path):
        null_df=pd.read_csv(ROOT/null_path)
        null_ok={'target','actual_f1','null_f1_mean','null_f1_ci95_low','null_f1_ci95_high','permutation_draws'}.issubset(null_df.columns) and set(null_df.target)=={'premise','source'} and set(null_df.permutation_draws.astype(int))=={2000}
        add('learned provenance identifiability null', 'verified' if null_ok else 'partial', str(null_path), f"targets={','.join(sorted(null_df.target.astype(str).unique()))}; draws={int(null_df.permutation_draws.iloc[0]) if len(null_df) else 0}")
    else:
        add('learned provenance identifiability null', 'missing', str(null_path))
    margin_path=Path('results_submission/provenance_semantic_hard_final/margin_analysis.csv')
    if file_ok(margin_path):
        margin=pd.read_csv(ROOT/margin_path)
        margin_ok={'target','recovery','margin_bin','action_correct','cross_root_merge_rate'}.issubset(margin.columns) and set(margin['recovery'])=={'learned','oracle'} and len(margin)>0
        add('provenance recovery margin diagnostic', 'verified' if margin_ok else 'partial', str(margin_path), f"rows={len(margin)}; learned_bins={int(((margin.recovery=='learned') & (margin.n>0)).sum()) if 'n' in margin else 0}")
    else:
        add('provenance recovery margin diagnostic', 'missing', str(margin_path))
    add('provenance recovery breakdown/curve', 'verified' if file_ok(Path('results_submission/report/provenance_edge_breakdown.csv')) and file_ok(Path('results_submission/report/provenance_noise_curve.csv')) and file_ok(Path('results_submission/report/provenance_noise_curve.png')) else 'missing', 'results_submission/report/provenance_edge_breakdown.csv;provenance_noise_curve.csv;provenance_noise_curve.png')
    sw=pd.read_csv(ROOT/'results_submission/report/scienceworld_v7_dev8_summary.csv')
    interactive_metrics={'success','reward','regret_proxy','steps','tokens','calibration_brier','action_flip_rate'}
    add('interactive metric coverage', 'verified' if interactive_metrics.issubset(sw.columns) else 'partial', 'results_submission/report/scienceworld_v7_dev8_summary.csv', ','.join(sorted(interactive_metrics.intersection(sw.columns))))
    local_sw_path=Path('results_submission/report/scienceworld_local_qwen_compact_summary.csv')
    if file_ok(local_sw_path):
        local_sw=pd.read_csv(ROOT/local_sw_path)
        local_sw_ok=set(local_sw['method'])=={'flat','provenance'} and len(local_sw)==2 and set(local_sw['success'].astype(int))=={0}
        grid_path=Path('results_submission/report/scienceworld_local_qwen_compact_grid_summary.csv')
        grid_note=''
        if file_ok(grid_path):
            grid=pd.read_csv(ROOT/grid_path)
            grid_ok=set(grid['method'])=={'flat','provenance'} and len(grid)==6 and set(grid['variation'].astype(int))=={152,158,161}
            grid_note=f"; grid_rows={len(grid)}; grid_schema={'ok' if grid_ok else 'check'}"
        add('local Qwen interactive schema smoke', 'partial' if local_sw_ok else 'missing', str(local_sw_path) + (f'; {grid_path}' if file_ok(grid_path) else ''), f"valid_rows={len(local_sw)}; methods={','.join(sorted(local_sw.method)) if len(local_sw) else ''}; planner_claim=False{grid_note}")
    else:
        add('local Qwen interactive schema smoke', 'missing', str(local_sw_path))
    qwen_mvp_dir=ROOT/'results_submission/report/scienceworld_local_qwen_mvp'
    qwen_mvp_paths=[qwen_mvp_dir/'metadata.json', qwen_mvp_dir/'rows.csv', qwen_mvp_dir/'summary.csv', qwen_mvp_dir/'paired.csv']
    if all(p.exists() for p in qwen_mvp_paths):
        qwen_mvp_meta=json.loads((qwen_mvp_dir/'metadata.json').read_text())
        qwen_mvp_summary=pd.read_csv(qwen_mvp_dir/'summary.csv')
        qwen_mvp_ok=(qwen_mvp_meta.get('valid_rows')==4 and qwen_mvp_meta.get('error_rows')==0 and qwen_mvp_meta.get('duplications')==[1,4] and qwen_mvp_meta.get('methods')==['flat','provenance'] and qwen_mvp_meta.get('step_limit')==2 and qwen_mvp_meta.get('episode_success_claim') is False and qwen_mvp_meta.get('planner_claim') is False and qwen_mvp_meta.get('inference_device')=='cuda:0' and qwen_mvp_meta.get('gpu_probe_status')=='verified' and len(qwen_mvp_summary)==4)
        add('ScienceWorld local Qwen short-horizon smoke', 'verified' if qwen_mvp_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_mvp_paths), f"valid_rows={qwen_mvp_meta.get('valid_rows')}; errors={qwen_mvp_meta.get('error_rows')}; step_limit={qwen_mvp_meta.get('step_limit')}; device={qwen_mvp_meta.get('inference_device')}; planner_claim={qwen_mvp_meta.get('planner_claim')}")
    else:
        add('ScienceWorld local Qwen short-horizon smoke', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_mvp_paths))
    qwen_pilot_dir=ROOT/'results_submission/report/local_qwen_episode_pilot'
    qwen_pilot_paths=[qwen_pilot_dir/'metadata.json', qwen_pilot_dir/'rows.csv', qwen_pilot_dir/'summary.csv', qwen_pilot_dir/'paired.csv', qwen_pilot_dir/'bootstrap.csv']
    if all(p.exists() for p in qwen_pilot_paths):
        qwen_pilot_meta=json.loads((qwen_pilot_dir/'metadata.json').read_text()); qwen_pilot_rows=pd.read_csv(qwen_pilot_dir/'rows.csv'); qwen_pilot_pairs=pd.read_csv(qwen_pilot_dir/'paired.csv')
        qwen_pilot_ok=(qwen_pilot_meta.get('valid_rows')==12 and qwen_pilot_meta.get('error_rows')==0 and qwen_pilot_meta.get('paired_episodes')==3 and qwen_pilot_meta.get('duplications')==[1,4] and qwen_pilot_meta.get('methods')==['flat','provenance_success'] and qwen_pilot_meta.get('planner_claim') is False and len(qwen_pilot_rows)==12 and len(qwen_pilot_pairs)==3)
        add('local Qwen full-episode interactive pilot', 'verified' if qwen_pilot_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_pilot_paths), f"valid_rows={qwen_pilot_meta.get('valid_rows')}; paired_episodes={qwen_pilot_meta.get('paired_episodes')}; planner_claim={qwen_pilot_meta.get('planner_claim')}")
    else:
        add('local Qwen full-episode interactive pilot', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_pilot_paths))
    expanded_qwen_dir=ROOT/'results_submission/report/local_qwen_expanded_pilot'
    expanded_qwen_paths=[expanded_qwen_dir/'metadata.json', expanded_qwen_dir/'rows.csv', expanded_qwen_dir/'summary.csv', expanded_qwen_dir/'paired.csv', expanded_qwen_dir/'bootstrap.csv', expanded_qwen_dir/'raw_errors.csv', expanded_qwen_dir/'curve.png']
    if all(p.exists() for p in expanded_qwen_paths):
        expanded_qwen_meta=json.loads((expanded_qwen_dir/'metadata.json').read_text()); expanded_qwen_rows=pd.read_csv(expanded_qwen_dir/'rows.csv'); expanded_qwen_pairs=pd.read_csv(expanded_qwen_dir/'paired.csv')
        expanded_qwen_ok=(expanded_qwen_meta.get('valid_rows')==20 and expanded_qwen_meta.get('error_rows')==0 and expanded_qwen_meta.get('paired_episodes')==5 and expanded_qwen_meta.get('raw_extension_error_rows')==10 and expanded_qwen_meta.get('variations')==[152,153,154,158,161] and expanded_qwen_meta.get('duplications')==[1,4] and expanded_qwen_meta.get('methods')==['flat','provenance_success'] and expanded_qwen_meta.get('planner_claim') is False and len(expanded_qwen_rows)==20 and len(expanded_qwen_pairs)==5)
        add('local Qwen expanded full-episode interactive pilot', 'verified' if expanded_qwen_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in expanded_qwen_paths), f"valid_rows={expanded_qwen_meta.get('valid_rows')}; paired_episodes={expanded_qwen_meta.get('paired_episodes')}; planner_claim={expanded_qwen_meta.get('planner_claim')}")
    else:
        add('local Qwen expanded full-episode interactive pilot', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in expanded_qwen_paths))
    value_replay_dir=ROOT/'results_submission/report/local_qwen_value_replay'
    value_replay_paths=[value_replay_dir/'metadata.json', value_replay_dir/'rows.csv', value_replay_dir/'summary.csv', value_replay_dir/'paired.csv', value_replay_dir/'bootstrap.csv', value_replay_dir/'cache_audit.json']
    if all(p.exists() for p in value_replay_paths):
        value_replay_meta=json.loads((value_replay_dir/'metadata.json').read_text()); value_replay_rows=pd.read_csv(value_replay_dir/'rows.csv'); value_replay_pairs=pd.read_csv(value_replay_dir/'paired.csv'); value_replay_cache=json.loads((value_replay_dir/'cache_audit.json').read_text())
        value_replay_ok=(value_replay_meta.get('valid_rows')==20 and value_replay_meta.get('paired_episodes')==5 and value_replay_meta.get('new_model_requests')==0 and value_replay_meta.get('variations')==[152,153,154,158,161] and value_replay_meta.get('methods')==['flat','provenance_value'] and value_replay_meta.get('planner_claim') is False and len(value_replay_rows)==20 and len(value_replay_pairs)==5 and len(value_replay_cache)==6 and all(x.get('byte_identical') is True and x.get('new_request_files')==[] and x.get('changed_request_files')==[] for x in value_replay_cache))
        add('local Qwen cached provenance-value replay', 'verified' if value_replay_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in value_replay_paths), f"valid_rows={value_replay_meta.get('valid_rows')}; paired_episodes={value_replay_meta.get('paired_episodes')}; new_model_requests={value_replay_meta.get('new_model_requests')}")
    else:
        add('local Qwen cached provenance-value replay', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in value_replay_paths))
    animal_qwen_dir=ROOT/'results_submission/report/local_qwen_animal_value_replay'
    animal_qwen_paths=[animal_qwen_dir/'metadata.json', animal_qwen_dir/'rows.csv', animal_qwen_dir/'summary.csv', animal_qwen_dir/'paired.csv', animal_qwen_dir/'bootstrap.csv']
    if all(p.exists() for p in animal_qwen_paths):
        animal_qwen_meta=json.loads((animal_qwen_dir/'metadata.json').read_text()); animal_qwen_rows=pd.read_csv(animal_qwen_dir/'rows.csv'); animal_qwen_pairs=pd.read_csv(animal_qwen_dir/'paired.csv')
        animal_qwen_ok=(animal_qwen_meta.get('valid_rows')==4 and animal_qwen_meta.get('paired_episodes')==1 and animal_qwen_meta.get('task')=='find-animal' and animal_qwen_meta.get('variation')==153 and animal_qwen_meta.get('methods')==['flat','provenance_value'] and animal_qwen_meta.get('planner_claim') is False and len(animal_qwen_rows)==4 and len(animal_qwen_pairs)==1)
        add('local Qwen second-task-family value control', 'verified' if animal_qwen_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in animal_qwen_paths), f"valid_rows={animal_qwen_meta.get('valid_rows')}; paired_episodes={animal_qwen_meta.get('paired_episodes')}; planner_claim={animal_qwen_meta.get('planner_claim')}")
    else:
        add('local Qwen second-task-family value control', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in animal_qwen_paths))
    animal_expanded_dir=ROOT/'results_submission/report/local_qwen_animal_expanded_pilot'
    animal_expanded_paths=[animal_expanded_dir/'metadata.json', animal_expanded_dir/'rows.csv', animal_expanded_dir/'summary.csv', animal_expanded_dir/'paired.csv', animal_expanded_dir/'bootstrap.csv', animal_expanded_dir/'raw_errors.csv']
    if all(p.exists() for p in animal_expanded_paths):
        animal_expanded_meta=json.loads((animal_expanded_dir/'metadata.json').read_text()); animal_expanded_rows=pd.read_csv(animal_expanded_dir/'rows.csv'); animal_expanded_pairs=pd.read_csv(animal_expanded_dir/'paired.csv'); animal_expanded_errors=pd.read_csv(animal_expanded_dir/'raw_errors.csv')
        animal_expanded_ok=(animal_expanded_meta.get('valid_rows')==17 and animal_expanded_meta.get('error_rows')==7 and animal_expanded_meta.get('paired_episodes')==3 and animal_expanded_meta.get('paired_variations')==[150,153,155] and animal_expanded_meta.get('methods')==['flat','provenance_value'] and animal_expanded_meta.get('planner_claim') is False and len(animal_expanded_rows)==17 and len(animal_expanded_pairs)==3 and len(animal_expanded_errors)==7)
        add('local Qwen expanded second-task-family pilot', 'verified' if animal_expanded_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in animal_expanded_paths), f"valid_rows={animal_expanded_meta.get('valid_rows')}; errors={animal_expanded_meta.get('error_rows')}; paired_episodes={animal_expanded_meta.get('paired_episodes')}")
    else:
        add('local Qwen expanded second-task-family pilot', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in animal_expanded_paths))
    qwen_main_dir=ROOT/'results_submission/report/local_qwen_interactive_main_table'
    qwen_main_paths=[qwen_main_dir/'metadata.json', qwen_main_dir/'rows.csv', qwen_main_dir/'summary.csv', qwen_main_dir/'paired.csv', qwen_main_dir/'contrasts.csv', qwen_main_dir/'curve.png']
    if all(p.exists() for p in qwen_main_paths):
        qwen_main_meta=json.loads((qwen_main_dir/'metadata.json').read_text()); qwen_main_rows=pd.read_csv(qwen_main_dir/'rows.csv'); qwen_main_summary=pd.read_csv(qwen_main_dir/'summary.csv'); qwen_main_pairs=pd.read_csv(qwen_main_dir/'paired.csv'); qwen_main_contrasts=pd.read_csv(qwen_main_dir/'contrasts.csv')
        qwen_main_ok=(qwen_main_meta.get('valid_rows')==32 and qwen_main_meta.get('paired_episodes')==8 and qwen_main_meta.get('plant_paired_episodes')==5 and qwen_main_meta.get('animal_paired_episodes')==3 and set(qwen_main_meta.get('task_families',[]))=={'find-plant','find-animal'} and qwen_main_meta.get('methods')==['flat','provenance_value'] and qwen_main_meta.get('duplications')==[1,4] and qwen_main_meta.get('planner_claim') is False and len(qwen_main_rows)==32 and len(qwen_main_pairs)==8 and len(qwen_main_contrasts)>0)
        add('local Qwen interactive main table', 'verified' if qwen_main_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_main_paths), f"valid_rows={qwen_main_meta.get('valid_rows')}; paired_episodes={qwen_main_meta.get('paired_episodes')}; families={','.join(qwen_main_meta.get('task_families',[]))}")
    else:
        add('local Qwen interactive main table', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_main_paths))
    qwen_fresh_dir=ROOT/'results_submission/report/local_qwen_expansion_162_164'
    qwen_fresh_paths=[qwen_fresh_dir/'metadata.json', qwen_fresh_dir/'rows.csv', qwen_fresh_dir/'summary.csv', qwen_fresh_dir/'paired.csv', qwen_fresh_dir/'bootstrap.csv', qwen_fresh_dir/'action_consistency.csv', qwen_fresh_dir/'action_bootstrap.csv', qwen_fresh_dir/'curve.png']
    if all(p.exists() for p in qwen_fresh_paths):
        qwen_fresh_meta=json.loads((qwen_fresh_dir/'metadata.json').read_text()); qwen_fresh_rows=pd.read_csv(qwen_fresh_dir/'rows.csv'); qwen_fresh_pairs=pd.read_csv(qwen_fresh_dir/'paired.csv'); qwen_fresh_boot=pd.read_csv(qwen_fresh_dir/'bootstrap.csv'); qwen_fresh_action=pd.read_csv(qwen_fresh_dir/'action_consistency.csv')
        qwen_fresh_ok=(qwen_fresh_meta.get('valid_rows')==24 and qwen_fresh_meta.get('raw_error_rows')==0 and qwen_fresh_meta.get('paired_episodes')==6 and set(qwen_fresh_meta.get('task_families',[]))=={'find-plant','find-animal'} and qwen_fresh_meta.get('duplications')==[1,4] and qwen_fresh_meta.get('methods')==['flat','provenance_value'] and qwen_fresh_meta.get('episode_success_claim') is False and len(qwen_fresh_rows)==24 and len(qwen_fresh_pairs)==6 and len(qwen_fresh_boot)>0 and set(qwen_fresh_action.method)=={'flat','provenance_value'})
        add('local Qwen fresh CUDA paired expansion', 'verified' if qwen_fresh_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_fresh_paths), f"valid_rows={qwen_fresh_meta.get('valid_rows')}; paired_episodes={qwen_fresh_meta.get('paired_episodes')}; errors={qwen_fresh_meta.get('raw_error_rows')}; episode_success_claim={qwen_fresh_meta.get('episode_success_claim')}")
    else:
        add('local Qwen fresh CUDA paired expansion', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_fresh_paths))
    qwen_fresh2_dir=ROOT/'results_submission/report/local_qwen_expansion_165_167'
    qwen_fresh2_paths=[qwen_fresh2_dir/'metadata.json', qwen_fresh2_dir/'rows.csv', qwen_fresh2_dir/'summary.csv', qwen_fresh2_dir/'paired.csv', qwen_fresh2_dir/'bootstrap.csv', qwen_fresh2_dir/'raw_errors.csv', qwen_fresh2_dir/'action_consistency.csv', qwen_fresh2_dir/'action_bootstrap.csv', qwen_fresh2_dir/'curve.png']
    if all(p.exists() for p in qwen_fresh2_paths):
        qwen_fresh2_meta=json.loads((qwen_fresh2_dir/'metadata.json').read_text()); qwen_fresh2_rows=pd.read_csv(qwen_fresh2_dir/'rows.csv'); qwen_fresh2_pairs=pd.read_csv(qwen_fresh2_dir/'paired.csv'); qwen_fresh2_errors=pd.read_csv(qwen_fresh2_dir/'raw_errors.csv'); qwen_fresh2_action=pd.read_csv(qwen_fresh2_dir/'action_consistency.csv')
        qwen_fresh2_ok=(qwen_fresh2_meta.get('valid_rows')==23 and qwen_fresh2_meta.get('raw_error_rows')==1 and qwen_fresh2_meta.get('paired_episodes')==5 and set(qwen_fresh2_meta.get('task_families',[]))=={'find-plant','find-animal'} and qwen_fresh2_meta.get('duplications')==[1,4] and qwen_fresh2_meta.get('methods')==['flat','provenance_value'] and qwen_fresh2_meta.get('episode_success_claim') is False and len(qwen_fresh2_rows)==23 and len(qwen_fresh2_pairs)==5 and len(qwen_fresh2_errors)==1 and set(qwen_fresh2_action.method)=={'flat','provenance_value'} and all(float(x)==0.0 for x in qwen_fresh2_action[qwen_fresh2_action.method=='provenance_value'].action_flip_rate))
        add('local Qwen second fresh CUDA paired expansion', 'verified' if qwen_fresh2_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_fresh2_paths), f"valid_rows={qwen_fresh2_meta.get('valid_rows')}; paired_episodes={qwen_fresh2_meta.get('paired_episodes')}; errors={qwen_fresh2_meta.get('raw_error_rows')}; provenance_flip_zero={qwen_fresh2_ok}")
    else:
        add('local Qwen second fresh CUDA paired expansion', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_fresh2_paths))
    fair_qwen_dir=ROOT/'results_submission/report/local_qwen_fair_flatvalue_168_169'
    fair_qwen_paths=[fair_qwen_dir/'metadata.json', fair_qwen_dir/'rows.csv', fair_qwen_dir/'summary.csv', fair_qwen_dir/'paired.csv', fair_qwen_dir/'bootstrap.csv', fair_qwen_dir/'raw_errors.csv', fair_qwen_dir/'action_consistency.csv', fair_qwen_dir/'action_bootstrap.csv', fair_qwen_dir/'curve.png']
    if all(p.exists() for p in fair_qwen_paths):
        fair_qwen_meta=json.loads((fair_qwen_dir/'metadata.json').read_text()); fair_qwen_rows=pd.read_csv(fair_qwen_dir/'rows.csv'); fair_qwen_pairs=pd.read_csv(fair_qwen_dir/'paired.csv'); fair_qwen_errors=pd.read_csv(fair_qwen_dir/'raw_errors.csv')
        fair_qwen_ok=(fair_qwen_meta.get('valid_rows')==16 and fair_qwen_meta.get('raw_error_rows')==0 and fair_qwen_meta.get('paired_episodes')==4 and fair_qwen_meta.get('methods')==['flat_value','provenance_value'] and fair_qwen_meta.get('duplications')==[1,4] and fair_qwen_meta.get('episode_success_claim') is False and len(fair_qwen_rows)==16 and len(fair_qwen_pairs)==4 and len(fair_qwen_errors)==0 and 'flat_value_definition' in fair_qwen_meta)
        add('local Qwen fair algorithmic flat-value control', 'verified' if fair_qwen_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in fair_qwen_paths), f"valid_rows={fair_qwen_meta.get('valid_rows')}; paired_episodes={fair_qwen_meta.get('paired_episodes')}; errors={fair_qwen_meta.get('raw_error_rows')}; methods={','.join(fair_qwen_meta.get('methods',[]))}")
    else:
        add('local Qwen fair algorithmic flat-value control', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in fair_qwen_paths))
    qwen_retry_dir=ROOT/'results_submission/report/local_qwen_expansion_170_171_retry'
    qwen_retry_paths=[qwen_retry_dir/'metadata.json', qwen_retry_dir/'rows.csv', qwen_retry_dir/'summary.csv', qwen_retry_dir/'paired.csv', qwen_retry_dir/'bootstrap.csv', qwen_retry_dir/'raw_errors.csv', qwen_retry_dir/'action_consistency.csv', qwen_retry_dir/'action_bootstrap.csv', qwen_retry_dir/'curve.png']
    if all(p.exists() for p in qwen_retry_paths):
        qwen_retry_meta=json.loads((qwen_retry_dir/'metadata.json').read_text()); qwen_retry_rows=pd.read_csv(qwen_retry_dir/'rows.csv'); qwen_retry_pairs=pd.read_csv(qwen_retry_dir/'paired.csv'); qwen_retry_errors=pd.read_csv(qwen_retry_dir/'raw_errors.csv'); qwen_retry_action=pd.read_csv(qwen_retry_dir/'action_consistency.csv')
        qwen_retry_ok=(qwen_retry_meta.get('valid_rows')==16 and qwen_retry_meta.get('raw_error_rows')==0 and qwen_retry_meta.get('paired_episodes')==4 and set(qwen_retry_meta.get('task_families',[]))=={'find-plant','find-animal'} and qwen_retry_meta.get('duplications')==[1,4] and qwen_retry_meta.get('methods')==['flat','provenance_value'] and qwen_retry_meta.get('episode_success_claim') is False and 'candidate_schema_retry' in qwen_retry_meta and len(qwen_retry_rows)==16 and len(qwen_retry_pairs)==4 and len(qwen_retry_errors)==0 and set(qwen_retry_action.method)=={'flat','provenance_value'} and all(float(x)==0.0 for x in qwen_retry_action[qwen_retry_action.method=='provenance_value'].action_flip_rate))
        add('local Qwen retry-validated held-out expansion', 'verified' if qwen_retry_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_retry_paths), f"valid_rows={qwen_retry_meta.get('valid_rows')}; paired_episodes={qwen_retry_meta.get('paired_episodes')}; errors={qwen_retry_meta.get('raw_error_rows')}; candidate_retry=True")
    else:
        add('local Qwen retry-validated held-out expansion', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_retry_paths))
    qwen_combined_dir=ROOT/'results_submission/report/local_qwen_interactive_expanded_table_v3'
    qwen_combined_paths=[qwen_combined_dir/'metadata.json', qwen_combined_dir/'rows.csv', qwen_combined_dir/'summary.csv', qwen_combined_dir/'paired.csv', qwen_combined_dir/'contrasts.csv', qwen_combined_dir/'curve.png']
    if all(p.exists() for p in qwen_combined_paths):
        qwen_combined_meta=json.loads((qwen_combined_dir/'metadata.json').read_text()); qwen_combined_rows=pd.read_csv(qwen_combined_dir/'rows.csv'); qwen_combined_pairs=pd.read_csv(qwen_combined_dir/'paired.csv'); qwen_combined_contrasts=pd.read_csv(qwen_combined_dir/'contrasts.csv')
        qwen_combined_ok=(qwen_combined_meta.get('valid_rows')==95 and qwen_combined_meta.get('paired_episodes')==23 and qwen_combined_meta.get('plant_paired_episodes')==12 and qwen_combined_meta.get('animal_paired_episodes')==11 and qwen_combined_meta.get('planner_claim') is False and qwen_combined_meta.get('methods')==['flat','provenance_value'] and qwen_combined_meta.get('duplications')==[1,4] and len(qwen_combined_rows)==95 and len(qwen_combined_pairs)==23 and len(qwen_combined_contrasts)>0)
        add('local Qwen consolidated 23-pair table', 'verified' if qwen_combined_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_combined_paths), f"valid_rows={qwen_combined_meta.get('valid_rows')}; paired_episodes={qwen_combined_meta.get('paired_episodes')}; plant_pairs={qwen_combined_meta.get('plant_paired_episodes')}; animal_pairs={qwen_combined_meta.get('animal_paired_episodes')}")
    else:
        add('local Qwen consolidated 23-pair table', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_combined_paths))
    qwen_combined4_dir=ROOT/'results_submission/report/local_qwen_interactive_expanded_table_v4'
    qwen_combined4_paths=[qwen_combined4_dir/'metadata.json', qwen_combined4_dir/'rows.csv', qwen_combined4_dir/'summary.csv', qwen_combined4_dir/'paired.csv', qwen_combined4_dir/'contrasts.csv', qwen_combined4_dir/'curve.png']
    if all(p.exists() for p in qwen_combined4_paths):
        qwen_combined4_meta=json.loads((qwen_combined4_dir/'metadata.json').read_text()); qwen_combined4_rows=pd.read_csv(qwen_combined4_dir/'rows.csv'); qwen_combined4_pairs=pd.read_csv(qwen_combined4_dir/'paired.csv'); qwen_combined4_contrasts=pd.read_csv(qwen_combined4_dir/'contrasts.csv')
        qwen_combined4_ok=(qwen_combined4_meta.get('valid_rows')==111 and qwen_combined4_meta.get('paired_episodes')==27 and qwen_combined4_meta.get('plant_paired_episodes')==14 and qwen_combined4_meta.get('animal_paired_episodes')==13 and qwen_combined4_meta.get('planner_claim') is False and qwen_combined4_meta.get('methods')==['flat','provenance_value'] and qwen_combined4_meta.get('duplications')==[1,4] and len(qwen_combined4_rows)==111 and len(qwen_combined4_pairs)==27 and len(qwen_combined4_contrasts)>0)
        add('local Qwen consolidated 27-pair table', 'verified' if qwen_combined4_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_combined4_paths), f"valid_rows={qwen_combined4_meta.get('valid_rows')}; paired_episodes={qwen_combined4_meta.get('paired_episodes')}; plant_pairs={qwen_combined4_meta.get('plant_paired_episodes')}; animal_pairs={qwen_combined4_meta.get('animal_paired_episodes')}")
    else:
        add('local Qwen consolidated 27-pair table', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_combined4_paths))
    qwen_fourth_dir=ROOT/'results_submission/report/local_qwen_expansion_172_173'
    qwen_fourth_paths=[qwen_fourth_dir/'metadata.json', qwen_fourth_dir/'rows.csv', qwen_fourth_dir/'summary.csv', qwen_fourth_dir/'paired.csv', qwen_fourth_dir/'bootstrap.csv', qwen_fourth_dir/'raw_errors.csv', qwen_fourth_dir/'action_consistency.csv', qwen_fourth_dir/'action_bootstrap.csv', qwen_fourth_dir/'curve.png']
    if all(p.exists() for p in qwen_fourth_paths):
        qwen_fourth_meta=json.loads((qwen_fourth_dir/'metadata.json').read_text()); qwen_fourth_rows=pd.read_csv(qwen_fourth_dir/'rows.csv'); qwen_fourth_pairs=pd.read_csv(qwen_fourth_dir/'paired.csv'); qwen_fourth_errors=pd.read_csv(qwen_fourth_dir/'raw_errors.csv'); qwen_fourth_action=pd.read_csv(qwen_fourth_dir/'action_consistency.csv')
        qwen_fourth_ok=(qwen_fourth_meta.get('valid_rows')==16 and qwen_fourth_meta.get('raw_error_rows')==0 and qwen_fourth_meta.get('paired_episodes')==4 and qwen_fourth_meta.get('task_families')==['find-animal','find-plant'] and qwen_fourth_meta.get('duplications')==[1,4] and qwen_fourth_meta.get('methods')==['flat','provenance_value'] and qwen_fourth_meta.get('episode_success_claim') is False and 'candidate_schema_retry' in qwen_fourth_meta and len(qwen_fourth_rows)==16 and len(qwen_fourth_pairs)==4 and len(qwen_fourth_errors)==0 and all(float(x)==0.0 for x in qwen_fourth_action[qwen_fourth_action.method=='provenance_value'].action_flip_rate))
        add('local Qwen fourth fresh CUDA expansion', 'verified' if qwen_fourth_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in qwen_fourth_paths), f"valid_rows={qwen_fourth_meta.get('valid_rows')}; paired_episodes={qwen_fourth_meta.get('paired_episodes')}; errors={qwen_fourth_meta.get('raw_error_rows')}; candidate_retry=True")
    else:
        add('local Qwen fourth fresh CUDA expansion', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in qwen_fourth_paths))
    heldout_dir=ROOT/'results_submission/report/local_qwen_heldout174_176'
    heldout_paths=[heldout_dir/'metadata.json', heldout_dir/'rows.csv', heldout_dir/'summary.csv', heldout_dir/'paired.csv', heldout_dir/'bootstrap.csv', heldout_dir/'action_consistency.csv', heldout_dir/'action_bootstrap.csv', heldout_dir/'curve.png']
    if all(p.exists() for p in heldout_paths):
        heldout_meta=json.loads((heldout_dir/'metadata.json').read_text()); heldout_rows=pd.read_csv(heldout_dir/'rows.csv'); heldout_pairs=pd.read_csv(heldout_dir/'paired.csv'); heldout_boot=pd.read_csv(heldout_dir/'bootstrap.csv'); heldout_action=pd.read_csv(heldout_dir/'action_consistency.csv')
        heldout_ok=(heldout_meta.get('valid_rows')==12 and heldout_meta.get('raw_error_rows')==0 and heldout_meta.get('paired_episodes')==3 and heldout_meta.get('task_families')==['find-plant'] and heldout_meta.get('duplications')==[1,4] and heldout_meta.get('methods')==['flat_value','provenance_value'] and heldout_meta.get('baseline_method')=='flat_value' and heldout_meta.get('episode_success_claim') is False and len(heldout_rows)==12 and len(heldout_pairs)==3 and len(heldout_boot)>0 and set(heldout_action.method)=={'flat_value','provenance_value'})
        add('local Qwen independent held-out null expansion', 'verified' if heldout_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in heldout_paths), f"valid_rows={heldout_meta.get('valid_rows')}; paired_episodes={heldout_meta.get('paired_episodes')}; errors={heldout_meta.get('raw_error_rows')}; baseline={heldout_meta.get('baseline_method')}")
    else:
        add('local Qwen independent held-out null expansion', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in heldout_paths))
    heldout_animal_dir=ROOT/'results_submission/report/local_qwen_heldout177_179'
    heldout_animal_paths=[heldout_animal_dir/'metadata.json', heldout_animal_dir/'rows.csv', heldout_animal_dir/'summary.csv', heldout_animal_dir/'paired.csv', heldout_animal_dir/'bootstrap.csv', heldout_animal_dir/'action_consistency.csv', heldout_animal_dir/'action_bootstrap.csv', heldout_animal_dir/'curve.png']
    if all(p.exists() for p in heldout_animal_paths):
        heldout_animal_meta=json.loads((heldout_animal_dir/'metadata.json').read_text()); heldout_animal_rows=pd.read_csv(heldout_animal_dir/'rows.csv'); heldout_animal_pairs=pd.read_csv(heldout_animal_dir/'paired.csv'); heldout_animal_boot=pd.read_csv(heldout_animal_dir/'bootstrap.csv'); heldout_animal_action=pd.read_csv(heldout_animal_dir/'action_consistency.csv')
        heldout_animal_ok=(heldout_animal_meta.get('valid_rows')==12 and heldout_animal_meta.get('raw_error_rows')==0 and heldout_animal_meta.get('paired_episodes')==3 and heldout_animal_meta.get('task_families')==['find-animal'] and heldout_animal_meta.get('duplications')==[1,4] and heldout_animal_meta.get('methods')==['flat_value','provenance_value'] and heldout_animal_meta.get('baseline_method')=='flat_value' and heldout_animal_meta.get('episode_success_claim') is False and len(heldout_animal_rows)==12 and len(heldout_animal_pairs)==3 and len(heldout_animal_boot)>0 and set(heldout_animal_action.method)=={'flat_value','provenance_value'})
        add('local Qwen independent held-out animal null expansion', 'verified' if heldout_animal_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in heldout_animal_paths), f"valid_rows={heldout_animal_meta.get('valid_rows')}; paired_episodes={heldout_animal_meta.get('paired_episodes')}; errors={heldout_animal_meta.get('raw_error_rows')}; baseline={heldout_animal_meta.get('baseline_method')}")
    else:
        add('local Qwen independent held-out animal null expansion', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in heldout_animal_paths))
    probe_dir=ROOT/'results_submission/qwen_semantic_bank_probe'
    probe_paths=[probe_dir/'protocol.json', probe_dir/'states.json', probe_dir/'results.json', probe_dir/'rows.csv', probe_dir/'summary.csv', probe_dir/'ANALYSIS.md']
    if all(p.exists() for p in probe_paths):
        probe_meta=json.loads((probe_dir/'protocol.json').read_text()); probe_rows=pd.read_csv(probe_dir/'rows.csv'); probe_summary=pd.read_csv(probe_dir/'summary.csv')
        compact=probe_summary[probe_summary.condition=='archived_compact'].iloc[0]; semantic=probe_summary[probe_summary.condition=='semantic'].iloc[0]
        probe_ok=(probe_meta.get('calls')==12 and len(probe_rows)==24 and int(semantic.valid)==8 and int(compact.valid)==12 and float(compact.constant_value_rate)==1.0 and float(semantic.constant_value_rate)==.375 and float(compact.example_values_rate)==1.0 and float(semantic.example_values_rate)==0.0 and probe_meta.get('episode_success_claim') is False)
        add('local Qwen semantic bank prompt intervention', 'verified' if probe_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in probe_paths), f"planned={probe_meta.get('calls')}; valid_semantic={int(semantic.valid)}; compact_constant={compact.constant_value_rate}; semantic_constant={semantic.constant_value_rate}; semantic_template={semantic.example_values_rate}")
    else:
        add('local Qwen semantic bank prompt intervention', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in probe_paths))
    heterogeneity_dir=ROOT/'results_submission/report/local_qwen_effect_heterogeneity'
    heterogeneity_paths=[heterogeneity_dir/'metadata.json', heterogeneity_dir/'heterogeneity.csv', heterogeneity_dir/'forest.png']
    if all(p.exists() for p in heterogeneity_paths):
        heterogeneity_meta=json.loads((heterogeneity_dir/'metadata.json').read_text()); heterogeneity=pd.read_csv(heterogeneity_dir/'heterogeneity.csv')
        heterogeneity_ok=(heterogeneity_meta.get('rows')==48 and heterogeneity_meta.get('reward_rows')==16 and heterogeneity_meta.get('paired_unit')=='episode variation' and heterogeneity_meta.get('pooled') is False and len(heterogeneity)==48 and set(heterogeneity.metric)=={'reward','success','repeated_no_visible_change'} and set(heterogeneity.duplication.astype(int))=={1,4})
        add('local Qwen effect heterogeneity', 'verified' if heterogeneity_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in heterogeneity_paths), f"rows={heterogeneity_meta.get('rows')}; reward_rows={heterogeneity_meta.get('reward_rows')}; pooled={heterogeneity_meta.get('pooled')}")
    else:
        add('local Qwen effect heterogeneity', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in heterogeneity_paths))
    cross_model_dir=ROOT/'results_submission/report/cross_model_interactive_table'
    cross_model_paths=[cross_model_dir/'metadata.json', cross_model_dir/'strata.csv', cross_model_dir/'ranges.csv']
    if all(p.exists() for p in cross_model_paths):
        cross_model_meta=json.loads((cross_model_dir/'metadata.json').read_text()); cross_model_strata=pd.read_csv(cross_model_dir/'strata.csv'); cross_model_ranges=pd.read_csv(cross_model_dir/'ranges.csv')
        cross_model_ok=(cross_model_meta.get('pooled') is False and set(cross_model_meta.get('models',[]))=={'DeepSeek','Qwen2.5-Coder-3B'} and cross_model_meta.get('rows')==104 and cross_model_meta.get('deepseek_rows')==60 and cross_model_meta.get('qwen_rows')==44 and cross_model_meta.get('alfworld_rows')==32 and len(cross_model_strata)==104 and len(cross_model_ranges)>0)
        add('cross-model interactive stratified table', 'verified' if cross_model_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in cross_model_paths), f"rows={cross_model_meta.get('rows')}; pooled={cross_model_meta.get('pooled')}; models={','.join(cross_model_meta.get('models',[]))}")
    else:
        add('cross-model interactive stratified table', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in cross_model_paths))
    probe_path=Path('results_submission/report/local_qwen_gpu_probe.json')
    if file_ok(probe_path):
        probe=json.loads((ROOT/probe_path).read_text())
        probe_ok=probe.get('status')=='verified' and probe.get('cuda_available') is True and probe.get('inference_device')=='cuda:0' and probe.get('generated_tokens',0)>=1
        add('local Qwen CUDA inference probe', 'verified' if probe_ok else 'partial', str(probe_path), f"device={probe.get('inference_device')}; gpu={probe.get('gpu_name')}; generated_tokens={probe.get('generated_tokens')}")
    else:
        add('local Qwen CUDA inference probe', 'missing', str(probe_path))
    flat1=float(sw[(sw.duplication==1)&(sw.method=='flat')].reward.iloc[0]); prov1=float(sw[(sw.duplication==1)&(sw.method=='provenance')].reward.iloc[0])
    flat4=float(sw[(sw.duplication==4)&(sw.method=='flat')].reward.iloc[0]); prov4=float(sw[(sw.duplication==4)&(sw.method=='provenance')].reward.iloc[0])
    add('ScienceWorld paired traces', 'verified' if file_ok(Path('results_submission/report/scienceworld_v7_method_bootstrap.csv')) else 'missing', 'results_submission/report/scienceworld_v7_method_bootstrap.csv', f'reward contrast dup1={prov1-flat1:.3f}; dup4={prov4-flat4:.3f}')
    expand_paths=[Path('results_submission/report/scienceworld_expand_summary.csv'), Path('results_submission/report/scienceworld_expand_method_bootstrap.csv'), Path('results_submission/report/scienceworld_expand_action_bootstrap.csv'), Path('results_submission/report/scienceworld_expand_metadata.json')]
    if all(file_ok(p) for p in expand_paths):
        em=json.loads((ROOT/expand_paths[-1]).read_text()); es=pd.read_csv(ROOT/expand_paths[0]); ea=pd.read_csv(ROOT/expand_paths[2])
        expand_ok=(em.get('episodes')==72 and em.get('valid_episodes')==66 and em.get('error_episodes')==6 and set(em.get('duplications',[]))=={1,4} and set(em.get('tasks',[]))=={'find-plant','find-animal'} and set(ea['method'])=={'flat','no_imagination','provenance'})
        add('ScienceWorld fixed-grid expansion', 'verified' if expand_ok else 'partial', ';'.join(str(p) for p in expand_paths), f"episodes={em.get('episodes')}; valid={em.get('valid_episodes')}; errors={em.get('error_episodes')}")
    else:
        add('ScienceWorld fixed-grid expansion', 'missing', ';'.join(str(p) for p in expand_paths))
    value_paths=[Path('results_submission/report/scienceworld_expand_value_summary.csv'), Path('results_submission/report/scienceworld_expand_value_paired.csv'), Path('results_submission/report/scienceworld_expand_value_metadata.json')]
    if all(file_ok(p) for p in value_paths):
        vm=json.loads((ROOT/value_paths[-1]).read_text()); vs=pd.read_csv(ROOT/value_paths[0]); value_ok=(vm.get('episodes')==46 and set(vm.get('methods',[]))=={'flat','provenance_value'} and set(vm.get('duplications',[]))=={1,4} and set(vm.get('tasks',[]))=={'find-plant','find-animal'})
        add('ScienceWorld fixed-grid grouped-value null check', 'verified' if value_ok else 'partial', ';'.join(str(p) for p in value_paths), f"episodes={vm.get('episodes')}; readout={vm.get('readout')}")
    else:
        add('ScienceWorld fixed-grid grouped-value null check', 'missing', ';'.join(str(p) for p in value_paths))
    success_paths=[Path('results_submission/report/scienceworld_expand_success_summary.csv'), Path('results_submission/report/scienceworld_expand_success_paired.csv'), Path('results_submission/report/scienceworld_expand_success_metadata.json')]
    if all(file_ok(p) for p in success_paths):
        sm=json.loads((ROOT/success_paths[-1]).read_text()); ss=pd.read_csv(ROOT/success_paths[0]); success_ok=(sm.get('valid_rows')==46 and set(sm.get('methods',[]))=={'flat','provenance_success'} and set(sm.get('duplications',[]))=={1,4} and sm.get('matched_rows_per_duplication')==11)
        add('ScienceWorld fixed-grid success-aware provenance ablation', 'verified' if success_ok else 'partial', ';'.join(str(p) for p in success_paths), f"valid_rows={sm.get('valid_rows')}; matched={sm.get('matched_rows_per_duplication')}")
    else:
        add('ScienceWorld fixed-grid success-aware provenance ablation', 'missing', ';'.join(str(p) for p in success_paths))
    extension_paths=[Path('results_submission/scienceworld_expand2_m1/summaries.json'), Path('results_submission/scienceworld_expand2_success_m4/summaries.json')]
    if all(file_ok(p) for p in extension_paths):
        extension_rows=[]
        for p in extension_paths:
            extension_rows.extend(json.loads((ROOT/p).read_text()))
        extension_errors=[r for r in extension_rows if r.get('error')]
        extension_valid=[r for r in extension_rows if not r.get('error')]
        add('ScienceWorld contiguous extension', 'partial', ';'.join(str(p) for p in extension_paths), f"valid_rows={len(extension_valid)}; retained_api_errors={len(extension_errors)}; excluded_from_claims=True")
    else:
        add('ScienceWorld contiguous extension', 'missing', ';'.join(str(p) for p in extension_paths))
    refreshed_paths=[Path('results_submission/report/scienceworld_recharged_summary.csv'), Path('results_submission/report/scienceworld_recharged_paired.csv'), Path('results_submission/report/scienceworld_recharged_action_consistency.csv'), Path('results_submission/report/scienceworld_recharged_action_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_metadata.json'), Path('results_submission/report/scienceworld_recharged_curves.png')]
    if all(file_ok(p) for p in refreshed_paths):
        rm=json.loads((ROOT/refreshed_paths[4]).read_text()); rs=pd.read_csv(ROOT/refreshed_paths[0]); rp=pd.read_csv(ROOT/refreshed_paths[1])
        refreshed_ok=(rm.get('api_probe_status')==200 and rm.get('valid_rows')==60 and set(rm.get('duplications',[]))=={1,4} and set(rm.get('variations',[]))=={150,151,152,153,154,155} and {'flat','provenance','provenance_success','provenance_value','no_imagination'}.issubset(set(rs['experiment_method'])) and len(rp)>0)
        add('ScienceWorld refreshed DeepSeek comparison', 'verified' if refreshed_ok else 'partial', ';'.join(str(p) for p in refreshed_paths), f"valid_rows={rm.get('valid_rows')}; api_status={rm.get('api_probe_status')}; broad_success_claim=False")
    else:
        add('ScienceWorld refreshed DeepSeek comparison', 'missing', ';'.join(str(p) for p in refreshed_paths))
    refreshed_cf=Path('results_submission/report/scienceworld_recharged_counterfactual/summary.csv')
    if file_ok(refreshed_cf):
        rcf=pd.read_csv(ROOT/refreshed_cf)
        grouped=rcf[rcf['method']=='provenance_grouped']; flat4=rcf[(rcf['method']=='flat')&(rcf['duplication']==4)]
        cf_ok=(len(rcf)==18 and len(grouped)==6 and len(flat4)==6 and grouped['success'].astype(int).sum()==6 and flat4['success'].astype(int).sum()<6)
        add('ScienceWorld refreshed state-replay counterfactual', 'verified' if cf_ok else 'partial', str(refreshed_cf), f"rows={len(rcf)}; grouped_success={grouped.success.mean() if len(grouped) else 0:.3f}; flat_m4_success={flat4.success.mean() if len(flat4) else 0:.3f}")
    else:
        add('ScienceWorld refreshed state-replay counterfactual', 'missing', str(refreshed_cf))
    animal_paths=[Path('results_submission/report/scienceworld_recharged_animal/scienceworld_recharged_summary.csv'), Path('results_submission/report/scienceworld_recharged_animal/scienceworld_recharged_metadata.json'), Path('results_submission/report/scienceworld_recharged_animal/scienceworld_recharged_action_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_animal/scienceworld_recharged_curves.png')]
    if all(file_ok(p) for p in animal_paths):
        am=json.loads((ROOT/animal_paths[1]).read_text()); ar=pd.read_csv(ROOT/animal_paths[0])
        animal_ok=(am.get('task')=='find-animal' and am.get('api_probe_status')==200 and am.get('valid_rows')==36 and set(am.get('variations',[]))=={150,151,152,153,154,155} and {'flat','provenance','no_imagination'}.issubset(set(ar['experiment_method'])))
        add('ScienceWorld refreshed second-task-family check', 'verified' if animal_ok else 'partial', ';'.join(str(p) for p in animal_paths), f"valid_rows={am.get('valid_rows')}; transient_retry_archived=True; broad_success_claim=False")
    else:
        add('ScienceWorld refreshed second-task-family check', 'missing', ';'.join(str(p) for p in animal_paths))
    comparison_path=Path('results_submission/report/scienceworld_recharged_comparison_all.csv')
    if file_ok(comparison_path):
        comparison=pd.read_csv(ROOT/comparison_path)
        comparison_cols={'task_family','duplication','experiment_method','success','reward','root_confidence_first','root_confidence_last','root_confidence_delta','action_flip_rate_m1_vs_m4'}
        comparison_ok=(comparison_cols.issubset(comparison.columns) and
                       set(comparison['task_family'])=={'find-plant','find-animal'} and
                       set(comparison['duplication'].astype(int))=={1,4} and len(comparison)>=16)
        add('ScienceWorld refreshed comparison table', 'verified' if comparison_ok else 'partial', str(comparison_path), f"rows={len(comparison)}; metrics=confidence/action_flip/reward")
    else:
        add('ScienceWorld refreshed comparison table', 'missing', str(comparison_path))
    replay_paths=[Path('results_submission/report/scienceworld_recharged_counterfactual_all/summary.csv'), Path('results_submission/report/scienceworld_recharged_counterfactual_all/paired.csv'), Path('results_submission/report/scienceworld_recharged_counterfactual_all/bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_counterfactual_all/metadata.json')]
    if all(file_ok(p) for p in replay_paths):
        replay_meta=json.loads((ROOT/replay_paths[-1]).read_text()); replay=pd.read_csv(ROOT/replay_paths[0]); replay_bootstrap=pd.read_csv(ROOT/replay_paths[2])
        replay_ok=(replay_meta.get('states')==12 and replay_meta.get('rows')==36 and replay_meta.get('new_api_calls')==0 and set(replay.task)=={'find-plant','find-animal'} and len(replay_bootstrap)==8)
        add('ScienceWorld expanded fresh-state replay', 'verified' if replay_ok else 'partial', ';'.join(str(p) for p in replay_paths), f"states={replay_meta.get('states')}; rows={replay_meta.get('rows')}; new_api_calls={replay_meta.get('new_api_calls')}")
    else:
        add('ScienceWorld expanded fresh-state replay', 'missing', ';'.join(str(p) for p in replay_paths))
    extension_paths=[Path('results_submission/report/scienceworld_recharged_extension/summary.csv'), Path('results_submission/report/scienceworld_recharged_extension/paired.csv'), Path('results_submission/report/scienceworld_recharged_extension/paired_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_extension/action_consistency.csv'), Path('results_submission/report/scienceworld_recharged_extension/action_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_extension/metadata.json')]
    if all(file_ok(p) for p in extension_paths):
        extension_meta=json.loads((ROOT/extension_paths[-1]).read_text()); extension=pd.read_csv(ROOT/extension_paths[0]); extension_bootstrap=pd.read_csv(ROOT/extension_paths[2]); extension_action=pd.read_csv(ROOT/extension_paths[4])
        extension_ok=(extension_meta.get('api_probe_status')==200 and extension_meta.get('valid_rows')==24 and extension_meta.get('error_rows')==0 and set(extension.duplication.astype(int))=={1,4} and set(extension.experiment_method)=={'flat','provenance_success'} and len(extension_bootstrap)==8 and set(extension_action.method)=={'flat','provenance_success'})
        add('ScienceWorld fresh-key extension', 'verified' if extension_ok else 'partial', ';'.join(str(p) for p in extension_paths), f"valid_rows={extension_meta.get('valid_rows')}; errors={extension_meta.get('error_rows')}; api_status={extension_meta.get('api_probe_status')}")
    else:
        add('ScienceWorld fresh-key extension', 'missing', ';'.join(str(p) for p in extension_paths))
    animal_extension_paths=[Path('results_submission/report/scienceworld_recharged_animal_extension/summary.csv'), Path('results_submission/report/scienceworld_recharged_animal_extension/paired.csv'), Path('results_submission/report/scienceworld_recharged_animal_extension/paired_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_animal_extension/action_consistency.csv'), Path('results_submission/report/scienceworld_recharged_animal_extension/action_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_animal_extension/metadata.json')]
    if all(file_ok(p) for p in animal_extension_paths):
        animal_extension_meta=json.loads((ROOT/animal_extension_paths[-1]).read_text()); animal_extension=pd.read_csv(ROOT/animal_extension_paths[0]); animal_extension_bootstrap=pd.read_csv(ROOT/animal_extension_paths[2])
        animal_extension_ok=(animal_extension_meta.get('api_probe_status')==200 and animal_extension_meta.get('task')=='find-animal' and animal_extension_meta.get('valid_rows')==24 and animal_extension_meta.get('error_rows')==0 and set(animal_extension.duplication.astype(int))=={1,4} and set(animal_extension.experiment_method)=={'flat','provenance_success'} and len(animal_extension_bootstrap)==8)
        add('ScienceWorld fresh-key second-task extension', 'verified' if animal_extension_ok else 'partial', ';'.join(str(p) for p in animal_extension_paths), f"valid_rows={animal_extension_meta.get('valid_rows')}; errors={animal_extension_meta.get('error_rows')}; api_status={animal_extension_meta.get('api_probe_status')}")
    else:
        add('ScienceWorld fresh-key second-task extension', 'missing', ';'.join(str(p) for p in animal_extension_paths))
    llm_extension_paths=[Path('results_submission/report/scienceworld_recharged_llm_extension/summary.csv'), Path('results_submission/report/scienceworld_recharged_llm_extension/rows.csv'), Path('results_submission/report/scienceworld_recharged_llm_extension/confidence.csv'), Path('results_submission/report/scienceworld_recharged_llm_extension/paired.csv'), Path('results_submission/report/scienceworld_recharged_llm_extension/paired_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_llm_extension/action_consistency.csv'), Path('results_submission/report/scienceworld_recharged_llm_extension/action_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_llm_extension/metadata.json')]
    if all(file_ok(p) for p in llm_extension_paths):
        llm_extension_meta=json.loads((ROOT/llm_extension_paths[-1]).read_text()); llm_extension=pd.read_csv(ROOT/llm_extension_paths[0]); llm_extension_bootstrap=pd.read_csv(ROOT/llm_extension_paths[4]); llm_extension_action=pd.read_csv(ROOT/llm_extension_paths[6])
        llm_extension_ok=(llm_extension_meta.get('api_probe_status')==200 and llm_extension_meta.get('task')=='find-plant' and llm_extension_meta.get('valid_rows')==24 and llm_extension_meta.get('error_rows')==0 and set(llm_extension.duplication.astype(int))=={1,4} and set(llm_extension.experiment_method)=={'flat','provenance'} and len(llm_extension_bootstrap)==10 and set(llm_extension_action.method)=={'flat','provenance'})
        add('ScienceWorld fresh-key fair LLM extension', 'verified' if llm_extension_ok else 'partial', ';'.join(str(p) for p in llm_extension_paths), f"valid_rows={llm_extension_meta.get('valid_rows')}; errors={llm_extension_meta.get('error_rows')}; api_status={llm_extension_meta.get('api_probe_status')}")
    else:
        add('ScienceWorld fresh-key fair LLM extension', 'missing', ';'.join(str(p) for p in llm_extension_paths))
    animal_llm_extension_paths=[Path('results_submission/report/scienceworld_recharged_animal_llm_extension/summary.csv'), Path('results_submission/report/scienceworld_recharged_animal_llm_extension/rows.csv'), Path('results_submission/report/scienceworld_recharged_animal_llm_extension/confidence.csv'), Path('results_submission/report/scienceworld_recharged_animal_llm_extension/paired.csv'), Path('results_submission/report/scienceworld_recharged_animal_llm_extension/paired_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_animal_llm_extension/action_consistency.csv'), Path('results_submission/report/scienceworld_recharged_animal_llm_extension/action_bootstrap.csv'), Path('results_submission/report/scienceworld_recharged_animal_llm_extension/metadata.json')]
    if all(file_ok(p) for p in animal_llm_extension_paths):
        animal_llm_extension_meta=json.loads((ROOT/animal_llm_extension_paths[-1]).read_text()); animal_llm_extension=pd.read_csv(ROOT/animal_llm_extension_paths[0]); animal_llm_extension_bootstrap=pd.read_csv(ROOT/animal_llm_extension_paths[4]); animal_llm_extension_action=pd.read_csv(ROOT/animal_llm_extension_paths[6])
        animal_llm_extension_ok=(animal_llm_extension_meta.get('api_probe_status')==200 and animal_llm_extension_meta.get('task')=='find-animal' and animal_llm_extension_meta.get('valid_rows')==24 and animal_llm_extension_meta.get('error_rows')==0 and set(animal_llm_extension.duplication.astype(int))=={1,4} and set(animal_llm_extension.experiment_method)=={'flat','provenance'} and len(animal_llm_extension_bootstrap)==10 and set(animal_llm_extension_action.method)=={'flat','provenance'})
        add('ScienceWorld fresh-key fair LLM second-task extension', 'verified' if animal_llm_extension_ok else 'partial', ';'.join(str(p) for p in animal_llm_extension_paths), f"valid_rows={animal_llm_extension_meta.get('valid_rows')}; errors={animal_llm_extension_meta.get('error_rows')}; api_status={animal_llm_extension_meta.get('api_probe_status')}")
    else:
        add('ScienceWorld fresh-key fair LLM second-task extension', 'missing', ';'.join(str(p) for p in animal_llm_extension_paths))
    aggregate_paths=[Path('results_submission/report/scienceworld_interactive_aggregate/summary.csv'), Path('results_submission/report/scienceworld_interactive_aggregate/episodes.csv'), Path('results_submission/report/scienceworld_interactive_aggregate/confidence.csv'), Path('results_submission/report/scienceworld_interactive_aggregate/action_consistency.csv'), Path('results_submission/report/scienceworld_interactive_aggregate/paired.csv'), Path('results_submission/report/scienceworld_interactive_aggregate/paired_bootstrap.csv'), Path('results_submission/report/scienceworld_interactive_aggregate/metadata.json')]
    if all(file_ok(p) for p in aggregate_paths):
        aggregate_meta=json.loads((ROOT/aggregate_paths[-1]).read_text()); aggregate_summary=pd.read_csv(ROOT/aggregate_paths[0]); aggregate_bootstrap=pd.read_csv(ROOT/aggregate_paths[5])
        aggregate_ok=(aggregate_meta.get('api_calls')==0 and aggregate_meta.get('valid_rows')==96 and aggregate_meta.get('error_rows')==0 and set(aggregate_meta.get('protocols',[]))=={'refreshed_llm','fresh_key_success'} and set(aggregate_meta.get('task_families',[]))=={'find-plant','find-animal'} and {'protocol','task_family','duplication','experiment_method','success','reward','root_confidence_delta'}.issubset(aggregate_summary.columns) and len(aggregate_bootstrap)>0)
        add('ScienceWorld consolidated interactive ledger', 'verified' if aggregate_ok else 'partial', ';'.join(str(p) for p in aggregate_paths), f"valid_rows={aggregate_meta.get('valid_rows')}; protocols={','.join(aggregate_meta.get('protocols',[]))}; api_calls={aggregate_meta.get('api_calls')}")
    else:
        add('ScienceWorld consolidated interactive ledger', 'missing', ';'.join(str(p) for p in aggregate_paths))
    interactive_table_paths=[Path('results_submission/report/interactive_main_table/summary.csv'), Path('results_submission/report/interactive_main_table/contrasts.csv'), Path('results_submission/report/interactive_main_table/metadata.json')]
    if all(file_ok(p) for p in interactive_table_paths):
        interactive_table_meta=json.loads((ROOT/interactive_table_paths[-1]).read_text()); interactive_table=pd.read_csv(ROOT/interactive_table_paths[0]); interactive_contrasts=pd.read_csv(ROOT/interactive_table_paths[1])
        interactive_table_ok=(interactive_table_meta.get('summary_rows')==32 and interactive_table_meta.get('contrast_rows')==56 and set(interactive_table_meta.get('protocols',[]))=={'refreshed_llm','fresh_key_success','fresh_key_llm'} and {'task_family','duplication','experiment_method','success','reward','root_confidence_delta','action_flip_rate'}.issubset(interactive_table.columns) and len(interactive_contrasts)==56)
        add('ScienceWorld interactive main table', 'verified' if interactive_table_ok else 'partial', ';'.join(str(p) for p in interactive_table_paths), f"summary_rows={interactive_table_meta.get('summary_rows')}; contrast_rows={interactive_table_meta.get('contrast_rows')}; protocols={','.join(interactive_table_meta.get('protocols',[]))}")
    else:
        add('ScienceWorld interactive main table', 'missing', ';'.join(str(p) for p in interactive_table_paths))
    cf_path=ROOT/'results_submission/report/scienceworld_interactive_counterfactual.csv'
    if cf_path.exists():
        cf=pd.read_csv(cf_path)
        grouped=cf[cf['method']=='provenance_grouped']
        flat=cf[cf['method']=='flat']
        cf_ok=(len(grouped)>=2 and len(flat)==3*len(grouped) and
               grouped['success'].astype(int).sum()==len(grouped) and
               flat['success'].astype(int).sum()<len(flat))
        add('real-environment duplication counterfactual', 'partial', 'results_submission/report/scienceworld_interactive_counterfactual.csv;results_submission/report/scienceworld_input_integrity/summary.json', 'legacy trace-derived replay retained for auditability; serialized-input audit found post-action mutable history, so clean causal claim is withdrawn')
    else:
        add('real-environment duplication counterfactual', 'missing', str(cf_path))
    value_paths=[Path('results_submission/report/scienceworld_value_summary.csv'), Path('results_submission/report/scienceworld_value_paired.csv'), Path('results_submission/report/scienceworld_value_curve.png')]
    if all(file_ok(p) for p in value_paths):
        value=pd.read_csv(ROOT/value_paths[0]); paired=pd.read_csv(ROOT/value_paths[1])
        value_ok=set(value['duplication'].astype(int))=={1,4} and set(value['method'])=={'flat','provenance_value'} and len(paired)>=4
        add('grouped-value interactive exploratory run', 'verified' if value_ok else 'partial', ';'.join(str(p) for p in value_paths), 'four find-plant variations; grouped-value readout')
    else:
        add('grouped-value interactive exploratory run', 'missing', ';'.join(str(p) for p in value_paths))
    animal_paths=[Path('results_submission/report/scienceworld_value_animal_summary.csv'), Path('results_submission/report/scienceworld_value_animal_paired.csv'), Path('results_submission/report/scienceworld_value_animal_curve.png')]
    if all(file_ok(p) for p in animal_paths):
        animal=pd.read_csv(ROOT/animal_paths[0])
        animal_ok=set(animal['method'])=={'flat','provenance_value'} and set(animal['duplication'].astype(int))=={1,4} and len(animal)==4
        add('second-task-family grouped-value check', 'verified' if animal_ok else 'partial', ';'.join(str(p) for p in animal_paths), 'one find-animal variation; null control')
    else:
        add('second-task-family grouped-value check', 'missing', ';'.join(str(p) for p in animal_paths))
    alf_dir=ROOT/'results_submission/report/alfworld_interactive'
    alf_paths=[alf_dir/'metadata.json', alf_dir/'rows.csv', alf_dir/'summary.csv', alf_dir/'paired.csv', alf_dir/'paired_algorithmic.csv', alf_dir/'paired_objectives.csv', alf_dir/'action_consistency.csv', alf_dir/'bootstrap.csv', alf_dir/'action_bootstrap.csv', alf_dir/'curve.png']
    if all(p.exists() for p in alf_paths):
        alf_meta=json.loads((alf_dir/'metadata.json').read_text())
        alf_rows=pd.read_csv(alf_dir/'rows.csv')
        alf_summary=pd.read_csv(alf_dir/'summary.csv')
        alf_pairs=pd.read_csv(alf_dir/'paired.csv')
        alf_actions=pd.read_csv(alf_dir/'action_consistency.csv')
        alf_objectives=pd.read_csv(alf_dir/'paired_objectives.csv')
        alf_ok=(alf_meta.get('valid_rows',0)>=22 and len(alf_rows)==alf_meta.get('valid_rows') and len(alf_summary)>0 and len(alf_pairs)>=8 and len(alf_objectives)>=len(alf_pairs) and len(alf_actions)>=8 and
                alf_meta.get('uses_hidden_state_in_prompt') is False and alf_meta.get('uses_gold_plan') is False and
                set(alf_rows.method).issuperset({'flat','flat_value','provenance_value','provenance_success'}) and alf_rows.errors.astype(str).eq('[]').all() and
                alf_meta.get('qwen_cuda_inference',{}).get('device')=='cuda:0')
        add('ALFWorld TextWorld paired interactive comparison', 'verified' if alf_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in alf_paths), f"valid_rows={len(alf_rows)}; models={','.join(sorted(alf_rows.model.unique()))}; broad_success_claim=False")
        algo_path=alf_dir/'paired_algorithmic.csv'
        if algo_path.exists():
            algo=pd.read_csv(algo_path)
            algo_ok=(len(algo)>=6 and set(algo.duplication.astype(int))=={1,4} and
                     {'provenance_value_minus_flat_value_success','provenance_value_minus_flat_value_reward','provenance_value_minus_flat_value_mean_belief_drift'}.issubset(algo.columns) and
                     (algo['provenance_value_minus_flat_value_success']==0).all() and
                     (algo['provenance_value_minus_flat_value_reward']==0).all())
            add('ALFWorld fair algorithmic flat-value control', 'verified' if algo_ok else 'partial', str(algo_path.relative_to(ROOT)), f"paired_rows={len(algo)}; reward_success_null={algo_ok}; flat_value_rows={alf_meta.get('flat_value_valid_rows','?')}")
        else:
            add('ALFWorld fair algorithmic flat-value control', 'missing', str(algo_path.relative_to(ROOT)))
    else:
        add('ALFWorld TextWorld paired interactive comparison', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in alf_paths))
    add('broad interactive success gain', 'not_established', 'results_submission/report/scienceworld_v7_dev8_summary.csv;results_submission/report/scienceworld_expand_summary.csv', 'primary v7 flat/provenance success tied at 0.5; fixed-grid expansion has mixed task-family changes')
    add('fixed snapshot state replay', 'verified' if file_ok(Path('results_submission/scienceworld_snapshot_audit_v5/snapshot_summary.csv')) else 'missing', 'results_submission/scienceworld_snapshot_audit_v5/snapshot_summary.csv')
    clean_dir=ROOT/'results_submission/scienceworld_frozen_readout_clean_20260929'
    clean_paths=[clean_dir/'manifest.json',clean_dir/'states.json',clean_dir/'summary.csv',clean_dir/'paired.csv',clean_dir/'bootstrap.csv',clean_dir/'audit.json']
    if all(p.exists() for p in clean_paths):
        clean_meta=json.loads((clean_dir/'audit.json').read_text()); clean_summary=pd.read_csv(clean_dir/'summary.csv'); clean_pairs=pd.read_csv(clean_dir/'paired.csv')
        clean_ok=(clean_meta.get('planned_api_calls')==180 and clean_meta.get('valid_readouts')==180 and clean_meta.get('errors')==0 and clean_meta.get('states')==12 and clean_meta.get('candidate_replays')==48 and clean_meta.get('all_replays_match_public_state') is True and clean_meta.get('identical_input_request_blocks')==36 and {'task','method','duplication','root_p_true','action_flip_from_m1','immediate_reward'}.issubset(clean_summary.columns) and len(clean_pairs)==144)
        add('ScienceWorld clean fixed-bank readout intervention', 'verified' if clean_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in clean_paths), f"valid_readouts={clean_meta.get('valid_readouts')}; states={clean_meta.get('states')}; candidate_replays={clean_meta.get('candidate_replays')}; identical_input_blocks={clean_meta.get('identical_input_request_blocks')}; episode_success_claim={clean_meta.get('episode_success_claim')}")
    else:
        add('ScienceWorld clean fixed-bank readout intervention', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in clean_paths))
    expanded_dir=ROOT/'results_submission/scienceworld_frozen_readout_first4_20260929'
    expanded_paths=[expanded_dir/'manifest.json', expanded_dir/'states.json', expanded_dir/'summary.csv', expanded_dir/'paired.csv', expanded_dir/'bootstrap.csv', expanded_dir/'audit.json', expanded_dir/'curve.png']
    if all(p.exists() for p in expanded_paths):
        expanded_meta=json.loads((expanded_dir/'audit.json').read_text()); expanded_summary=pd.read_csv(expanded_dir/'summary.csv'); expanded_pairs=pd.read_csv(expanded_dir/'paired.csv'); expanded_bootstrap=pd.read_csv(expanded_dir/'bootstrap.csv')
        expanded_ok=(expanded_meta.get('planned_api_calls')==1440 and expanded_meta.get('valid_readouts')==1440 and expanded_meta.get('errors')==0 and expanded_meta.get('states')==96 and expanded_meta.get('candidate_replays')==384 and expanded_meta.get('all_replays_match_public_state') is True and expanded_meta.get('episode_success_claim') is False and expanded_meta.get('root_calibration_claim') is False and {'flat','same_prompt_dedup','provenance_value','provenance_success'}.issubset(set(expanded_summary.method)) and len(expanded_pairs)==1152 and len(expanded_bootstrap)>0)
        add('ScienceWorld expanded four-step fixed-bank intervention', 'verified' if expanded_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in expanded_paths), f"valid_readouts={expanded_meta.get('valid_readouts')}; states={expanded_meta.get('states')}; candidate_replays={expanded_meta.get('candidate_replays')}; errors={expanded_meta.get('errors')}; episode_success_claim={expanded_meta.get('episode_success_claim')}")
    else:
        add('ScienceWorld expanded four-step fixed-bank intervention', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in expanded_paths))
    analysis_paths=[expanded_dir/'ANALYSIS.md', expanded_dir/'episode_bootstrap.csv', expanded_dir/'episode_units.csv', expanded_dir/'paired_events.csv', expanded_dir/'joint_events.csv', expanded_dir/'verified_analysis.json', expanded_dir/'episode_curves.png', expanded_dir/'qualitative_events.md']
    if all(p.exists() for p in analysis_paths):
        analysis_meta=json.loads((expanded_dir/'verified_analysis.json').read_text()); episode_bootstrap=pd.read_csv(expanded_dir/'episode_bootstrap.csv'); events=pd.read_csv(expanded_dir/'paired_events.csv')
        analysis_ok=(analysis_meta.get('raw_inputs_verified') is True and analysis_meta.get('resampling_unit')=='task/variation episode' and analysis_meta.get('bootstrap_draws')==20000 and analysis_meta.get('requests_verified')==1440 and analysis_meta.get('candidate_replays_verified')==384 and analysis_meta.get('joint_change_events')==0 and analysis_meta.get('confidence_mediated_action_claim') is False and len(episode_bootstrap)>0 and len(events)==1152)
        add('ScienceWorld fixed-bank episode-cluster analysis', 'verified' if analysis_ok else 'partial', ';'.join(str(p.relative_to(ROOT)) for p in analysis_paths), f"episodes={analysis_meta.get('episodes')}; paired_comparisons={analysis_meta.get('paired_nonbaseline_comparisons')}; joint_changes={analysis_meta.get('joint_change_events')}; confidence_mediated_claim={analysis_meta.get('confidence_mediated_action_claim')}")
    else:
        add('ScienceWorld fixed-bank episode-cluster analysis', 'missing', ';'.join(str(p.relative_to(ROOT)) for p in analysis_paths))
    integrity=ROOT/'results_submission/report/scienceworld_input_integrity/summary.json'
    if integrity.exists():
        integrity_meta=json.loads(integrity.read_text()); clean_scope=integrity_meta.get('scopes',{}).get('scienceworld_frozen_readout_clean_20260929',{})
        expanded_scope=integrity_meta.get('scopes',{}).get('scienceworld_frozen_readout_first4_20260929',{})
        integrity_ok=clean_scope.get('requests')==180 and clean_scope.get('current_or_future_outcome')==0 and expanded_scope.get('requests')==1440 and expanded_scope.get('current_or_future_outcome')==0
        add('ScienceWorld serialized-input integrity audit', 'verified' if integrity_ok else 'partial', str(integrity.relative_to(ROOT)), f"clean_requests={clean_scope.get('requests')}; clean_future_outcomes={clean_scope.get('current_or_future_outcome')}; expanded_requests={expanded_scope.get('requests')}; expanded_future_outcomes={expanded_scope.get('current_or_future_outcome')}; legacy_trace_rows_with_post_action_history={integrity_meta.get('trace_rows_with_post_action_history')}")
    else:
        add('ScienceWorld serialized-input integrity audit', 'missing', str(integrity.relative_to(ROOT)))
    paper_files=['paper/draft.md','paper/experiments_draft.md','paper/submission_checklist.md','paper/tables.md','paper/figure_manifest.md','results_submission/report/REPORT.md','results_submission/report/main_table.csv','results_submission/report/main_table_ci.csv','results_submission/report/ablation_endpoint.csv','results_submission/report/qualitative_examples.md']
    add('paper outputs', 'verified' if all(file_ok(Path(p)) for p in paper_files) else 'missing', ';'.join(paper_files))
    manifest_paths=[Path('results_submission/reproducibility_manifest.json'), Path('paper/reproducibility.md')]
    if all(file_ok(p) for p in manifest_paths):
        manifest=json.loads((ROOT/manifest_paths[0]).read_text())
        manifest_ok=(len(manifest.get('hashes',[]))>=40 and not manifest.get('missing') and len(manifest.get('commands',[]))>=8 and manifest.get('generated_by')=='submission.reproducibility_manifest')
        add('reproducibility manifest', 'verified' if manifest_ok else 'partial', ';'.join(str(p) for p in manifest_paths), f"hashes={len(manifest.get('hashes',[]))}; missing={len(manifest.get('missing',[]))}; commands={len(manifest.get('commands',[]))}")
    else:
        add('reproducibility manifest', 'missing', ';'.join(str(p) for p in manifest_paths))
    add('reviewer evidence matrix', 'verified' if file_ok(Path('paper/evidence_matrix.md')) else 'missing', 'paper/evidence_matrix.md')
    secret_hits=[]
    for p in [ROOT/'submission',ROOT/'paper']:
        for f in p.rglob('*'):
            if f.is_file() and f.suffix in {'.py','.md','.json','.csv'}:
                try:
                    if re.search(r'sk-[A-Za-z0-9]{20,}',f.read_text(errors='ignore')): secret_hits.append(str(f))
                except OSError: pass
    add('secret hygiene', 'verified' if not secret_hits else 'failed', 'submission/ and paper/', str(secret_hits))
    payload={'generated_by':'submission.evidence_audit','checks':checks,'verified_count':sum(c['status']=='verified' for c in checks),'partial_count':sum(c['status']=='partial' for c in checks),'not_established_count':sum(c['status']=='not_established' for c in checks)}
    OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
