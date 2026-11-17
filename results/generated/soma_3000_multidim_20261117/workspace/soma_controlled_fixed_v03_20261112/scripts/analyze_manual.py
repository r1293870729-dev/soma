"""Controlled fixed-weight signal development; no method selection or final inference."""
from pathlib import Path
import argparse,csv,json,statistics,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.b_weight_sensitivity.formal_common import load_config,planned_units,unit_id
from experiments.manual_v03.signal import FixedPsi,SignalConfig
from scripts.smoke_manual import replay_run

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def validate_fixed_weights(rows):
    # The training logger records the actual float32 tensor coefficient exactly.
    expected_kl=float(np.float32(.001))
    if any(float(w['reconstruction'])!=1. or float(w['kl'])!=expected_kl for w in rows):
        raise ValueError('Human fixed weights changed')

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-id',required=True);p.add_argument('--analysis-id');a=p.parse_args()
    if not a.experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in a.experiment_id):raise ValueError('Unsafe experiment ID')
    source=ROOT/'results/generated'/a.experiment_id
    ledger=json.loads((source/'ledger.json').read_text());config=load_config()
    if ledger['status']!='complete' or ledger['completed']!=len(planned_units(config)):raise ValueError('Incomplete planned matrix; preserve failure denominator')
    analysis_id=a.analysis_id or a.experiment_id
    if not analysis_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in analysis_id):raise ValueError('Unsafe analysis ID')
    out=ROOT/'reports/controlled'/analysis_id;out.mkdir(parents=True,exist_ok=False)
    summaries=[];stages=[];replays=[];scales=[]
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for run in sorted(ledger['runs'],key=lambda r:r['seed']):
        path=Path(run['run_dir']);error=replay_run(path)
        metrics=read_csv(path/'metrics.csv');signals=[json.loads(s) for s in (path/'psi_signal.jsonl').read_text().splitlines()]
        if len(metrics)!=3000 or len(signals)!=3000 or any(int(x['step'])!=y['step'] or x['checkpoint_id']!=y['checkpoint_id'] for x,y in zip(metrics,signals)):raise ValueError('Signal/pre-update model alignment failed')
        weights=read_csv(path/'weights.csv')
        validate_fixed_weights(weights)
        cal=json.loads((path/'calibration.json').read_text());cover=cal['edge_cover']
        summaries.append(dict(seed=run['seed'],arm=run['arm'],updates=run['updates'],validation_reconstruction_diagnostic=run['validation_reconstruction'],wall_seconds=run['wall_seconds'],edge=cover['rows'][0]['edge'],edge_cover_windows=cover['windows'],edge_cover_exceedances=cover['exceedances'],edge_cover_fraction=cover['exceedance_fraction'],independent_generators=1,scope='development-only; nested seeds'))
        replays.append(dict(seed=run['seed'],rows=len(signals),maximum_error=error))
        for index in range(3):
            rows=signals[index*1000:(index+1)*1000];eligible=[r for r in rows if r['eligible']]
            stages.append(dict(seed=run['seed'],historical_stage_index=index+1,rows=len(rows),eligible_rows=len(eligible),low_energy_rows=sum(r['status']=='low-energy' for r in rows),mean_psi=statistics.mean(r['psi'] for r in eligible),maximum_psi=max(r['psi'] for r in eligible),mean_total_variance=statistics.mean(r['mean_total_variance'] for r in eligible),multi_mode_fraction=sum(r['active_modes']>=2 for r in eligible)/len(eligible),interpretation='descriptive historical stage; not frozen Q1 event AUROC'))
        x=np.arange(1,3001);label='seed '+str(run['seed'])
        values=[[float(r['reconstruction']) for r in metrics],[float(r['kl']) for r in metrics],[r['psi'] if r['psi'] is not None else np.nan for r in signals],[r.get('mean_total_variance',np.nan) for r in signals],[r['active_modes'] if r['active_modes'] is not None else np.nan for r in signals],[r['spectrum'][0] if r['spectrum'] else np.nan for r in signals]]
        for axis,ys in zip(axes.flat,values):axis.plot(x,ys,label=label,linewidth=.7,alpha=.8)
        with np.load(path/'fixed_reference.npz',allow_pickle=False) as z:reference=FixedPsi(z['mean'],z['inverse_sqrt'],float(z['edge']),SignalConfig(**cal['config']),cal)
        raw=np.load(path/'preupdate_readout_residuals.npy',mmap_mode='r',allow_pickle=False)
        for step in [64,1000,1064,2000,2064,3000]:
            window=np.asarray(raw[step-1],dtype=np.float64);base=reference.measure_window(window)
            for multiplier in [-2.,.25,4.]:
                changed=reference.measure_window(window*multiplier);eligible=base['status']!='low-energy' and changed['status']!='low-energy'
                delta=abs(base['psi']-changed['psi']) if eligible else None
                if eligible and delta>1e-10:raise ValueError('I4 numerical invariance fixture failed')
                scales.append(dict(seed=run['seed'],step=step,multiplier=multiplier,i4_scale_eligible=eligible,absolute_psi_error=delta,scope='engineering invariance check, not full B10 intervention experiment'))
    titles=['Raw reconstruction loss','Raw KL loss','Manual v0.3 Psi','Whitened mean total variance','Active modes above empirical edge','Largest normalized covariance eigenvalue']
    for axis,title in zip(axes.flat,titles):
        axis.set_title(title);axis.set_xlabel('model update');axis.grid(alpha=.2)
        for boundary in [1001,2001]:axis.axvline(boundary,color='gray',ls=':',lw=.8)
    axes[0,0].legend();fig.suptitle('Controlled development: rec=1, KL=0.001; one historical generator, three nested seeds')
    fig.tight_layout();fig.savefig(out/'controlled_signal_trajectories.png',dpi=150);plt.close(fig)
    write_csv(out/'run_summary.csv',summaries);write_csv(out/'stage_descriptive.csv',stages);write_csv(out/'signal_replay.csv',replays);write_csv(out/'scale_invariance_checks.csv',scales)
    result=dict(status='complete_controlled_development',runs=len(summaries),model_updates=sum(r['updates'] for r in summaries),fixed_weights={'reconstruction':1,'kl':.001},replay_maximum_error=max(r['maximum_error'] for r in replays),scale_invariance_maximum_error=max(r['absolute_psi_error'] for r in scales if r['i4_scale_eligible']),independent_generators=1,nested_seeds=[0,1,2],image_training_started=False,weight_search_performed=False,final_test_used=False,alarm_metrics='not-evaluated-no-sustained-alarm-contract',full_manual_38_tasks_complete=False)
    (out/'analysis_status.json').write_text(json.dumps(result,indent=2)+'\n')
    text='# Controlled 固定人工权重信号开发验证\n\n'
    text+='三个 seed 均使用 rec=1、KL=0.001，各 3000 步。未进行权重搜索或自动加权。\n\n'
    text+='逐步残差可独立重算全部 Psi；run_summary.csv 给出固定快照覆盖的描述性比例，不能代替持续在线虚警率。stage_descriptive.csv 仅按历史阶段描述，不把阶段边界自动当作通过新手册合同的结构事件。\n\n'
    text+='该历史生成器尚未通过新手册的独立多生成器、局部分裂区域、moment matching 与 separability 冻结。三 seed 嵌套于一个生成器，不增加生成器总体独立样本量。没有 Q1 最终 AUROC、检测/预警推断、主动编辑、CLEAR/CLOC 或最终 test。\n'
    (out/'RESULTS.md').write_text(text);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
