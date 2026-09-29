#!/usr/bin/env python3
"""DESCRIPTIVE historical audit only; never overwrite imported assets.
Legacy labels: stage>=2 and local_step<=200 on logged rows. Not matched-FAR,
not event-level final inference. AP handles ties; -entropy is post-hoc audit.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path);a=ap.parse_args();root=a.root.resolve()
    out=(a.output or root/'reports/local/recomputed').resolve()
    if out==root/'results/imported' or root/'results/imported' in out.parents:raise SystemExit('Refuse imported output')
    out.mkdir(parents=True,exist_ok=True);metrics=[];runs=[];inputs=[]
    def record(p):
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        inputs.append({'path':str(p.relative_to(root)),'sha256':digest})
    for p in sorted((root/'results/imported/passive_xib').rglob('summary.json')):
        s=json.loads(p.read_text());h=p.parent/'train_history.csv'
        if not h.exists():continue
        record(h);df=pd.read_csv(h);y=((df.stage_index>=2)&(df.local_step<=200)).astype(int)
        row={'run_id':p.parent.name,'dataset':s['dataset'],'latent_dim':s['latent_dim'],'seed':s['seed'],
             'reported_status':s['status'],'rows':len(df),'last_step':int(df.step.max()),
             'done_exists':(p.parent/'DONE').exists(),'checkpoint_exists':(p.parent/'checkpoint.pt').exists()}
        runs.append(row)
        scores={'mse':df.reconstruction_mse,'psi_mp':df.xi_b,'entropy_raw':df.spectral_entropy,
                'negative_entropy_diagnostic':-df.spectral_entropy,'spike_rank':df.spike_rank,
                'effective_rank':df.effective_rank,'delta_objective':df.objective.diff().abs().fillna(0),
                'delta_kl':df.kl.diff().abs().fillna(0)}
        for name,v in scores.items():
            if not np.isfinite(v).all():raise ValueError(f'Nonfinite {h}: {name}')
            metrics.append({**{k:row[k] for k in ('run_id','dataset','latent_dim','seed')},'detector':name,
                'auroc':roc_auc_score(y,v),'average_precision':average_precision_score(y,v),
                'prevalence':float(y.mean()),'n_logged_rows':len(y)})
    if len(runs)!=30:raise SystemExit(f'Expected 30 archived runs, got {len(runs)}; full bundle/LFS needed')
    pd.DataFrame(runs).to_csv(out/'passive_runs.csv',index=False)
    m=pd.DataFrame(metrics);m.to_csv(out/'passive_recomputed.csv',index=False)
    m.groupby(['dataset','detector']).agg(n_runs=('run_id','size'),auroc_mean=('auroc','mean'),auroc_sd=('auroc','std'),ap_mean=('average_precision','mean'),prevalence=('prevalence','mean')).reset_index().to_csv(out/'passive_summary_descriptive.csv',index=False)
    active=[]
    for p in sorted((root/'results/imported/active_soma').rglob('structure_logs.json')):
        record(p);log=json.loads(p.read_text())
        ins=[r for r in log if r.get('trigger_ready') and r.get('reason') in ('accepted','rejected_all_candidates','no_proposals')]
        acc=[r for r in log if r.get('accepted_edit')]
        active.append({'method':p.parent.parent.name,'seed':p.parent.name,'checkpoints':len(log),'inspections':len(ins),
                       'accepted':len(acc),'inspection_steps':[r['step'] for r in ins],'accepted_steps':[r['step'] for r in acc]})
    (out/'active_counts.json').write_text(json.dumps(active,indent=2))
    iv=[]
    for p in sorted((root/'results/imported/interventions').rglob('i1_i4_intervention_summary.csv')):
        record(p);d=pd.read_csv(p)
        for name,g in d.groupby('intervention'):
            delta=g['delta_psi_vs_parent_computed']
            iv.append({'file':str(p.relative_to(root)),'intervention':name,'rows':len(g),'parent_runs':g.run_id.nunique(),
                       'delta_psi_mean':float(delta.mean()),'delta_psi_abs_max':float(delta.abs().max())})
    pd.DataFrame(iv).to_csv(out/'intervention_summary.csv',index=False)
    (out/'input_hashes.json').write_text(json.dumps(inputs,indent=2))
    print(json.dumps({'passive_runs':len(runs),'active_runs':len(active),'intervention_rows':sum(x['rows'] for x in iv),
                      'output':str(out),'inference':'descriptive-only'},ensure_ascii=False))
if __name__=='__main__':main()
