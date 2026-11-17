"""Reproduce historical Scheme B plots from pinned original CSVs, not new training."""
import csv,json,hashlib,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'reports/source_manifest.json').read_text())
for item in manifest['original_run_logs']:
    for name,metadata in item['files'].items():
        file=R/'archive_logs'/f'B_z{item["dimension"]}_{item["mode"]}'/name
        if hashlib.sha256(file.read_bytes()).hexdigest()!=metadata['sha256']:raise ValueError('Original log copy changed')
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(name,rows):
 with (R/'reports'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'svg.fonttype':'none'})
rows=[];curves={};maximum_identity=0;maximum_objective=0
for d in [2,4,8,16,32]:
 for mode in ['loss_only','loss_plus_xi']:
  folder=R/'archive_logs'/f'B_z{d}_{mode}';args=json.loads((folder/'resolved_args.json').read_text());tr=read(folder/'train_history.csv');pr=[r for r in read(folder/'probe_history.csv') if r['task']=='theta'];final=json.loads((folder/'probe_summary.json').read_text());curves[(d,mode)]=(tr,pr)
  assert {r['task'] for r in read(folder/'probe_history.csv')}=={'theta','mean'}
  assert args['online_probe_tasks']==['theta'] and args['pw_baseline_mode']=='state_conditioned'
  assert int(tr[-1]['step'])==3000 and [int(x) for x in args['controlled_stage_steps']]==[1000]*3
  x=np.array([float(r['xi_pw']) for r in tr]);lm=np.array([float(r['lambda_max_batch']) for r in tr]);edge=np.array([float(r['lambda_edge']) for r in tr]);rec=np.array([float(r['reconstruction']) for r in tr]);kl=np.array([float(r['kl']) for r in tr]);obj=np.array([float(r['objective']) for r in tr])
  error=float(np.max(np.abs(x-np.maximum(lm-edge,0))));err_obj=float(np.max(np.abs(obj-rec-.001*kl-(.001*x if mode=='loss_plus_xi' else 0))));maximum_identity=max(maximum_identity,error);maximum_objective=max(maximum_objective,err_obj)
  assert error<1e-4 and err_obj<1e-5
  pre1=next(float(r['accuracy']) for r in pr if int(r['step'])==900);post1=next(float(r['accuracy']) for r in pr if int(r['step'])==1100);pre2=next(float(r['accuracy']) for r in pr if int(r['step'])==1900);post2=next(float(r['accuracy']) for r in pr if int(r['step'])==2100)
  rows.append(dict(latent=d,mode=mode,seed=0,last_training_step=3000,final_reconstruction=rec[-1],final_KL=kl[-1],final_Xi=x[-1],max_Xi=float(max(x)),MP_edge=float(edge[-1]),max_spike_rank=max(int(r['spike_rank']) for r in tr),final_theta_from_separate_probe=final['theta_accuracy'],online_theta_step900=pre1,online_theta_step1100=post1,online_theta_step1900=pre2,online_theta_step2100=post2,online_theta_last=float(pr[-1]['accuracy']),online_last_step=int(pr[-1]['step']),Xi_formula_max_error=error,objective_identity_max_error=err_obj,source='archived_original_logs_not_fresh_training'))
# Faithful screenshot replot for z2/z4, plus loss-only control in lower row.
fig,axes=plt.subplots(3,2,figsize=(15,9),sharex=True,gridspec_kw={'height_ratios':[1.5,.7,1]})
for col,d in enumerate([2,4]):
 tr,pr=curves[(d,'loss_plus_xi')];t=[int(r['step']) for r in tr]
 ax=axes[0,col];right=ax.twinx();ax.plot(t,[float(r['reconstruction']) for r in tr],color='#2474b5',label='reconstruction');right.plot(t,[float(r['xi_pw']) for r in tr],color='#e32f27',label='Xi');ax.set_ylabel('Reconstruction MSE',color='#2474b5');right.set_ylabel('Xi (raw MP excess)',color='#e32f27');ax.set_title(f'Scheme B | controlled | z={d} | loss_plus_xi | archived seed0')
 ax.legend(loc='upper left',fontsize=8);right.legend(loc='upper right',fontsize=8)
 axes[1,col].plot(t,[int(r['spike_rank']) for r in tr],color='#9467bd',lw=1.2);axes[1,col].set_ylabel('spike_rank k')
 axes[2,col].plot([int(r['step']) for r in pr],[float(r['accuracy']) for r in pr],color='#2a9d45',marker='.',label='theta, loss_plus_xi')
 _,control=curves[(d,'loss_only')];axes[2,col].plot([int(r['step']) for r in control],[float(r['accuracy']) for r in control],color='#777777',ls='--',label='theta, loss_only control');axes[2,col].set_ylim(0,1.02);axes[2,col].set_ylabel('Current-stage theta probe accuracy');axes[2,col].legend(fontsize=8);axes[2,col].set_xlabel('Training step')
 for ax in axes[:,col]:
  ax.axvline(1000,color='black',ls=':',lw=.7);ax.axvline(2000,color='black',ls=':',lw=.7);ax.set_xlim(0,3000);ax.grid(alpha=.15)
fig.suptitle('Original-log replot: reconstruction, legacy Xi and changing-stage theta probe\nBlue/red/purple/green correspond to the supplied report. Gray theta control is added for context.\nThis figure is not a new training reproduction and the probe does not evaluate fixed S1 examples after a stage switch.',fontsize=11)
fig.tight_layout(rect=(0,0,1,.90));fig.savefig(R/'figures/original_log_z2_z4.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(3,5,figsize=(19,10),sharex=True)
for col,d in enumerate([2,4,8,16,32]):
 for mode,color,style in [('loss_only','#777777','--'),('loss_plus_xi','#245bb2','-')]:
  tr,pr=curves[(d,mode)];t=[int(r['step']) for r in tr];axes[0,col].plot(t,[float(r['reconstruction']) for r in tr],color=color,ls=style,label=mode);axes[1,col].plot(t,[float(r['xi_pw']) for r in tr],color=color,ls=style,label=mode);axes[2,col].plot([int(r['step']) for r in pr],[float(r['accuracy']) for r in pr],color=color,ls=style,label=mode)
 axes[0,col].set_title(f'latent z={d}');axes[0,col].set_ylabel('Reconstruction MSE');axes[1,col].set_ylabel('Legacy Xi');axes[2,col].set_ylabel('Current-stage theta accuracy');axes[2,col].set_ylim(0,1.02);axes[2,col].set_xlabel('Training step')
 for ax in axes[:,col]:ax.set_xlim(0,3000);ax.axvline(1000,color='black',ls=':',lw=.6);ax.axvline(2000,color='black',ls=':',lw=.6);ax.grid(alpha=.15);ax.legend(fontsize=6)
fig.suptitle('Scheme B, all five dimensions and both objectives | original archived logs | one seed per condition\nReplot only: no training was performed to create this figure. Current-stage probe data/split change over time.',fontsize=12);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(R/'figures/original_log_all_dimensions.png',dpi=170);plt.close(fig)
write('archived_metrics.csv',rows)
audit=dict(status='archived_log_replot_complete',runs=10,training_steps_per_run=3000,seeds=[0],Xi_identity='max(lambda_max_batch - lambda_edge,0)',max_Xi_identity_error=maximum_identity,max_objective_identity_error=maximum_objective,legacy_MP_edge=4,updated_source_MP_edge=(1+math.sqrt(64/63))**2,updated_trace_normalized_raw_lambda_upper_bound=64,updated_edge_readout_upper_bound=64/((1+math.sqrt(64/63))**2)-1,archived_max_Xi=max(r['max_Xi'] for r in rows),requested_online_tasks=['theta'],probe_evaluation_source='current stage loader.dataset, first2048, fresh random80/20 split at each snapshot',fresh_exact_training_status='not_started_matching_original_source_not_located',current_source_math_differs_from_report=True)
(R/'reports/archive_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(audit,ensure_ascii=False,indent=2))
