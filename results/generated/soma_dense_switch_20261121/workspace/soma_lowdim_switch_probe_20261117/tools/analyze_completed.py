"""Read-only final analysis; all seeds, both cuts, no retuning or new training."""
from pathlib import Path
import csv,json,sys
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from study_paths import PLAN,OLD,sha,write,rows,verify
import plots_switch as p
OUT=ROOT/'reports/final_analysis_20261119';OUT.mkdir(exist_ok=True)
def save(name,rr):
    if not rr:return
    with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def average(points,tt):
    d=dict(points)
    if not all(t in d for t in tt):raise ValueError('Missing genuine timepoint '+str(tt))
    return float(np.mean([d[t] for t in tt]))
def agg(rr,keys,values):
    groups=defaultdict(list)
    for r in rr:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,group in sorted(groups.items()):
        d=dict(zip(keys,key));d['n']=len(group)
        for value in values:
            v=[float(r[value]) for r in group];d[value+'_mean']=float(np.mean(v));d[value+'_min']=min(v);d[value+'_max']=max(v)
        out.append(d)
    return out
def home(c):
    root=ROOT/'results'/PLAN['run_id'] if c['summary'].get('new_lowdim') else OLD/'results/all_3000_001'
    return root/f'{c["dataset"]}_z{c["dim"]}_seed{c["seed"]}'
def main():
    verify();live=json.loads((ROOT/'reports/LIVE_PROGRESS.json').read_text());assert live['status']=='complete' and not live['errors']
    cases=p.load();assert len(cases)==60 and len(p.FROZEN)==45
    final=[];effects=[];signals=[];validation=[];diagnostics=[];route=[];coverage=[];control=[];factor=[];statuses=defaultdict(int)
    for c in cases:
        base=dict(dataset=c['dataset'],dim=c['dim'],seed=c['seed'])
        if c['dataset']=='controlled':
            values=p.get(c,'probe');end=average(values,[3000]);before=max(t for t,y in values if t<=1000)
            control.append(dict(**base,theta_final_accuracy=end,actual_pre_cut_point=before,theta_before_first_cut=dict(values)[before],metric='ordinary current-stage theta accuracy; native training-pool 80/20, not independent fixed-S1 forgetting'))
            continue
        for rd in ['mu','sampled_z']:
            fixed=dict(p.get(c,'probe',readout=rd));current=dict(p.get(c,'probe',readout=rd,kind='current'));stable=dict(p.get(c,'probe',b='stable',readout=rd));frozen=dict(p.get(c,'probe',readout=rd,kind='frozen'));fstable=dict(p.get(c,'probe',b='stable',readout=rd,kind='frozen'))
            tasks=sorted({r['task'] for r in c['branches']['changing']['probes'] if r['probe_group']=='S1'});ks={r['task']:int(r['classes']) for r in c['branches']['changing']['probes'] if r['probe_group']=='S1'}
            chance=float(np.mean([1/ks[t] for t in tasks]))
            for task in tasks:
                a=dict(p.get(c,'probe',readout=rd,task=task));b=dict(p.get(c,'probe',b='stable',readout=rd,task=task))
                factor.append(dict(**base,readout=rd,task=task,classes=ks[task],chance=1/ks[task],changing_1000=a[1000],changing_3000=a[3000],stable_3000=b[3000],final_paired_pp=100*(a[3000]-b[3000])))
            final.append(dict(**base,readout=rd,chance=chance,S1_refit_1000=fixed[1000],S1_refit_3000=fixed[3000],current_refit_3000=current[3000],stable_refit_3000=stable[3000],S1_frozen_900=frozen[900],S1_frozen_1000=frozen[1000],S1_frozen_3000=frozen[3000],stable_frozen_3000=fstable[3000],refit_final_changing_minus_stable_pp=100*(fixed[3000]-stable[3000]),frozen_final_changing_minus_stable_pp=100*(frozen[3000]-fstable[3000]),refit_temporal_change_pp=100*(fixed[3000]-fixed[1000]),frozen_pre_cut_drop_pp=100*(frozen[1000]-frozen[900])))
            for cut,pre,post in [(1000,[900,950,1000],[1001,1050,1100]),(2000,[1900,1950,2000],[2001,2050,2100])]:
                for kind in ['fixed','frozen']:
                    a=p.get(c,'probe',readout=rd,kind=kind);b=p.get(c,'probe',b='stable',readout=rd,kind=kind);da=100*(average(a,post)-average(a,pre));db=100*(average(b,post)-average(b,pre))
                    effects.append(dict(**base,readout=rd,kind=kind,cut=cut,changing_delta_pp=da,stable_delta_pp=db,paired_increment_pp=da-db,all_three_pre_and_post_points=True))
        for cut,pre,post in [(1000,[900,950,1000],[1001,1050,1100]),(2000,[1900,1950,2000],[2001,2050,2100])]:
            for metric in ['ref0','ref1000']:
                ca=p.get(c,metric);st=p.get(c,metric,b='stable');onset=cut+1
                d=dict(**base,cut=cut,metric=metric,changing_onset=dict(ca)[onset],stable_onset=dict(st)[onset],onset_changing_minus_stable=dict(ca)[onset]-dict(st)[onset],post_mean_changing=average(ca,post),post_mean_stable=average(st,post),post_changing_minus_stable=average(ca,post)-average(st,post))
                d['paired_prepost_increment']=None if metric=='ref1000' and cut==1000 else d['post_changing_minus_stable']-(average(ca,pre)-average(st,pre));signals.append(d)
            leg=p.read(home(c)/'changing/legacy_every_batch.csv');window=[r for r in leg if cut+1<=int(r['step'])<=cut+25];eligible=[r for r in window if p.old.truth(r['eligible'])]
            route.append(dict(**base,cut=cut,window_start=cut+1,window_end=cut+25,fallback_batches=sum(r['baseline_source'].startswith('fallback:') for r in window),own_stage_batches=sum(r['baseline_source'].startswith('state:') for r in window),xi_peak_first25=max(float(r['xi']) for r in eligible),xi_onset=float(window[0]['xi']),first_own_stage_step=min(int(r['step']) for r in window if r['baseline_source'].startswith('state:'))))
        for branch in ['changing','stable']:
            rr=p.read(home(c)/branch/'validation.csv');v=next(r for r in rr if int(r['step'])==3000 and int(r['stage'])==1)
            validation.append(dict(**base,branch=branch,stage=1,step=3000,pixel_mse=float(v['pixel_mse']),reconstruction=float(v['reconstruction']),kl=float(v['kl'])))
            di=p.read(home(c)/branch/'diagnostics.csv');tr=c['branches'][branch]['trajectory'];m=max(tr,key=lambda r:float(r['pixel_mse']))
            diagnostics.append(dict(**base,branch=branch,max_clip=max(float(r['logvar_clipped_fraction']) for r in di),max_pixel_mse=float(m['pixel_mse']),max_pixel_mse_step=int(m['step']),final_mu_var=float(di[-1]['posterior_mu_variance']),final_noise_var=float(di[-1]['posterior_noise_variance'])))
            for ref,records in c['branches'][branch]['signals'].items():
                for record in records:statuses[record['status']]+=1
        for age in [0,1000]:
            f=home(c)/f'calibration_ref{age}.json';v=json.loads(p.pin(f).read_text())
            coverage.append(dict(**base,reference_updates=age,windows=v['edge_cover']['windows'],exceedances=v['edge_cover']['exceedances'],rate=v['edge_cover']['exceedances']/v['edge_cover']['windows'],reference_sha256=v['reference_sha256']))
    measures=['S1_refit_1000','S1_refit_3000','current_refit_3000','stable_refit_3000','S1_frozen_900','S1_frozen_1000','S1_frozen_3000','stable_frozen_3000','refit_final_changing_minus_stable_pp','frozen_final_changing_minus_stable_pp','refit_temporal_change_pp','frozen_pre_cut_drop_pp']
    save('final_by_seed.csv',final);save('final_by_dimension.csv',agg(final,['dataset','dim','readout','chance'],measures));save('transition_macro_by_seed.csv',effects);save('transition_macro_by_dimension.csv',agg(effects,['dataset','dim','readout','kind','cut'],['changing_delta_pp','stable_delta_pp','paired_increment_pp']))
    save('signals_by_seed.csv',signals);save('signals_by_dimension.csv',agg(signals,['dataset','dim','cut','metric'],['changing_onset','stable_onset','onset_changing_minus_stable','post_changing_minus_stable']))
    save('S1_validation_by_seed.csv',validation);save('S1_validation_by_dimension.csv',agg(validation,['dataset','dim','branch'],['pixel_mse','reconstruction','kl']));save('numerical_diagnostics.csv',diagnostics);save('legacy_stage_routing.csv',route);save('reference_coverage.csv',coverage);save('controlled_by_seed.csv',control);save('controlled_by_dimension.csv',agg(control,['dim'],['theta_final_accuracy']))
    save('S1_factor_by_seed.csv',factor);save('S1_factor_by_dimension.csv',agg(factor,['dataset','dim','readout','task','classes','chance'],['changing_1000','changing_3000','stable_3000','final_paired_pp']))
    for file,h in p.INPUTS.items():
        if sha(file)!=h:raise ValueError('Completed raw source changed')
    write(OUT/'input_manifest.json',dict(analysis_script_sha256=sha(Path(__file__)),files=p.INPUTS,training_updates_launched=0,final_image_probe_updates_completed=2999,validation_updates_completed=3000))
    write(OUT/'analysis_receipt.json',dict(status='complete',conditions=60,image_conditions=45,new_conditions=18,new_updates=90000,readouts=2,cuts=[1000,2000],image_seeds=3,signal_status_counts=dict(statuses),window_policy='original pre/post 3 genuine points, no result-dependent window selection',same_model_support='see original GALLERY; current pool changes are not temporal forgetting',exploratory=True,significance_claim=False,new_training_launched=False))
    print(json.dumps(dict(status='complete',conditions=len(cases),probe_rows=len(final),macro_transition_rows=len(effects),signal_rows=len(signals),diagnostic_rows=len(diagnostics))))
if __name__=='__main__':main()
