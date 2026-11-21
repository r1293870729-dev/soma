"""Unmodified historical runner, extra current-theta probes isolated from all RNGs."""
import argparse,copy,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from dense_common import ROOT,BASE,HIST,PLAN,selected,wanted,cases,pin,read,jread,write,append,verify,check_sources,compare,reference
sys.path.insert(0,str(HIST/'reconstruction'))
from runtime import load_runner,case_args,digest
sys.path.insert(0,str(BASE/'soma_controlled_fixed_v03_20261112'))
from experiments.manual_v03.signal import FixedPsi,SignalConfig

def run_case(case,meta):
    name=case.name;out=ROOT/'results'/name
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text());assert old['status']=='complete';check_sources(old['source_files']);return old
    out.mkdir(exist_ok=False);native=out/'native';native.mkdir();pins={};started=time.monotonic();step=0;phase='calibration';count=0
    try:
        jread(pins,case/'summary.json')
        dim,seed=meta['latent_dim'],meta['seed'];src=Path(meta['source_dir']);parent=HIST/'results/reconstructed_B_001'/f'B_z{dim}_loss_plus_xi'
        old_train={int(r['step']):r for r in read(pins,src/'native/train_history.csv')};old_probe={(int(r['step']),r['task']):r for r in read(pins,src/'native/probe_history.csv')}
        old_sig={age:{r['step']:r for r in [json.loads(v) for v in pin(pins,src/f'ref{age}/signals.jsonl').read_text().splitlines() if v]} for age in [0,1000]}
        sys.modules.pop('restored_scheme_b_candidate',None);runner=load_runner();args,_,_=case_args(runner,dim,'loss_plus_xi',native);args.seed=seed;args.quiet_console=True
        old_append=runner.append_csv;old_loss=runner.reconstruction_loss;old_native_probe=runner.run_online_controlled_probe
        def status():write(out/'progress.json',dict(status='running',case=name,dataset='controlled',phase=phase,step=step,replayed_updates=count,planned_updates=6000,wall_seconds=time.monotonic()-started,updated_at=time.time()))
        status();runner.set_seed(seed);cal=runner.run_calibration(args,torch.device('cpu'),native);count=3000
        expected_cal=jread(pins,src/'native/summary.json')['calibration'];compare(expected_cal,cal,list(expected_cal),0.)
        model,stages,bounds=runner.build_experiment(args,torch.device('cpu'));r0,m0=reference(src,0,pins)
        if digest(model)!=m0['reference_model_sha256']:raise ValueError('Initial controlled model differs')
        refs={0:r0};logged=probed=checked_snapshots=signal_checks=0;phase='formal_replay';args.log_every=1;status()
        def loss(hat,x):
            nonlocal step,signal_checks
            step+=1
            if selected(step):
                raw=(x-hat).detach().cpu().numpy()[-64:];h=digest(model)
                for age,s in refs.items():
                    row=dict(step=step,updates_completed=step-1,pre_model_sha256=h,reference_updates=age,reference_sha256=s.reference_hash(),**s.measure_window(raw))
                    e=old_sig[age].get(step)
                    if e is not None:compare(e,row,['psi'],1e-10);assert e['pre_model_sha256']==h;signal_checks+=1
                    with (out/f'psi_ref{age}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            return old_loss(hat,x)
        def native_probe(*a,**kw):
            nonlocal checked_snapshots
            result=old_native_probe(*a,**kw);t=kw['global_step'];snapshot=src/'native/snapshots'/f'step_{t:04d}.pt'
            if not snapshot.exists():snapshot=parent/'snapshots'/f'step_{t:04d}.pt'
            state=torch.load(pin(pins,snapshot),map_location='cpu',weights_only=True)
            if digest(model)!=state['model_sha256']:raise ValueError('Original controlled checkpoint differs '+str(t))
            checked_snapshots+=1;return result
        def observed(path,row):
            nonlocal logged,probed,count
            old_append(path,row);filename=Path(path).name
            if filename=='probe_history.csv':
                e=old_probe[int(row['step']),row['task']];compare(e,row,['accuracy'],0.);probed+=1
            if filename!='train_history.csv':return
            t=int(row['step']);count=3000+t
            if t!=step:raise ValueError('Unexpected extra main forward')
            if t in old_train:
                compare(old_train[t],row,['objective','reconstruction','kl','xi_pw','lambda_max_batch','lambda_edge','spike_rank','grad_norm'],0.);logged+=1
            if t==1000:
                s,m=reference(src,1000,pins)
                if digest(model)!=m['reference_model_sha256']:raise ValueError('Reference1000 controlled model differs')
                refs[1000]=s
            if selected(t):
                append(out/'dense_metrics.csv',[dict(step=t,stage=row['stage_index'],reconstruction=row['reconstruction'],pixel_mse=row['reconstruction'],xi=row['xi_pw'],eligible=bool(row['baseline_ready'] and row['window_ready']),kl=row['kl'])])
                before=digest(model);rng=torch.get_rng_state().clone();np_state=np.random.get_state();py_state=random.getstate()
                try:
                    with torch.random.fork_rng(devices=[]):result=old_native_probe(args=args,model=copy.deepcopy(model),stage=copy.deepcopy(stages[int(row['stage_index'])-1]),global_step=t,stage_index=int(row['stage_index']),step_in_stage=int(row['step_in_stage']),device=torch.device('cpu'))
                finally:np.random.set_state(np_state);random.setstate(py_state)
                if digest(model)!=before or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Extra controlled probe changed model/RNG')
                pr=[dict(step=t,updates_completed=t,stage=row['stage_index'],task=r['task'],accuracy=r['accuracy'],model_sha256=before,measurement_timing='post_update_current_stage') for r in result if r['row_type']=='accuracy' and r['task']!='mean'];append(out/'dense_probe.csv',pr)
            if t%100==0 or selected(t) and t%10==0:status()
        runner.append_csv=observed;runner.reconstruction_loss=loss;runner.run_online_controlled_probe=native_probe
        runner.train_formal(args,torch.device('cpu'),native,model,stages,bounds,cal)
        last=src/'native/last.pt'
        if not last.exists():last=parent/'last.pt'
        last_state=torch.load(pin(pins,last),map_location='cpu',weights_only=True)
        if any(not torch.equal(model.state_dict()[k],v) for k,v in last_state['model_state'].items()):raise ValueError('Original final controlled weights differ')
        if sorted({int(r['step']) for r in read({},out/'dense_probe.csv')})!=wanted():raise ValueError('Missing dense theta steps')
        if step!=3000 or logged!=len(old_train) or probed!=len(old_probe) or checked_snapshots!=15:raise ValueError('Incomplete controlled exact checks')
        check_sources(pins);verify();result=dict(status='complete',case=name,dataset='controlled',dim=dim,seed=seed,replayed_updates=6000,new_treatment_updates=0,original_log_rows_verified=logged,original_probe_rows_verified=probed,original_snapshots_verified=checked_snapshots,original_signal_rows_verified=signal_checks,final_model_exact=True,dense_steps=122,measurement_timing='post_update_current_stage',source_files=pins,wall_seconds=time.monotonic()-started)
        write(out/'summary.json',result);write(out/'progress.json',result);print('COMPLETE '+name,flush=True);return result
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e),step=step,phase=phase,traceback=traceback.format_exc()));raise
def main(first=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);verify();done=[]
    try:
        for case,meta in cases('controlled'):
            done.append(run_case(case,meta));write(ROOT/'reports/worker_controlled.json',dict(status='running',completed=len(done),planned=15,cases=done,updated_at=time.time()))
            if first:break
        write(ROOT/'reports/worker_controlled.json',dict(status='qualified' if first else 'complete',completed=len(done),planned=15,cases=done,updated_at=time.time()))
    except BaseException as e:write(ROOT/'reports/worker_controlled.json',dict(status='failed',completed=len(done),planned=15,error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--first',action='store_true');args=a.parse_args();main(args.first)
