"""Historical circle/torus/local controlled protocol, all dimensions, no probe replacement."""
import argparse,csv,hashlib,json,sys,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from paths import ROOT,PLAN,HIST,ANCHOR,sha,write,read,verify
sys.path.insert(0,str(HIST/'reconstruction'))
from runtime import load_runner,case_args,digest
sys.path.insert(0,str(ROOT.parent/'soma_controlled_fixed_v03_20261112'))
from experiments.manual_v03.signal import FixedPsi,SignalConfig

def namespace(*parts):return int.from_bytes(hashlib.sha256('|'.join(['report-controlled-manual-reference']+[str(p) for p in parts]).encode()).digest()[:8],'little')%1000000000
def run(dim,seed):
    out=ROOT/'results'/PLAN['run_id']/f'controlled_z{dim}_seed{seed}';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();step=0;phase='prepare'
    try:
        if seed==0 and dim in [2,4]:
            source=ANCHOR/'results'/f'B_z{dim}';s=json.loads((source/'summary.json').read_text())
            if s['status']!='complete' or not s['original_primary_theta_preserved']:raise ValueError('Invalid existing exact replay')
            pinned={str(f.resolve()):sha(f) for f in source.rglob('*') if f.is_file() and f.suffix in ['.csv','.json','.jsonl']}
            write(out/'source_inputs.json',pinned)
            result=dict(s,dataset='controlled',latent_dim=dim,seed=seed,status='complete',source_dir=str(source),reuse=True,actual_new_updates=0,stage_steps=[1000]*3)
            write(out/'summary.json',result);write(out/'progress.json',result);return result
        # Reload native module so every case starts with original functions and no old observer closures.
        sys.modules.pop('restored_scheme_b_candidate',None);runner=load_runner();parent=HIST/'results/reconstructed_B_001'/f'B_z{dim}_loss_plus_xi';native=out/'native';native.mkdir();(native/'snapshots').mkdir()
        args,archived,extra=case_args(runner,dim,'loss_plus_xi',native);args.seed=seed
        if list(args.controlled_stage_steps)!=[1000]*3:raise ValueError('Historical schedule differs')
        write(out/'actual_args.json',dict(actual=runner.vars_for_json(args),archived=archived,only_treatment_override='seed; added observers do not enter loss'))
        pinned={str(f.resolve()):sha(f) for f in parent.rglob('*') if f.is_file() and (f.suffix=='.pt' or f.name in ['train_history.csv','probe_history.csv','execution.json'])} if seed==0 else {}
        write(out/'source_inputs.json',pinned);original_train={int(r['step']):r for r in read(parent/'train_history.csv')};original_probe={(int(r['step']),r['task']):r for r in read(parent/'probe_history.csv')};expected_steps=set(original_train)
        native_append=runner.append_csv;native_loss=runner.reconstruction_loss;native_probe=runner.run_online_controlled_probe;cal_rows=0
        def status():
            v=dict(status='running',unit=f'controlled_z{dim}_seed{seed}',phase=phase,step=step,calibration_updates=cal_rows,dimension=dim,seed=seed,wall_seconds=time.monotonic()-started,updated_at=time.time());write(out/'progress.json',v);print(json.dumps(v),flush=True)
        def cal_append(path,row):
            nonlocal cal_rows
            native_append(path,row)
            if Path(path).name.startswith('calibration_'):
                cal_rows+=1
                if cal_rows%100==0:status()
        runner.append_csv=cal_append;runner.set_seed(seed);phase='separate_calibration';status();cal=runner.run_calibration(args,torch.device('cpu'),native)
        model,stages,bounds=runner.build_experiment(args,torch.device('cpu'));initial={k:v.detach().clone() for k,v in model.state_dict().items()};initial_hash=digest(model)
        if seed==0 and initial_hash!=torch.load(parent/'initial.pt',map_location='cpu',weights_only=True)['model_sha256']:raise ValueError('Original initial model differs')
        torch.save(dict(model_state=initial,model_sha256=initial_hash),native/'initial.pt');embedding=runner.make_orthogonal_embedding(args.observation_dim,args.intrinsic_dim,seed+123);roles={};samples={};noises={}
        for role,n in [('white',512),('edge',2048),('edge-cover',2048)]:
            ns=namespace('data',seed,role);data=runner.ControlledPhaseDataset('s1',embedding,n,args.controlled_noise_std,ns,args.factor_bins,args.split_offset,args.local_probability,args.split_mode)
            samples[role]=torch.stack([data[i][0] for i in range(n)]);roles[role]=[f'legacy-controlled:embedding{seed+123}:synthetic-seed{ns}:row{i}' for i in range(n)]
            noises[role]=torch.from_numpy(np.random.default_rng(namespace('noise',seed,role)).normal(size=(n,max(4,dim))).astype(np.float32))[:,:dim]
        if len(set(namespace('data',seed,r) for r in roles))!=3 or any(namespace('data',seed,r) in [seed,seed+101,seed+202] for r in roles):raise ValueError('Calibration namespaces overlap')
        refs={};counts={0:0,1000:0};probe_checks=log_checks=snapshot_checks=0;captured1={}
        def calibrate(age):
            before=digest(model);rng=torch.get_rng_state().clone();was=model.training;model.eval();raw={}
            with torch.no_grad():
                for role,x in samples.items():
                    pieces=[]
                    for start in range(0,len(x),128):
                        batch=x[start:start+128];mu,lv=model.encode(batch);hat=model.decoder(mu+torch.exp(.5*lv)*noises[role][start:start+128]);pieces.append((batch-hat).numpy())
                    raw[role]=np.concatenate(pieces)
            model.train(was)
            if digest(model)!=before or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Calibration modified training model/RNG')
            signal=FixedPsi.fit(raw['white'],raw['edge'],SignalConfig(**PLAN['signal']),roles);folder=out/f'ref{age}';folder.mkdir();np.savez_compressed(folder/'calibration_residuals.npz',**raw);write(folder/'calibration.json',dict(reference_updates=age,reference_model_sha256=before,reference_sha256=signal.reference_hash(),edge=signal.edge,config=PLAN['signal'],roles=roles,coverage=signal.coverage(raw['edge-cover']),model_and_RNG_unchanged=True));refs[age]=signal
        phase='formal';calibrate(0)
        def loss(hat,batch):
            nonlocal step
            step+=1
            if step in expected_steps:
                raw=(batch-hat).detach().numpy()[-64:];pre=digest(model);rng=torch.get_rng_state().clone()
                for age,sig in refs.items():
                    if age==1000 and step<=1000:continue
                    r=dict(step=step,updates_completed=step-1,pre_model_sha256=pre,reference_updates=age,reference_sha256=sig.reference_hash(),raw_residual_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),**sig.measure_window(raw))
                    with (out/f'ref{age}/signals.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
                    counts[age]+=1
                if digest(model)!=pre or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Manual observer changed model/RNG')
            return native_loss(hat,batch)
        def observed_append(path,row):
            nonlocal probe_checks,log_checks
            native_append(path,row);name=Path(path).name
            if name=='train_history.csv':
                if seed==0:
                    old=original_train[int(row['step'])]
                    for k in ['objective','reconstruction','kl','xi_pw','lambda_max_batch','lambda_edge','spike_rank','grad_norm']:
                        if float(row[k])!=float(old[k]):raise ValueError(f'Original training changed at{step} {k}')
                    log_checks+=1
                if int(row['step'])==1:captured1.update({k:v.detach().clone() for k,v in model.state_dict().items()})
                if int(row['step'])==1000:calibrate(1000)
                if step%100==0 or step in [1,1001,2001]:status()
            elif name=='probe_history.csv' and seed==0:
                if float(row['accuracy'])!=float(original_probe[int(row['step']),row['task']]['accuracy']):raise ValueError('Original theta probe changed')
                probe_checks+=1
        def observed_probe(*a,**kw):
            nonlocal snapshot_checks
            result=native_probe(*a,**kw);t=kw['global_step'];h=digest(model)
            if seed==0 and h!=torch.load(parent/'snapshots'/f'step_{t:04d}.pt',map_location='cpu',weights_only=True)['model_sha256']:raise ValueError('Original checkpoint differs')
            torch.save(dict(model_state=model.state_dict(),model_sha256=h,step=t),native/'snapshots'/f'step_{t:04d}.pt');snapshot_checks+=1;return result
        runner.reconstruction_loss=loss;runner.append_csv=observed_append;runner.run_online_controlled_probe=observed_probe;runner.train_formal(args,torch.device('cpu'),native,model,stages,bounds,cal)
        final=digest(model);final_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if step!=3000 or snapshot_checks!=15:raise ValueError('Incomplete native3000-step run')
        if seed==0:
            last=torch.load(parent/'last.pt',map_location='cpu',weights_only=True)
            if any(not torch.equal(final_state[k],v) for k,v in last['model_state'].items()):raise ValueError('Original final model differs')
        torch.save(dict(model_state=final_state,model_sha256=final,calibration=cal),native/'last.pt');extended=[];phase='offline_probe_extensions';status()
        for t,state,index in [(0,initial,0),(1,captured1,0),(3000,final_state,2)]:
            model.load_state_dict(state)
            with torch.random.fork_rng(devices=[]):rows=native_probe(args=args,model=model,stage=stages[index],global_step=t,stage_index=index+1,step_in_stage=0 if t==0 else 1 if t==1 else 1000,device=torch.device('cpu'))
            for r in rows:
                if r['row_type']=='accuracy' and r['task']=='theta':extended.append(dict(step=t,accuracy=r['accuracy'],task='theta',view='original_online_rules_current_stage',origin='offline_extension_original_fitter_seed_rule'))
        with (out/'online_theta_extensions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(extended[0]));w.writeheader();w.writerows(extended)
        if any(sha(f)!=h for f,h in pinned.items()):raise ValueError('Original parent changed')
        verify();result=dict(status='complete',dataset='controlled',latent_dim=dim,dimension=dim,seed=seed,stage_steps=[1000]*3,source_dir=str(out),reuse=False,formal_updates=3000,separate_calibration_updates=cal_rows,initial_model_sha256=initial_hash,final_model_sha256=final,seed0_exact_replay=seed==0,original_training_logs_exact=log_checks,original_probe_rows_exact=probe_checks,checkpoint_count=snapshot_checks,manual_points=counts,manual_observer_changes_model=False,legacy_xi_enters_loss=True,original_primary_theta_preserved=True,wall_seconds=time.monotonic()-started)
        write(out/'summary.json',result);write(out/'progress.json',result);return result
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e),phase=phase,step=step,traceback=traceback.format_exc()));raise

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);verify();cases=[]
    for seed in PLAN['controlled_seeds']:
        for dim in PLAN['controlled_dimensions']:
            try:cases.append(run(dim,seed))
            except Exception as e:write(ROOT/'reports/worker_controlled.json',dict(status='failed',cases=cases,error=repr(e)));raise
            write(ROOT/'reports/worker_controlled.json',dict(status='running',cases=cases))
    write(ROOT/'reports/worker_controlled.json',dict(status='complete',cases=cases))
if __name__=='__main__':main()
