"""Unchanged native historical training replay with passive manual-v0.3 observers."""
import argparse,csv,json,sys,hashlib,time,traceback,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;HIST=BASE/'soma_scheme_ab_reproduction_20261116';P= json.loads((ROOT/'configs/plan.json').read_text())
sys.path.insert(0,str(HIST/'reconstruction'))
from runtime import load_runner,case_args,digest,sha,write
sys.path.insert(0,str(BASE/'soma_controlled_fixed_v03_20261112'))
from experiments.manual_v03.signal import FixedPsi,SignalConfig

def namespace(*parts):return int.from_bytes(hashlib.sha256('|'.join(['report-controlled-manual-reference']+[str(p) for p in parts]).encode()).digest()[:8],'little')%1000000000
def run(dim):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);runner=load_runner();parent=HIST/'results/reconstructed_B_001'/f'B_z{dim}_loss_plus_xi';out=ROOT/'results'/f'B_z{dim}';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();step=0
    try:
        pinned={str(f.relative_to(parent)):sha(f) for f in parent.rglob('*') if f.is_file() and (f.suffix=='.pt' or f.name in ['train_history.csv','probe_history.csv','execution.json'])};write(out/'parent_inputs.json',dict(parent=str(parent),files=pinned));args,_,_=case_args(runner,dim,'loss_plus_xi',out/'native');(out/'native').mkdir()
        original_train={int(r['step']):r for r in csv.DictReader((parent/'train_history.csv').open())};original_probe={(int(r['step']),r['task']):r for r in csv.DictReader((parent/'probe_history.csv').open())};expected_steps=set(original_train)
        runner.set_seed(args.seed);cal=runner.run_calibration(args,torch.device('cpu'),out/'native');model,stages,bounds=runner.build_experiment(args,torch.device('cpu'));initial=torch.load(parent/'initial.pt',map_location='cpu',weights_only=True)
        if digest(model)!=initial['model_sha256']:raise ValueError('Formal initial model does not match source')
        embedding=runner.make_orthogonal_embedding(args.observation_dim,args.intrinsic_dim,args.seed+123);roles={};samples={};noises={}
        for role,n in [('white',512),('edge',2048),('edge-cover',2048)]:
            seed=namespace('data',args.seed,role);dataset=runner.ControlledPhaseDataset('s1',embedding,n,args.controlled_noise_std,seed,args.factor_bins,args.split_offset,args.local_probability,args.split_mode)
            samples[role]=torch.stack([dataset[i][0] for i in range(n)]);roles[role]=[f'legacy-controlled:embedding{args.seed+123}:synthetic-seed{seed}:row{i}' for i in range(n)]
            noises[role]=torch.from_numpy(np.random.default_rng(namespace('noise',args.seed,role)).normal(size=(n,4)).astype(np.float32))[:,:dim]
        if any(namespace('data',args.seed,r) in [args.seed,args.seed+101,args.seed+202] for r in roles) or len(set(namespace('data',args.seed,r) for r in roles))!=3:raise ValueError('Calibration source seed overlap')
        refs={};reference_audits=[]
        def calibrate(age):
            before=digest(model);rng=torch.get_rng_state().clone();was=model.training;model.eval();raw={}
            with torch.no_grad():
                for role,x in samples.items():
                    pieces=[]
                    for start in range(0,len(x),128):
                        batch=x[start:start+128];mu,lv=model.encode(batch);hat=model.decoder(mu+torch.exp(.5*lv)*noises[role][start:start+128]);pieces.append((batch-hat).numpy())
                    raw[role]=np.concatenate(pieces)
            model.train(was)
            if digest(model)!=before or not torch.equal(rng,torch.get_rng_state()):raise ValueError('New calibration changed original model/RNG')
            signal=FixedPsi.fit(raw['white'],raw['edge'],SignalConfig(**P['manual_signal']),roles);folder=out/f'ref{age}';folder.mkdir();np.savez_compressed(folder/'calibration_residuals.npz',**raw);meta=dict(reference_updates=age,reference_model_sha256=before,reference_sha256=signal.reference_hash(),edge=signal.edge,config=P['manual_signal'],roles=roles,source='fresh independent S1 synthetic namespaces; original data generator/embedding',model_and_RNG_unchanged=True,coverage=signal.coverage(raw['edge-cover']));write(folder/'calibration.json',meta);reference_audits.append(dict(age=age,model=before,reference=signal.reference_hash()));refs[age]=signal
        calibrate(0);manual_rows={};append=runner.append_csv;native_loss=runner.reconstruction_loss;native_probe=runner.run_online_controlled_probe;probe_checks=0;log_checks=0;snapshot_checks=0;captured1={}
        def loss(reconstruction,batch):
            nonlocal step
            step+=1
            if step in expected_steps:
                raw=(batch-reconstruction).detach().cpu().numpy()[-64:];pre=digest(model);manual_rows[step]={}
                for age,signal in refs.items():
                    if age==1000 and step<=1000:continue
                    record=dict(step=step,updates_completed=step-1,pre_model_sha256=pre,reference_updates=age,reference_sha256=signal.reference_hash(),raw_residual_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),**signal.measure_window(raw));manual_rows[step][age]=record
                    with (out/f'ref{age}/signals.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            return native_loss(reconstruction,batch)
        def observed_append(path,row):
            nonlocal probe_checks,log_checks
            append(path,row);name=Path(path).name
            if name=='train_history.csv':
                original=original_train[int(row['step'])]
                for key in ['objective','reconstruction','kl','xi_pw','lambda_max_batch','lambda_edge','spike_rank','grad_norm']:
                    if float(row[key])!=float(original[key]):raise ValueError(f'Original training values changed {step} {key}')
                log_checks+=1
                if int(row['step'])==1:captured1.update({k:v.detach().clone() for k,v in model.state_dict().items()})
                if int(row['step'])==1000:calibrate(1000)
                if step%100==0 or step in [1,1001,2001]:write(out/'progress.json',dict(status='running',step=step,dimension=dim,training_logs_verified=log_checks,probes_verified=probe_checks,wall_seconds=time.monotonic()-started));print(f'original controlled z{dim} step{step}: exact native values, passive new signal',flush=True)
            elif name=='probe_history.csv':
                original=original_probe[int(row['step']),row['task']]
                if float(row['accuracy'])!=float(original['accuracy']):raise ValueError('Original theta probe changed')
                probe_checks+=1
        def observed_probe(*a,**kw):
            nonlocal snapshot_checks
            result=native_probe(*a,**kw);t=kw['global_step'];state=torch.load(parent/'snapshots'/f'step_{t:04d}.pt',map_location='cpu',weights_only=True)
            if digest(model)!=state['model_sha256']:raise ValueError('Original checkpoint trajectory changed')
            snapshot_checks+=1;return result
        runner.reconstruction_loss=loss;runner.append_csv=observed_append;runner.run_online_controlled_probe=observed_probe
        runner.train_formal(args,torch.device('cpu'),out/'native',model,stages,bounds,cal)
        final=digest(model);last=torch.load(parent/'last.pt',map_location='cpu',weights_only=True)
        if any(not torch.equal(model.state_dict()[k],v) for k,v in last['model_state'].items()) or step!=3000 or snapshot_checks!=15:raise ValueError('Final historical replay failed')
        final_state={k:v.detach().clone() for k,v in model.state_dict().items()};extended=[]
        for t,state,index in [(0,initial['model_state'],0),(1,captured1,0),(3000,final_state,2)]:
            model.load_state_dict(state)
            with torch.random.fork_rng(devices=[]):rows=native_probe(args=args,model=model,stage=stages[index],global_step=t,stage_index=index+1,step_in_stage=0 if t==0 else 1 if t==1 else 1000,device=torch.device('cpu'))
            for r in rows:
                if r['row_type']=='accuracy' and r['task']=='theta':extended.append(dict(step=t,accuracy=r['accuracy'],task='theta',view='original_online_rules_current_stage',origin='new_offline_extension_using_original_fitter_and_seed_rule'))
        with (out/'online_theta_extensions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(extended[0]));w.writeheader();w.writerows(extended)
        if any(sha(parent/n)!=h for n,h in pinned.items()):raise ValueError('Original parent files changed')
        write(out/'summary.json',dict(status='complete',dimension=dim,seed=0,formal_replay_updates=3000,calibration_replay_updates=3000,new_treatment_updates=0,initial_model_exact=True,final_model_exact=True,original_checkpoint_hashes_verified=snapshot_checks,original_training_logs_exact=log_checks,original_probe_rows_exact=probe_checks,original_primary_theta_preserved=True,manual_points={str(age):sum(age in v for v in manual_rows.values()) for age in [0,1000]},independent_calibration_roles=True,reference_model_and_rng_unchanged=True,new_signal_enters_loss=False,reference_audits=reference_audits,wall_seconds=time.monotonic()-started))
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e),step=step,traceback=traceback.format_exc()));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dim',required=True,type=int,choices=[2,4]);v=a.parse_args();run(v.dim)
