"""Replay every native optimizer update; no alternate treatment is trained."""
import argparse,copy,csv,json,time,traceback,types,math
from pathlib import Path
import numpy as np
import torch
from runtime import ROOT,PLAN,PARENT,PARENT_RUN,REFS,sha,write,read,verify
from observer import SchemeBObserver
from common import sources,seed_for,hash_array,tensor_digest,plan as parent_plan
from engine import model_for,forward
from assets import Assets

def run(dataset,seed,fixture=False):
    verify();dim=PLAN['datasets'][dataset];name=f'{dataset}_z{dim}_seed{seed}';parent=PARENT_RUN/name;out=ROOT/'results'/PLAN['run_id']/(name+('_fixture100' if fixture else ''));out.mkdir(parents=True,exist_ok=False)
    source,entry,models=sources();p=parent_plan();assets=Assets(dataset);before_source={}
    paths=['checkpoint_initial.pt','checkpoint_fork.pt','summary.json','source_ids.npz','batch_indices.npy']
    for branch in ['changing','stable']:paths += [branch+'/'+v for v in ['trajectory.csv','summary.json','checkpoint_final.pt','probe_history.csv','psi.jsonl']]
    for rel in paths:before_source[rel]=sha(parent/rel)
    manual=REFS/name
    if json.loads((manual/'summary.json').read_text())['status']!='complete':raise ValueError('Manual two-reference observer unavailable')
    manual_hash={str(f.relative_to(manual)):sha(f) for f in manual.rglob('*.jsonl')}
    write(out/'parent_assets.json',dict(parent=str(parent),parent_files=before_source,manual_observer=str(manual),manual_files=manual_hash))
    with np.load(parent/'source_ids.npz',allow_pickle=False) as a:ids={s:a[f'train_{s}'].copy() for s in range(3)}
    orders=np.load(parent/'batch_indices.npy',allow_pickle=False)
    eps=torch.from_numpy(np.random.default_rng(seed_for('training-epsilon',dataset,seed)).normal(size=(4000,64,dim)).astype(np.float32))
    model,likelihood=model_for(entry,dataset,seed,dim,dataset);optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    initial=torch.load(parent/'checkpoint_initial.pt',map_location='cpu',weights_only=True);model.load_state_dict(initial['model']);optimizer.load_state_dict(initial['optimizer']);model.train()
    observer=SchemeBObserver(seed);fork_observer=None;started=time.monotonic();updates=0;branch='changing';step=0;results=[];probe_hashes={}
    try:
        for branch,start,end in [('changing',1,100 if fixture else 4000)]+([] if fixture else [('stable',2001,4000)]):
            if branch=='stable':
                state=torch.load(parent/'checkpoint_fork.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);observer=fork_observer;model.train()
                if tensor_digest(model)!=state['model_sha256']:raise ValueError('Fork model mismatch')
            folder=out/branch;folder.mkdir();expected={int(r['step']):r for r in read(parent/branch/'trajectory.csv')};checks=0
            with (folder/'signals_every_batch.csv').open('w',newline='') as f,(folder/'spectra_diagnostic.jsonl').open('w') as spec:
                writer=None
                for step in range(start,end+1):
                    original=expected[step];stage=0 if branch=='stable' or step<=2000 else 1 if step<=3000 else 2;pre=tensor_digest(model)
                    if pre!=original['pre_model_sha256']:raise ValueError(f'Model trajectory mismatch {branch} step {step}')
                    batch_ids=ids[stage][orders[step-1]]
                    if hash_array(batch_ids)!=original['input_ID_sha256']:raise ValueError(f'Input IDs differ {branch} {step}')
                    x=assets.read(stage,batch_ids,stage);logits,mu,lv=forward(model,x,eps[step-1]);rec,mse,hat=entry.compute_reconstruction_terms(types.SimpleNamespace(dataset=dataset),logits,x,likelihood);kl=models.kl_standard_normal(mu,lv);objective=rec+p['beta_kl']*kl
                    for key,v in [('reconstruction',rec),('pixel_mse',mse),('kl',kl),('objective',objective)]:
                        if not torch.isfinite(v) or float(v.detach())!=float(original[key]):raise ValueError(f'Objective differs {branch} {step} {key}')
                    rng=torch.get_rng_state().clone();measured,spectrum=observer((x-hat).detach(),stage)
                    if tensor_digest(model)!=pre or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Passive observer mutated model/RNG')
                    if measured['eligible']:
                        expected_xi=float(np.maximum(np.float32(measured['lambda_max'])-np.float32(measured['lambda_edge']),np.float32(0)))
                        if measured['xi']!=expected_xi:raise ValueError('Absolute-edge equation mismatch')
                    row=dict(step=step,updates_before_forward=step-1,branch=branch,pre_model_sha256=pre,input_ID_sha256=original['input_ID_sha256'],**measured)
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
                    if step%50==0 or step in [1,21,1001,2001,2021,3001,3021]:spec.write(json.dumps(dict(step=step,branch=branch,spectrum=spectrum,xi=measured['xi'],eligible=measured['eligible'],edge=measured['lambda_edge'],baseline_source=measured['baseline_source']),allow_nan=False)+'\n')
                    optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step();updates+=1;checks+=1
                    if branch=='changing' and step==2000:fork_observer=copy.deepcopy(observer)
                    if step%100==0 or step in [1,21,2001,2021,3001,3021]:
                        f.flush();spec.flush();status=dict(status='running',unit=name,branch=branch,step=step,replay_updates=updates,planned_updates=100 if fixture else 6000,xi=measured['xi'],wall_seconds=time.monotonic()-started,updated_at=time.time());write(out/'progress.json',status);print(json.dumps(status),flush=True)
            final=tensor_digest(model);wanted=expected[end+1]['pre_model_sha256'] if fixture else json.loads((parent/branch/'summary.json').read_text())['final_model_sha256']
            if final!=wanted:raise ValueError('Final model hash mismatch')
            result=dict(status='fixture_complete' if fixture else 'complete',branch=branch,steps=checks,exact_model_checks=checks,exact_input_checks=checks,exact_loss_checks=checks,final_model_sha256=final,parent_final_model_sha256=wanted,probe_reused=True)
            if not fixture:
                for file in ['probe_history.csv','trajectory.csv','validation.csv']:
                    if (parent/branch/file).exists():__import__('shutil').copy2(parent/branch/file,folder/file)
                (folder/'manual_observer').symlink_to(manual,target_is_directory=True)
            write(folder/'summary.json',result);results.append(result)
        if any(sha(parent/n)!=h for n,h in before_source.items()) or any(sha(manual/n)!=h for n,h in manual_hash.items()):raise ValueError('Parent artifacts changed')
        verify();source.verify()
        result=dict(status='fixture_complete' if fixture else 'complete',unit=name,dataset=dataset,latent_dim=dim,seed=seed,replay_optimizer_updates=updates,new_treatment_optimizer_updates=0,branches=results,parent_assets_unchanged=True,all_model_updates_identical=True,all_inputs_and_losses_identical=True,signal_enters_loss=False,probes_reused_unchanged=True,legacy_true_stage_assisted=True,observer_reads_all_batches=True,wall_seconds=time.monotonic()-started,updated_at=time.time())
        write(out/'summary.json',result);write(out/'progress.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:write(out/'FAILED.json',dict(error=repr(exc),branch=branch,step=step,replay_updates=updates,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dataset',required=True,choices=PLAN['datasets']);a.add_argument('--seed',required=True,type=int,choices=PLAN['seeds']);a.add_argument('--fixture',action='store_true');v=a.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);run(v.dataset,v.seed,v.fixture)

