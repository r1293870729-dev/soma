"""Add only missing image latent2/4 conditions; unchanged objectives and data roles."""
import argparse,copy,json,math,time,traceback,types
import numpy as np
import torch
from study_paths import ROOT,PLAN,OLD,PARENT,sha,write,rows,append,stage_at,verify
from study_runtime import sources,tensor_digest,seed_for,hash_array,model_for,forward,Reader,Bank,probe,validate,calibrate,Assets,prepare_image,SchemeBObserver

def case(ds,seed,dim,entry,models,assets,reader,bank,fixture=False,monitor=True):
    name=f'{ds}_z{dim}_seed{seed}';folder=ROOT/'results'/('fixtures' if fixture else PLAN['run_id'])/(name+('_on' if monitor else '_off') if fixture else name);folder.mkdir(parents=True,exist_ok=False);(folder/'prepared').mkdir();started=time.monotonic();step=0;branch='prepare'
    try:
        p=copy.deepcopy(PLAN);data=prepare_image(assets,seed,p,folder/'prepared')
        for g in data['groups']:g['first_step']=1+sum(p['stage_steps'][:g['stage']])
        primary=json.loads((OLD/'configs/plan.json').read_text())['primary_latent_dims'][ds];parent=OLD/'results/all_3000_001'/f'{ds}_z{primary}_seed{seed}';source_ids=parent/'prepared/source_ids.npz';source_hash=sha(source_ids)
        with np.load(source_ids,allow_pickle=False) as a,np.load(folder/'prepared/source_ids.npz',allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Source image allocation changed')
        model,likelihood=model_for(entry,ds,seed,dim,ds);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay']);initial=tensor_digest(model);torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_sha256=initial),folder/'checkpoint_initial.pt')
        refs={};observer=SchemeBObserver(seed) if monitor else None;fork_observer=None;updates=0;branches=[]
        def status():
            v=dict(status='running',unit=name,branch=branch,step=step,updates=updates,planned=100 if fixture else 5000,wall_seconds=time.monotonic()-started,updated_at=time.time());write(folder/'progress.json',v);print(json.dumps(v),flush=True)
        def reference(age):
            target=folder/f'ref{age}';target.mkdir();spec=copy.deepcopy(p);spec['reference_updates']=age;h=tensor_digest(model);refs[age]=calibrate(model,assets,data,spec,seed,ds,target,reader,bank,lambda s:None)
            meta=json.loads((target/'calibration.json').read_text())
            if tensor_digest(model)!=h or meta['reference_model_sha256']!=h:raise ValueError('Reference model changed')
            write(folder/f'calibration_ref{age}.json',meta)
        if monitor:reference(0)
        eps=torch.from_numpy(np.random.default_rng(seed_for('training-epsilon',ds,seed)).normal(size=(3000,64,dim)).astype(np.float32));orders=[]
        for s,length in enumerate([1000]*3):
            rng=np.random.default_rng(seed_for('training-order',ds,seed,s));n=p['training_rows_per_stage'];orders.append(np.concatenate([rng.permutation(n) for _ in range(math.ceil(length*64/n))])[:length*64].reshape(length,64))
        orders=np.concatenate(orders);np.save(folder/'batch_indices.npy',orders)
        for branch,start,end in [('changing',1,100 if fixture else 3000)]+([] if fixture else [('stable',1001,3000)]):
            out=folder/branch;out.mkdir();(out/'snapshots').mkdir();raws=[];raw_steps=[]
            if branch=='stable':
                state=torch.load(folder/'checkpoint_fork.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);model.train();observer=fork_observer
                if tensor_digest(model)!=state['model_sha256'] or {int(v['step']) for v in optimizer.state.values()}!={1000}:raise ValueError('Fork model/optimizer age differs')
            for step in range(start,end+1):
                s=0 if branch=='stable' else stage_at(step);idx=orders[step-1];ids=data['train'][s][idx];pre=tensor_digest(model);rng=torch.get_rng_state().clone();diag=step%p['signal_every']==0 or step in p['diagnostic_extra_steps']
                if not fixture and (step%p['probe_every']==0 or step in p['diagnostic_extra_steps']):
                    with torch.random.fork_rng(devices=[]):pr,heads=probe(model,assets,data,p,seed,ds,step,s)
                    if any(r['probe_stage']>s+1 for r in pr):raise ValueError('Future probe task')
                    append(out/'probe_history.csv',pr);torch.save(dict(model=model.state_dict(),heads=heads,step=step,updates_completed=step-1,pre_model_sha256=pre),out/'snapshots'/f'pre_{step:04d}.pt')
                x=assets.read(s,ids,s);logits,mu,lv=forward(model,x,eps[step-1]);rec,mse,hat=entry.compute_reconstruction_terms(types.SimpleNamespace(dataset=ds),logits,x,likelihood);kl=models.kl_standard_normal(mu,lv);objective=rec+p['beta_kl']*kl
                if not torch.isfinite(objective):raise ValueError('Nonfinite objective')
                if monitor:
                    legacy,_=observer((x-hat).detach(),s);append(out/'legacy_every_batch.csv',[dict(step=step,updates_completed=step-1,offline_stage=s+1,pre_model_sha256=pre,**legacy)])
                    if diag:
                        raw=bank.get(s,ids,s)-reader(hat.detach());raws.append(raw);raw_steps.append(step)
                        for age,signal in refs.items():
                            record=dict(step=step,updates_completed=step-1,pre_model_sha256=pre,reference_updates=age,reference_sha256=signal.reference_hash(),raw_residual_sha256=hash_array(raw),**signal.measure_window(raw))
                            with (out/f'psi_ref{age}.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                        with torch.no_grad():h=model.encoder(x).flatten(1);rawlv=model.fc_logvar(h)
                        append(out/'diagnostics.csv',[dict(step=step,raw_logvar_min=float(rawlv.min()),raw_logvar_max=float(rawlv.max()),logvar_clipped_fraction=float(((rawlv< -12)|(rawlv>12)).float().mean()),posterior_mu_variance=float(mu.detach().var(0,unbiased=False).mean()),posterior_noise_variance=float(lv.detach().exp().mean()))])
                if tensor_digest(model)!=pre or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Probe or observer changed main model/RNG')
                optimizer.zero_grad(set_to_none=True);objective.backward()
                if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise ValueError('Nonfinite gradient')
                optimizer.step();updates+=1;append(out/'trajectory.csv',[dict(step=step,updates_before_forward=step-1,offline_stage=s+1,reconstruction=float(rec.detach()),pixel_mse=float(mse.detach()),kl=float(kl.detach()),objective=float(objective.detach()),pre_model_sha256=pre,input_ID_sha256=hash_array(ids))])
                if branch=='changing' and step==1000:
                    reference(1000);torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_sha256=tensor_digest(model),updates_completed=1000),folder/'checkpoint_fork.pt');fork_observer=copy.deepcopy(observer)
                if not fixture and step in [1000,2000,3000]:append(out/'validation.csv',[dict(step=step,**v) for v in validate(model,assets,data,p,seed,ds,entry,likelihood,s)])
                if step%100==0 or step in [1,1001,2001]:status()
            if raws:np.savez_compressed(out/'diagnostic_residuals.npz',steps=np.asarray(raw_steps),residuals=np.stack(raws))
            final=tensor_digest(model);torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates_completed=end),out/'checkpoint_final.pt');v=dict(status='fixture_complete' if fixture else 'complete',branch=branch,start=start,end=end,actual_optimizer_updates=end-start+1,final_model_sha256=final);write(out/'summary.json',v);branches.append(v)
        if sha(source_ids)!=source_hash:raise ValueError('Original source allocation modified')
        verify();reader.verify();result=dict(status='fixture_complete' if fixture else 'complete',unit=name,dataset=ds,latent_dim=dim,seed=seed,stage_steps=[1000]*3,actual_optimizer_updates=updates,branches=branches,new_lowdim=True,source_IDs_match_existing=True,source_ID_file_sha256=source_hash,initial_model_sha256=initial,reference_fit='own model at0 and1000, identical S1 roles; no highdim reference reused',signal_enters_loss=False,wall_seconds=time.monotonic()-started);write(folder/'summary.json',result);write(folder/'progress.json',result);return result
    except BaseException as e:write(folder/'FAILED.json',dict(branch=branch,step=step,error=repr(e),traceback=traceback.format_exc()));raise

def main(ds,fixture=False):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);verify();source,entry,models=sources();assets=Assets(ds);reader=Reader(ds,PLAN);bank=Bank(assets,reader,PARENT/'data_cache'/f'{ds}_observation_features.npz');bank.path=ROOT/'data_cache'/f'{ds}_observations.npz';done=[]
    if fixture:
        a=case(ds,10,2,entry,models,assets,reader,bank,True,True);b=case(ds,10,2,entry,models,assets,reader,bank,True,False);root=ROOT/'results/fixtures';ar=rows(root/f'{ds}_z2_seed10_on/changing/trajectory.csv');br=rows(root/f'{ds}_z2_seed10_off/changing/trajectory.csv')
        if ar!=br or a['branches'][0]['final_model_sha256']!=b['branches'][0]['final_model_sha256']:raise ValueError('Read-only monitoring changed training')
        write(ROOT/'reports'/f'qualification_{ds}.json',dict(status='pass',steps=100,every_training_row_identical_on_off=True,final_model_identical_on_off=True,own_ref0_fitted=True));return
    try:
        for seed in PLAN['image_seeds']:
            for dim in PLAN['image_dimensions'][ds]:
                done.append(case(ds,seed,dim,entry,models,assets,reader,bank));bank.save();write(ROOT/'reports'/f'worker_{ds}.json',dict(status='running',cases=done))
        write(ROOT/'reports'/f'worker_{ds}.json',dict(status='complete',cases=done))
    except BaseException as e:write(ROOT/'reports'/f'worker_{ds}.json',dict(status='failed',cases=done,error=repr(e)));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['dsprites','shapes3d','cifar'],required=True);p.add_argument('--fixture',action='store_true');a=p.parse_args();main(a.dataset,a.fixture)
