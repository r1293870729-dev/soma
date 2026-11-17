"""Fresh1000/1000/1000 image trajectories, paired stable forks, passive old/new signals."""
import argparse,copy,csv,json,math,sys,time,traceback,types
import numpy as np
import torch
from paths import ROOT,PLAN,PARENT,IMAGE,PAIR,stage_at,sha,write,read,verify
sys.path.insert(0,str(PAIR/'src'))
from observer import SchemeBObserver
from common import sources,tensor_digest,seed_for,hash_array,FixedPsi,SignalConfig
from engine import model_for,forward,Reader,Bank,probe,validate
from assets import Assets,prepare_image

def rebuild(folder):
    meta=json.loads((folder/'calibration.json').read_text())
    with np.load(folder/'calibration_residuals.npz',allow_pickle=False) as a:s=FixedPsi.fit(a['white'],a['edge'],SignalConfig(**meta['config']),meta['roles'])
    if s.reference_hash()!=meta['reference_sha256']:raise ValueError('Source reference reconstruction failed')
    return s,meta

def run_case(ds,seed,dim,source,entry,models,assets,reader,bank,fixture=False):
    name=f'{ds}_z{dim}_seed{seed}';out=ROOT/'results'/('fixtures' if fixture else PLAN['run_id'])/name;out.mkdir(parents=True,exist_ok=False);(out/'prepared').mkdir();oldparent=IMAGE/'results/generated/image_dims_refs_001'/name;oldobs=IMAGE/'results/generated/image_dims_refs_001/observers'/name;p=copy.deepcopy(PLAN);started=time.monotonic();step=0;branch='prepare'
    try:
        files={str(f.resolve()):sha(f) for f in [oldparent/'checkpoint_initial.pt',oldparent/'checkpoint_reference.pt',oldparent/'source_ids.npz',oldparent/'changing/trajectory.csv',oldparent/'calibration.json',oldparent/'calibration_residuals.npz',oldobs/'ref0/calibration.json',oldobs/'ref0/calibration_residuals.npz']};write(out/'source_inputs.json',files)
        data=prepare_image(assets,seed,p,out/'prepared')
        for g in data['groups']:g['first_step']=1+sum(p['stage_steps'][:g['stage']])
        with np.load(oldparent/'source_ids.npz',allow_pickle=False) as a,np.load(out/'prepared/source_ids.npz',allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Image role/source allocation changed')
        model,likelihood=model_for(entry,ds,seed,dim,ds);initial=torch.load(oldparent/'checkpoint_initial.pt',map_location='cpu',weights_only=True);model.load_state_dict(initial['model']);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay']);optimizer.load_state_dict(initial['optimizer'])
        initial_hash=tensor_digest(model);torch.save(initial,out/'checkpoint_initial.pt');reference0,cal0=rebuild(oldobs/'ref0');reference1000,cal1000=rebuild(oldparent)
        if cal0['reference_model_sha256']!=initial_hash or cal0['roles']!=cal1000['roles']:raise ValueError('Reference0 initial model or roles differ')
        for age,m in [(0,cal0),(1000,cal1000)]:write(out/f'calibration_ref{age}.json',dict(**m,reused_causally_from_existing_exact_model=True))
        observer=SchemeBObserver(seed);fork_observer=None;eps=torch.from_numpy(np.random.default_rng(seed_for('training-epsilon',ds,seed)).normal(size=(3000,64,dim)).astype(np.float32));orders=[]
        for s,nsteps in enumerate(p['stage_steps']):
            rng=np.random.default_rng(seed_for('training-order',ds,seed,s));n=p['training_rows_per_stage'];orders.append(np.concatenate([rng.permutation(n) for _ in range(math.ceil(nsteps*64/n))])[:nsteps*64].reshape(nsteps,64))
        orders=np.concatenate(orders);np.save(out/'batch_indices.npy',orders);original={int(r['step']):r for r in read(oldparent/'changing/trajectory.csv') if int(r['step'])<=1000};updates=prefix_checks=0;results=[];limit=100 if fixture else 3000;reference_ready=False
        for branch,start,end in [('changing',1,limit)]+([] if fixture else [('stable',1001,3000)]):
            folder=out/branch;folder.mkdir();(folder/'snapshots').mkdir();raw_steps=[];raws=[];old_rows=[];raw_logvar_max_clip=0.
            if branch=='stable':
                state=torch.load(out/'checkpoint_fork.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);observer=fork_observer;model.train();reference_ready=True
                if tensor_digest(model)!=state['model_sha256'] or any(int(v['step'])!=1000 for v in optimizer.state.values()):raise ValueError('Stable fork model/Adam age invalid')
            handles={age:(folder/f'psi_ref{age}.jsonl').open('w') for age in [0,1000]};history=[];old_file=(folder/'legacy_every_batch.csv').open('w',newline='');old_writer=None
            try:
                for step in range(start,end+1):
                    natural=stage_at(step);stage=0 if branch=='stable' else natural;idx=orders[step-1];pre=tensor_digest(model);ids=data['train'][stage][idx];ids_hash=hash_array(ids);diag=step%p['signal_every']==0 or step in p['diagnostic_extra_steps']
                    if branch=='changing' and step<=1000 and (pre!=original[step]['pre_model_sha256'] or ids_hash!=original[step]['input_ID_sha256']):raise ValueError(f'Prefix model/input does not match source at{step}')
                    if not fixture and (step%p['probe_every']==0 or step in p['diagnostic_extra_steps']):
                        before_rng=torch.get_rng_state().clone()
                        with torch.random.fork_rng(devices=[]):pr,heads=probe(model,assets,data,p,seed,ds,step,stage)
                        if tensor_digest(model)!=pre or not torch.equal(before_rng,torch.get_rng_state()):raise ValueError('Probe changed main model/RNG')
                        if any(r['probe_stage']>stage+1 for r in pr):raise ValueError('Future task leaked into probe')
                        append_csv(folder/'probe_history.csv',pr);torch.save(dict(model=model.state_dict(),heads=heads,step=step,updates_completed=step-1,pre_model_sha256=pre),folder/'snapshots'/f'pre_{step:04d}.pt')
                    x=assets.read(stage,ids,stage);logits,mu,lv=forward(model,x,eps[step-1]);rec,mse,hat=entry.compute_reconstruction_terms(types.SimpleNamespace(dataset=ds),logits,x,likelihood);kl=models.kl_standard_normal(mu,lv);objective=rec+p['beta_kl']*kl
                    if not torch.isfinite(objective):raise ValueError('Nonfinite training objective')
                    if branch=='changing' and step<=1000:
                        for k,v in [('reconstruction',rec),('pixel_mse',mse),('kl',kl),('objective',objective)]:
                            if float(v.detach())!=float(original[step][k]):raise ValueError(f'Prefix loss differs at{step} {k}')
                        prefix_checks+=1
                    before_rng=torch.get_rng_state().clone();legacy,_=observer((x-hat).detach(),stage)
                    row=dict(step=step,updates_completed=step-1,offline_stage=stage+1,pre_model_sha256=pre,**legacy)
                    if old_writer is None:old_writer=csv.DictWriter(old_file,fieldnames=list(row));old_writer.writeheader()
                    old_writer.writerow(row)
                    measured={}
                    if diag:
                        raw=bank.get(stage,ids,stage)-reader(hat.detach());raws.append(raw);raw_steps.append(step)
                        for age,signal in [(0,reference0),(1000,reference1000)]:
                            if age==1000 and not reference_ready:continue
                            m=signal.measure_window(raw);measured[age]=m;handles[age].write(json.dumps(dict(step=step,updates_completed=step-1,pre_model_sha256=pre,raw_residual_sha256=hash_array(raw),reference_updates=age,reference_sha256=signal.reference_hash(),**m),allow_nan=False)+'\n')
                        with torch.no_grad():h=model.encoder(x).flatten(1);rawlv=model.fc_logvar(h);clip=float(((rawlv< -12)|(rawlv>12)).float().mean());raw_logvar_max_clip=max(raw_logvar_max_clip,clip)
                        append_csv(folder/'diagnostics.csv',[dict(step=step,raw_logvar_min=float(rawlv.min()),raw_logvar_max=float(rawlv.max()),logvar_clipped_fraction=clip,posterior_mu_variance=float(mu.detach().var(0,unbiased=False).mean()),posterior_noise_variance=float(lv.detach().exp().mean()))])
                    if tensor_digest(model)!=pre or not torch.equal(before_rng,torch.get_rng_state()):raise ValueError('Observer changed main model/RNG')
                    optimizer.zero_grad(set_to_none=True);objective.backward()
                    if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise ValueError('Nonfinite gradient')
                    optimizer.step();updates+=1;append_csv(folder/'trajectory.csv',[dict(step=step,updates_before_forward=step-1,offline_stage=stage+1,reconstruction=float(rec.detach()),pixel_mse=float(mse.detach()),kl=float(kl.detach()),objective=float(objective.detach()),pre_model_sha256=pre,input_ID_sha256=ids_hash)])
                    if branch=='changing' and step==1000:
                        source_state=torch.load(oldparent/'checkpoint_reference.pt',map_location='cpu',weights_only=True)
                        if any(not torch.equal(model.state_dict()[k],v) for k,v in source_state['model'].items()) or tensor_digest(model)!=cal1000['reference_model_sha256']:raise ValueError('Model after1000 differs; cannot reuse reference1000')
                        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_sha256=tensor_digest(model),updates_completed=1000),out/'checkpoint_fork.pt');fork_observer=copy.deepcopy(observer);reference_ready=True
                    if not fixture and step in [1000,2000,3000]:append_csv(folder/'validation.csv',[dict(step=step,**v) for v in validate(model,assets,data,p,seed,ds,entry,likelihood,stage)])
                    if step%100==0 or step in [1,1001,2001]:
                        old_file.flush();status=dict(status='running',unit=name,branch=branch,step=step,updates=updates,planned=100 if fixture else 5000,prefix_exact_checks=prefix_checks,wall_seconds=time.monotonic()-started,updated_at=time.time());write(out/'progress.json',status);print(json.dumps(status),flush=True)
            finally:
                old_file.close()
                for h in handles.values():h.close()
                if raws:np.savez_compressed(folder/'diagnostic_residuals.npz',steps=np.asarray(raw_steps),residuals=np.stack(raws))
            final=tensor_digest(model);torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates_completed=end),folder/'checkpoint_final.pt');result=dict(status='fixture_complete' if fixture else 'complete',branch=branch,start=start,end=end,actual_optimizer_updates=end-start+1,final_model_sha256=final,max_observed_logvar_clip=raw_logvar_max_clip);write(folder/'summary.json',result);results.append(result)
            if fixture and final!=original[end+1]['pre_model_sha256']:raise ValueError('Fixture endpoint does not match original prefix')
        if any(sha(f)!=h for f,h in files.items()):raise ValueError('Input source modified')
        verify();source.verify();reader.verify();bank.save();result=dict(status='fixture_complete' if fixture else 'complete',unit=name,dataset=ds,latent_dim=dim,seed=seed,stage_steps=[1000,1000,1000],changing_curve_end=3000,stable_curve_end=3000,prefix_exact_checks=prefix_checks,reference0_causal=True,reference1000_model_exact=True,actual_optimizer_updates=updates,branches=results,signal_enters_loss=False,all_planned_seeds_retained=True,wall_seconds=time.monotonic()-started);write(out/'summary.json',result);write(out/'progress.json',result);return result
    except BaseException as e:write(out/'FAILED.json',dict(step=step,branch=branch,error=repr(e),traceback=traceback.format_exc()));raise

def append_csv(path,rows):
    if not rows:return
    first=not path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader() if first else None;w.writerows(rows)

def main(ds,fixture=False):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);verify();source,entry,models=sources();assets=Assets(ds);reader=Reader(ds,PLAN);bank=Bank(assets,reader,PARENT/'data_cache'/f'{ds}_observation_features.npz');bank.path=ROOT/'data_cache'/f'{ds}_observations.npz';cases=[];failures=[]
    order=[(10,PLAN['primary_latent_dims'][ds])] if fixture else [(s,d) for s in PLAN['image_seeds'] for d in PLAN['image_dimensions'][ds]]
    for seed,dim in order:
        try:cases.append(run_case(ds,seed,dim,source,entry,models,assets,reader,bank,fixture))
        except Exception as exc:
            failures.append(dict(dataset=ds,seed=seed,dimension=dim,error=repr(exc)));write(ROOT/'reports'/f'worker_{ds}.json',dict(status='failed',cases=cases,failures=failures));raise
        write(ROOT/'reports'/f'worker_{ds}.json',dict(status='fixture_complete' if fixture else 'running',cases=cases,failures=failures))
    write(ROOT/'reports'/f'worker_{ds}.json',dict(status='fixture_complete' if fixture else 'complete',cases=cases,failures=failures))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=PLAN['image_dimensions']);p.add_argument('--fixture',action='store_true');a=p.parse_args();main(a.dataset,a.fixture)
