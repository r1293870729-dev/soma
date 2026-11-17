"""Fixed-weight historical VAE, passive manual Psi, and a matched stable fork."""
import copy,hashlib,json,math,time,types
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,plan,write_json,append,seed_for,hash_array,tensor_digest,FixedPsi,SignalConfig,FrozenReadout
from assets import prepare_image,prepare_controlled
from probes import fit_head

class Reader:
    def __init__(self,dataset,p):
        self.dataset=dataset;self.calls=0;self.rows=0
        spec={'kind':'identity'} if dataset=='controlled' else dict(p['readout'])
        if dataset!='controlled':spec['weights_path']=str(ROOT/spec['weights_path'])
        self.inner=FrozenReadout(dataset,spec,torch.device('cpu'));self.size=64 if dataset=='controlled' else p['readout']['microbatch']
        self.frozen_hash=tensor_digest(self.inner.model) if self.inner.model is not None else 'identity'
    @torch.no_grad()
    def __call__(self,x):
        pieces=[]
        for i in range(0,len(x),self.size):
            part=x[i:i+self.size];pieces.append(self.inner(part));self.calls+=1;self.rows+=len(part)
        a=torch.cat(pieces).cpu().numpy()
        if not np.isfinite(a).all():raise ValueError('Nonfinite readout')
        return a
    def verify(self):
        if self.inner.model is not None and tensor_digest(self.inner.model)!=self.frozen_hash:raise ValueError('Frozen reader mutated')

class Bank:
    def __init__(self,assets,reader,path):
        self.assets,self.reader,self.path=assets,reader,path;self.values={}
        if path.exists():
            with np.load(path,allow_pickle=False) as a:self.values={k.tobytes():v.copy() for k,v in zip(a['keys'],a['values'])}
    def get(self,stage,ids,arrived):
        keys=self.assets.obs_keys(stage,ids,arrived);missing={}
        for i,key in zip(ids,keys):
            if key.tobytes() not in self.values:missing.setdefault(key.tobytes(),int(i))
        items=list(missing.items())
        for start in range(0,len(items),64):
            block=items[start:start+64];x=self.assets.read(stage,[i for _,i in block],arrived);value=self.reader(x)
            self.values.update((key,row) for (key,_),row in zip(block,value))
        return np.stack([self.values[key.tobytes()] for key in keys])
    def save(self):
        keys=sorted(self.values);np.savez_compressed(self.path,keys=np.array([np.void(k) for k in keys],dtype='V32'),values=np.stack([self.values[k] for k in keys]))

def model_for(entry,dataset,seed,dimension,family):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for('model-init',family,seed))
        if dataset=='controlled':
            maximal=entry.MLPVAE(64,8,32,1);model=entry.MLPVAE(64,dimension,32,1)
            source=maximal.state_dict();target=model.state_dict()
            for name in target:
                if name.startswith('fc_mu.') or name.startswith('fc_logvar.'):target[name]=source[name][:dimension].clone()
                elif name=='decoder.0.weight':target[name]=source[name][:,:dimension].clone()*math.sqrt(8/dimension)
                elif name=='decoder.0.bias':target[name]=source[name].clone()*math.sqrt(8/dimension)
                else:target[name]=source[name].clone()
            model.load_state_dict(target);likelihood='gaussian'
        else:model,likelihood=entry.build_model(types.SimpleNamespace(dataset=dataset),dimension,torch.device('cpu'))
    return model,likelihood

def forward(model,x,epsilon,controlled=False):
    mu,lv=model.encode(x);z=mu+torch.exp(.5*lv)*epsilon[:,:mu.shape[1]]
    logits=model.decoder(z) if controlled else model.decode(z)
    return logits,mu,lv

def image_data(assets,prepared,key,stage,indices=None,arrived=None):
    values=prepared[key][stage]
    if assets is None:
        a=values['x'] if indices is None else values['x'][indices]
        return torch.from_numpy(a)
    ids=values if indices is None else values[indices]
    return assets.read(stage,ids,stage if arrived is None else arrived)

def probe(model,assets,data,p,seed,family,step,arrived):
    frozen=copy.deepcopy(model).eval();rows=[];heads={};dimension=model.fc_mu.out_features;maximum=8 if assets is None else dimension
    for group in data['groups']:
        if group['stage']>arrived or step<group['first_step']:continue
        features={'mu':{},'sampled_z':{}};labels={}
        for role,values in group['roles'].items():
            count=len(values['x']) if assets is None else len(values);mus=[];lvs=[]
            with torch.no_grad():
                for start in range(0,count,64):
                    x=torch.from_numpy(values['x'][start:start+64]) if assets is None else assets.read(group['stage'],values[start:start+64],arrived)
                    mu,lv=frozen.encode(x);mus.append(mu);lvs.append(lv)
                mu,lv=torch.cat(mus),torch.cat(lvs)
                eps=torch.from_numpy(np.random.default_rng(seed_for('probe-noise',family,seed,group['name'],role)).normal(size=(count,maximum)).astype(np.float32))[:,:dimension]
                features['mu'][role]=mu.double();features['sampled_z'][role]=(mu+torch.exp(.5*lv)*eps).double()
            labels[role]=values['labels'] if assets is None else assets.labels[values]
        for task in group['tasks']:
            column=task['column'];y={role:torch.from_numpy(v[:,column]) for role,v in labels.items()}
            classes=[1,2,3,4] if assets is None and task['name']=='branch' else list(range(task['classes']))
            for readout in ['mu','sampled_z']:
                result,state=fit_head(features[readout],y,classes,p['controlled_probe_head'] if assets is None else p['image_probe_head']);candidates=result.pop('candidates')
                rows.append(dict(step=step,updates_completed=step-1,probe_group=group['name'],probe_stage=group['stage']+1,task=task['name'],primary_task=task['primary'],readout=readout,classes=task['classes'],samples_train=len(y['probe-train']),samples_val=len(y['probe-val']),samples_test=len(y['probe-test']),**result))
                heads[group['name']+':'+task['name']+':'+readout]=dict(**state,candidates=candidates)
    return rows,heads

def calibrate(model,assets,data,p,seed,family,parent,reader,bank,status):
    before=tensor_digest(model);rng=torch.get_rng_state().clone();saved={};ids={};was=model.training;model.eval();maximum=8 if assets is None else model.fc_mu.out_features
    with torch.no_grad():
        for role,values in data['calibration'].items():
            count=len(values['x']) if assets is None else len(values)
            ids[role]=values['ids'] if assets is None else assets.identity_strings(values)
            eps=torch.from_numpy(np.random.default_rng(seed_for('calibration-epsilon',family,seed,role)).normal(size=(count,maximum)).astype(np.float32));pieces=[]
            obs=values['x'] if assets is None else bank.get(0,values,0)
            for start in range(0,count,64):
                x=torch.from_numpy(values['x'][start:start+64]) if assets is None else assets.read(0,values[start:start+64],0)
                logits,_,_=forward(model,x,eps[start:start+64],assets is None);hat=logits if assets is None else torch.sigmoid(logits)
                pieces.append(obs[start:start+64]-reader(hat))
                if start%512==0:status(f'calibrating fixed reference: {role} {start}/{count}')
            saved[role]=np.concatenate(pieces)
    model.train(was)
    if tensor_digest(model)!=before or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Calibration mutated training state')
    signal=FixedPsi.fit(saved['white'],saved['edge'],SignalConfig(**p['signal']),ids);cover=signal.coverage(saved['edge-cover'])
    np.savez_compressed(parent/'calibration_residuals.npz',white=saved['white'],edge=saved['edge'],edge_cover=saved['edge-cover'])
    write_json(parent/'calibration.json',dict(config=p['signal'],reference_updates=p['reference_updates'],reference_model_sha256=before,reference_sha256=signal.reference_hash(),edge=signal.edge,roles=ids,
        edge_cover=cover,independent_physical_sources=True,online_alarm_coverage_tested=False,model_and_rng_unchanged=True,readout_sha256=reader.frozen_hash))
    status(f'fixed reference ready; edge-cover {cover["exceedances"]}/{cover["windows"]}')
    return signal

def validate(model,assets,data,p,seed,family,entry,likelihood,arrived):
    was=model.training;model.eval();rows=[];maximum=8 if assets is None else model.fc_mu.out_features
    with torch.no_grad():
        for stage in range(arrived+1):
            n=p['validation_rows_per_stage'];eps=torch.from_numpy(np.random.default_rng(seed_for('validation-noise',family,seed,stage)).normal(size=(n,maximum)).astype(np.float32));totals=np.zeros(3)
            for start in range(0,n,64):
                x=image_data(assets,data,'validation',stage,np.arange(start,min(n,start+64)),arrived)
                logits,mu,lv=forward(model,x,eps[start:start+64],assets is None)
                rec,mse,_=entry.compute_reconstruction_terms(types.SimpleNamespace(dataset=data['dataset']),logits,x,likelihood)
                kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean();totals+=np.array([float(rec),float(mse),float(kl)])*len(x)
            rows.append(dict(stage=stage+1,rows=n,reconstruction=float(totals[0]/n),pixel_mse=float(totals[1]/n),kl=float(totals[2]/n)))
    model.train(was);return rows

def execute_branch(model,optimizer,signal,assets,data,p,seed,family,entry,models,likelihood,parent,branch,reader,bank,epsilon,orders,start,end,fixture=False,monitor_enabled=True):
    out=parent/branch;out.mkdir(exist_ok=False);(out/'snapshots').mkdir();rows=[];raw_steps=[];raw_values=[];started=time.monotonic();step=start-1;monitor_seconds=probe_seconds=0.;clip_max=0.;input_order=hashlib.sha256();initial=tensor_digest(model)
    def status(text):write_json(parent/'progress.json',dict(unit=parent.name,branch=branch,status='running',step=step,branch_start=start,branch_end=end,unit_updates_done=(step if branch=='changing' else 4000+step-2000),unit_updates_total=6000,message=text,wall_seconds=time.monotonic()-started,updated_at=time.time()))
    try:
        for step in range(start,end+1):
            natural=0 if step<=2000 else (1 if step<=3000 else 2);stage=0 if branch=='stable' else natural
            index=orders[step-1];pre=tensor_digest(model);diagnostic=step%p['signal_every']==0 or step in p['diagnostic_extra_steps'];status_tick=step%100==0 or step in [1,1001,2001,3001]
            if not fixture and (step%p['probe_every']==0 or step in p['diagnostic_extra_steps']):
                previous_rng=torch.get_rng_state().clone();pt=time.monotonic()
                with torch.random.fork_rng(devices=[]):probes,heads=probe(model,assets,data,p,seed,family,step,stage)
                if tensor_digest(model)!=pre or not torch.equal(previous_rng,torch.get_rng_state()):raise ValueError('Probe mutated training model/RNG')
                append(out/'probe_history.csv',probes);torch.save(dict(model=model.state_dict(),heads=heads,pre_model_sha256=pre,updates_completed=step-1,step=step),out/'snapshots'/f'pre_{step:04d}.pt');probe_seconds+=time.monotonic()-pt
            x=image_data(assets,data,'train',stage,index,stage)
            logits,mu,lv=forward(model,x,epsilon[step-1],assets is None);rec,mse,hat=entry.compute_reconstruction_terms(types.SimpleNamespace(dataset=data['dataset']),logits,x,likelihood)
            kl=models.kl_standard_normal(mu,lv);objective=rec+p['beta_kl']*kl
            if not torch.isfinite(objective) or float(rec)<0 or float(kl)<0:raise ValueError('Invalid training objective')
            measured=None
            if diagnostic and monitor_enabled:
                pt=time.monotonic();rng=torch.get_rng_state().clone()
                raw=(x.detach().numpy()-hat.detach().numpy()) if assets is None else bank.get(stage,data['train'][stage][index],stage)-reader(hat.detach())
                if not np.isfinite(raw).all():raise ValueError('Nonfinite residual')
                raw_steps.append(step);raw_values.append(raw)
                if signal is not None:
                    measured=signal.measure_window(raw)
                    with (out/'psi.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,pre_model_sha256=pre,reference_sha256=signal.reference_hash(),**measured),allow_nan=False)+'\n')
                params=list(model.encoder.parameters());gr=torch.autograd.grad(rec,params,retain_graph=True);gk=torch.autograd.grad(kl,params,retain_graph=True)
                gr=torch.cat([v.flatten() for v in gr]);gk=torch.cat([v.flatten() for v in gk]);nr,nk=gr.norm(),gk.norm();cos=float(torch.dot(gr,gk)/(nr*nk)) if float(nr*nk)>0 else None
                with torch.no_grad():
                    h=model.encoder(x);h=h if assets is None else h.flatten(1);rawlv=model.fc_logvar(h);clip=float(((rawlv< -12)|(rawlv>12)).float().mean());clip_max=max(clip_max,clip)
                append(out/'diagnostics.csv',[dict(step=step,rec_gradient_norm=float(nr),KL_gradient_norm=float(nk),weighted_KL_gradient_norm=float(nk)*p['beta_kl'],gradient_cosine=cos,
                    posterior_mu_variance=float(mu.detach().var(0,unbiased=False).mean()),posterior_noise_variance=float(lv.detach().exp().mean()),raw_logvar_min=float(rawlv.min()),raw_logvar_max=float(rawlv.max()),logvar_clipped_fraction=clip)])
                if tensor_digest(model)!=pre or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Monitor changed model/RNG')
                monitor_seconds+=time.monotonic()-pt
            optimizer.zero_grad(set_to_none=True);objective.backward()
            if any(a.grad is not None and not torch.isfinite(a.grad).all() for a in model.parameters()):raise ValueError('Invalid gradient')
            optimizer.step()
            if assets is None:ids_hash=hash_array(index)
            else:ids_hash=hash_array(data['train'][stage][index])
            input_order.update(ids_hash.encode())
            append(out/'trajectory.csv',[dict(step=step,updates_before_forward=step-1,offline_stage=stage+1,reconstruction=float(rec.detach()),pixel_mse=float(mse.detach()),kl=float(kl.detach()),objective=float(objective.detach()),beta_kl=p['beta_kl'],
                psi=None if measured is None else measured['psi'],mean_total_variance=None if measured is None else measured['mean_total_variance'],pre_model_sha256=pre,input_ID_sha256=ids_hash)])
            if monitor_enabled and branch=='changing' and step==p['reference_updates']:
                signal=calibrate(model,assets,data,p,seed,family,parent,reader,bank,status)
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates_completed=step),parent/'checkpoint_reference.pt')
            if branch=='changing' and step==2000:
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates_completed=step,model_sha256=tensor_digest(model)),parent/'checkpoint_fork.pt')
            if not fixture and step in [2000,3000,4000]:
                values=validate(model,assets,data,p,seed,family,entry,likelihood,stage)
                append(out/'validation.csv',[dict(step=step,**v) for v in values])
            if status_tick:
                text=f'{parent.name} {branch} {step}/{end} | rec={float(rec.detach()):.5g} KL={float(kl.detach()):.5g} Psi={None if measured is None else measured["psi"]}'
                print(text,flush=True);status(text)
        final=tensor_digest(model);torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates_completed=end),out/'checkpoint_final.pt')
        if raw_values:np.savez_compressed(out/'diagnostic_residuals.npz',steps=np.array(raw_steps),residuals=np.stack(raw_values))
        result=dict(status='fixture_complete' if fixture else 'complete',unit=parent.name,branch=branch,start=start,end=end,actual_optimizer_updates=end-start+1,initial_model_sha256=initial,final_model_sha256=final,
            input_order_sha256=input_order.hexdigest(),fixed_reference_sha256=None if signal is None else signal.reference_hash(),diagnostic_windows=len(raw_values),probe_seconds=probe_seconds,monitor_seconds=monitor_seconds,wall_seconds=time.monotonic()-started,
            max_observed_logvar_clipped_fraction=clip_max,reference_fitted_at=None if signal is None else p['reference_updates'],monitor_enabled=monitor_enabled,alarms_calibrated=False,final_test_unlocked=False)
        write_json(out/'summary.json',result);status('branch complete');return result,signal
    except BaseException as e:
        if raw_values:np.savez_compressed(out/'partial_residuals.npz',steps=np.array(raw_steps),residuals=np.stack(raw_values))
        write_json(out/'FAILED.json',dict(step=step,error=repr(e)));raise

def run_parent(entry,models,assets,generator,seed,dimension,p,root,reader,bank,fixture=False,monitor_enabled=True):
    dataset='controlled' if assets is None else assets.dataset;family=generator.family if generator is not None else dataset
    name=f'{family}_z{dimension}_seed{seed}';parent=root/name;parent.mkdir(exist_ok=False);started=time.monotonic()
    data=prepare_controlled(generator,seed,p,parent) if assets is None else prepare_image(assets,seed,p,parent)
    model,likelihood=model_for(entry,dataset,seed,dimension,family);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    initial=tensor_digest(model);torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),parent/'checkpoint_initial.pt')
    maximum=8 if assets is None else dimension
    epsilon=torch.from_numpy(np.random.default_rng(seed_for('training-epsilon',family,seed)).normal(size=(4000,64,maximum)).astype(np.float32))
    orders=[]
    for stage,length in enumerate([2000,1000,1000]):
        rng=np.random.default_rng(seed_for('training-order',family,seed,stage));n=p['training_rows_per_stage']
        orders.append(np.concatenate([rng.permutation(n) for _ in range(math.ceil(length*64/n))])[:length*64].reshape(length,64))
    orders=np.concatenate(orders);np.save(parent/'batch_indices.npy',orders)
    main,signal=execute_branch(model,optimizer,None,assets,data,p,seed,family,entry,models,likelihood,parent,'changing',reader,bank,epsilon,orders,1,100 if fixture else 4000,fixture,monitor_enabled)
    results=[main]
    if not fixture:
        state=torch.load(parent/'checkpoint_fork.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);model.train()
        if tensor_digest(model)!=state['model_sha256']:raise ValueError('Parent fork mismatch')
        ages=[int(v['step']) for v in optimizer.state.values()]
        if not ages or any(age!=2000 for age in ages):raise ValueError('Adam fork age mismatch')
        write_json(parent/'fork_audit.json',dict(updates_completed=2000,model_sha256=state['model_sha256'],optimizer='AdamW',optimizer_state_tensors=len(ages),optimizer_update_ages=sorted(set(ages)),full_optimizer_state_restored=True))
        stable,_=execute_branch(model,optimizer,copy.deepcopy(signal),assets,data,p,seed,family,entry,models,likelihood,parent,'stable',reader,bank,epsilon,orders,2001,4000,False)
        changing_rows=__import__('csv').DictReader((parent/'changing/trajectory.csv').open());atfork=next(r for r in changing_rows if int(r['step'])==2001)
        if stable['initial_model_sha256']!=atfork['pre_model_sha256']:raise ValueError('Fork branches did not share parent')
        if stable['fixed_reference_sha256']!=main['fixed_reference_sha256']:raise ValueError('Fork reference changed')
        results.append(stable)
    reader.verify()
    if bank is not None:bank.save()
    result=dict(status='fixture_complete' if fixture else 'complete',dataset=dataset,family=family,seed=seed,latent_dim=dimension,beta_kl=p['beta_kl'],initial_model_sha256=initial,
        actual_model_updates=sum(r['actual_optimizer_updates'] for r in results),branches=results,wall_seconds=time.monotonic()-started,reader_calls_cumulative=reader.calls,reader_images_cumulative=reader.rows,
        final_test_unlocked=False,signal_enters_loss=False,parameters=sum(a.numel() for a in model.parameters()))
    write_json(parent/'summary.json',result);write_json(parent/'progress.json',dict(status=result['status'],unit=name,unit_updates_done=result['actual_model_updates'],unit_updates_total=6000))
    return result
