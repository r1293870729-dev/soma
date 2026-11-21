"""Exact image trajectory replay plus real per-step switch-window measurements."""
import argparse,copy,json,sys,time,traceback,types
from pathlib import Path
import numpy as np
import torch
from dense_common import ROOT,LOW,PARENT,PLAN,SOURCE_PLAN,selected,wanted,cases,pin,read,jread,write,append,verify,check_sources,compare,reference
from study_runtime import sources,tensor_digest,seed_for,hash_array,model_for,forward,Reader,Bank,probe,Assets,prepare_image,SchemeBObserver,metrics
from replay_frozen import head_hash

def frozen_probe(model,assets,data,heads,epsilon,step,stage):
    group=next(g for g in data['groups'] if g['name']=='S1');ids=group['roles']['probe-test'];mus=[];lvs=[];frozen=copy.deepcopy(model).eval()
    with torch.no_grad():
        for start in range(0,len(ids),64):
            mu,lv=frozen.encode(assets.read(0,ids[start:start+64],stage));mus.append(mu);lvs.append(lv)
        mu,lv=torch.cat(mus),torch.cat(lvs);features={'mu':mu.double(),'sampled_z':(mu+torch.exp(.5*lv)*epsilon).double()};result=[]
        for task in group['tasks']:
            y=torch.from_numpy(assets.labels[ids,task['column']]).long()
            for rd in ['mu','sampled_z']:
                h=heads[f'S1:{task["name"]}:{rd}'];m=metrics((features[rd]-h['mean'])/h['std']@h['weight']+h['bias'],y,len(h['classes']))
                result.append(dict(step=step,updates_completed=step-1,probe_group='S1',task=task['name'],readout=rd,classes=task['classes'],**m))
    return result

def run_case(ds,case,meta,entry,models,assets,reader,bank):
    out=ROOT/'results'/case.name
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text());assert old['status']=='complete';check_sources(old['source_files']);return old
    out.mkdir(exist_ok=False);(out/'prepared').mkdir();pins={};start=time.monotonic();step=0;branch='prepare';count=0
    try:
        jread(pins,case/'summary.json')
        seed,dim=meta['seed'],meta['latent_dim'];spec=copy.deepcopy(SOURCE_PLAN);data=prepare_image(assets,seed,spec,out/'prepared')
        for g in data['groups']:g['first_step']=1+sum(spec['stage_steps'][:g['stage']])
        with np.load(pin(pins,case/'prepared/source_ids.npz'),allow_pickle=False) as a,np.load(out/'prepared/source_ids.npz',allow_pickle=False) as b:assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
        original={};original_probes={};original_old={};original_signals={}
        for b in ['changing','stable']:
            original[b]={int(r['step']):r for r in read(pins,case/b/'trajectory.csv')}
            original_probes[b]={(int(r['step']),r['probe_group'],r['task'],r['readout']):r for r in read(pins,case/b/'probe_history.csv')}
            original_old[b]={int(r['step']):r for r in read(pins,case/b/'legacy_every_batch.csv')}
            original_signals[b]={a:{r['step']:r for r in [json.loads(v) for v in pin(pins,case/b/f'psi_ref{a}.jsonl').read_text().splitlines() if v]} for a in [0,1000]}
        prior_frozen=read(pins,LOW/'results/probe_replay'/case.name/'frozen_probe.csv');prior_frozen={(r['branch'],int(r['step']),r['task'],r['readout']):r for r in prior_frozen}
        anchor=torch.load(pin(pins,case/'changing/snapshots/pre_0900.pt'),map_location='cpu',weights_only=True);heads={k:v for k,v in anchor['heads'].items() if k.startswith('S1:')};hh=head_hash(heads)
        n=len(next(g for g in data['groups'] if g['name']=='S1')['roles']['probe-test']);probe_noise=torch.from_numpy(np.random.default_rng(seed_for('probe-noise',ds,seed,'S1','probe-test')).normal(size=(n,dim)).astype(np.float32))
        model,likelihood=model_for(entry,ds,seed,dim,ds);initial=torch.load(pin(pins,case/'checkpoint_initial.pt'),map_location='cpu',weights_only=True);model.load_state_dict(initial['model']);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay']);optimizer.load_state_dict(initial['optimizer'])
        r0,m0=reference(case,0,pins);assert m0['reference_model_sha256']==tensor_digest(model);refs={0:r0};observer=SchemeBObserver(seed);fork=None
        orders=np.load(pin(pins,case/'batch_indices.npy'),allow_pickle=False);eps=torch.from_numpy(np.random.default_rng(seed_for('training-epsilon',ds,seed)).normal(size=(3000,64,dim)).astype(np.float32))
        probe_checks=signal_checks=frozen_checks=0
        def status():write(out/'progress.json',dict(status='running',dataset=ds,case=case.name,branch=branch,step=step,replayed_updates=count,planned_updates=3060,wall_seconds=time.monotonic()-start,updated_at=time.time()))
        for branch,lo,hi in [('changing',1,2030),('stable',1001,2030)]:
            folder=out/branch;folder.mkdir()
            if branch=='stable':
                state=torch.load(pin(pins,case/'checkpoint_fork.pt'),map_location='cpu',weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);observer=fork;model.train()
                assert tensor_digest(model)==state['model_sha256']
            for step in range(lo,hi+1):
                stage=0 if branch=='stable' else (step-1)//1000;ids=data['train'][stage][orders[step-1]];before=tensor_digest(model);old=original[branch][step]
                if before!=old['pre_model_sha256'] or hash_array(ids)!=old['input_ID_sha256']:raise ValueError(f'Exact model/input replay failed {branch} {step}')
                rng=torch.get_rng_state().clone()
                if selected(step):
                    with torch.random.fork_rng(devices=[]):pr,_=probe(model,assets,data,spec,seed,ds,step,stage);fr=frozen_probe(model,assets,data,heads,probe_noise,step,stage)
                    for row in pr:
                        expected=original_probes[branch].get((step,row['probe_group'],row['task'],row['readout']))
                        if expected is not None:
                            compare(expected,row,['accuracy','balanced_accuracy','macro_f1'],0.);compare(expected,row,['cross_entropy'],1e-10);assert float(expected['selected_l2'])==row['selected_l2'];probe_checks+=1
                    for row in fr:
                        expected=prior_frozen.get((branch,step,row['task'],row['readout']))
                        if expected is not None:compare(expected,row,['accuracy','balanced_accuracy'],0.);compare(expected,row,['cross_entropy'],1e-10);frozen_checks+=1
                    append(folder/'dense_probe.csv',pr);append(folder/'dense_frozen_probe.csv',fr)
                x=assets.read(stage,ids,stage);logits,mu,lv=forward(model,x,eps[step-1]);rec,mse,hat=entry.compute_reconstruction_terms(types.SimpleNamespace(dataset=ds),logits,x,likelihood);kl=models.kl_standard_normal(mu,lv);objective=rec+spec['beta_kl']*kl
                compare(old,dict(reconstruction=float(rec.detach()),pixel_mse=float(mse.detach()),kl=float(kl.detach()),objective=float(objective.detach())),['reconstruction','pixel_mse','kl','objective'],0.)
                legacy,_=observer((x-hat).detach(),stage);expected=original_old[branch][step];compare(expected,legacy,['xi'],1e-10)
                if legacy['baseline_source']!=expected['baseline_source']:raise ValueError('Legacy state routing changed')
                if selected(step):
                    append(folder/'dense_metrics.csv',[dict(step=step,updates_completed=step-1,stage=stage+1,model_sha256=before,reconstruction=float(rec.detach()),pixel_mse=float(mse.detach()),kl=float(kl.detach()),xi=legacy['xi'],eligible=legacy['eligible'],baseline_source=legacy['baseline_source'])])
                    raw=bank.get(stage,ids,stage)-reader(hat.detach())
                    for age,signal in refs.items():
                        row=dict(step=step,updates_completed=step-1,pre_model_sha256=before,reference_updates=age,reference_sha256=signal.reference_hash(),raw_residual_sha256=hash_array(raw),**signal.measure_window(raw))
                        e=original_signals[branch][age].get(step)
                        if e is not None:
                            if row['raw_residual_sha256']!=e['raw_residual_sha256']:raise ValueError('Original residual readout changed')
                            compare(e,row,['psi'],1e-10);signal_checks+=1
                        with (folder/f'psi_ref{age}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                if tensor_digest(model)!=before or not torch.equal(rng,torch.get_rng_state()) or head_hash(heads)!=hh:raise ValueError('Dense observation changed model/head/RNG')
                optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step();count+=1
                if branch=='changing' and step==1000:
                    state=torch.load(pin(pins,case/'checkpoint_fork.pt'),map_location='cpu',weights_only=True)
                    if tensor_digest(model)!=state['model_sha256']:raise ValueError('Fork model differs')
                    r1,m1=reference(case,1000,pins)
                    if tensor_digest(model)!=m1['reference_model_sha256']:raise ValueError('Reference1000 model differs')
                    refs[1000]=r1;fork=copy.deepcopy(observer)
                if step%100==0 or selected(step) and step%10==0:status()
            if tensor_digest(model)!=original[branch][hi+1]['pre_model_sha256']:raise ValueError('Replay endpoint differs')
            points=sorted({int(r['step']) for r in read({},folder/'dense_metrics.csv')});expected=[t for t in wanted() if lo<=t<=hi]
            if points!=expected:raise ValueError('Missing dense steps')
        if count!=3060 or probe_checks==0 or signal_checks==0 or frozen_checks==0:raise ValueError('Missing exact qualification checks')
        check_sources(pins);verify();reader.verify();result=dict(status='complete',dataset=ds,dim=dim,seed=seed,case=case.name,replayed_updates=count,new_treatment_updates=0,model_hashes_and_losses_verified_each_step=count,original_probe_rows_verified=probe_checks,original_signal_rows_verified=signal_checks,original_frozen_rows_verified=frozen_checks,changing_dense_steps=122,stable_dense_steps=91,measurement_timing='pre_update',source_files=pins,wall_seconds=time.monotonic()-start)
        write(out/'summary.json',result);write(out/'progress.json',result);bank.save();print('COMPLETE '+case.name,flush=True);return result
    except BaseException as e:
        write(out/'FAILED.json',dict(error=repr(e),branch=branch,step=step,traceback=traceback.format_exc()));raise
def main(ds,first=False):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);verify();source,entry,models=sources();assets=Assets(ds);reader=Reader(ds,SOURCE_PLAN);bank=Bank(assets,reader,LOW/'data_cache'/f'{ds}_observations.npz');bank.path=ROOT/'data_cache'/f'{ds}_observations.npz';done=[]
    try:
        for case,meta in cases(ds):
            done.append(run_case(ds,case,meta,entry,models,assets,reader,bank));write(ROOT/'reports'/f'worker_{ds}.json',dict(status='running',completed=len(done),planned=15,cases=done,updated_at=time.time()))
            if first:break
        source.verify();write(ROOT/'reports'/f'worker_{ds}.json',dict(status='qualified' if first else 'complete',completed=len(done),planned=15,cases=done,updated_at=time.time()))
    except BaseException as e:write(ROOT/'reports'/f'worker_{ds}.json',dict(status='failed',completed=len(done),planned=15,error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['dsprites','shapes3d','cifar'],required=True);a.add_argument('--first',action='store_true');s=a.parse_args();main(s.dataset,s.first)
