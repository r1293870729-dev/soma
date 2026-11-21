"""Readonly S1 head replay, same pixels/noise; no main-model update or head refit."""
import argparse,copy,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from study_paths import ROOT,PLAN,OLD,sha,write,rows,append,verify
from study_runtime import sources,tensor_digest,seed_for,model_for,Assets,prepare_image,metrics,Reader,fit_head

def head_hash(heads):
    h=hashlib.sha256()
    for name,v in sorted(heads.items()):
        h.update(name.encode())
        for k in ['weight','bias','mean','std']:h.update(v[k].contiguous().numpy().tobytes())
        h.update(json.dumps(v['classes']).encode())
    return h.hexdigest()
def pin(mapping,p):p=Path(p);mapping[str(p)]=sha(p);return p
def available():
    result=[]
    for home in [OLD/'results/all_3000_001',ROOT/'results'/PLAN['run_id']]:
        for p in home.glob('*/summary.json'):
            v=json.loads(p.read_text())
            if v.get('status')=='complete' and v.get('dataset') in PLAN['image_dimensions']:result.append((p.parent,v))
    order={'cifar':0,'dsprites':1,'shapes3d':2}
    return sorted(result,key=lambda a:(order[a[1]['dataset']],a[1]['seed'],a[1]['latent_dim']))
class Context:
    def __init__(self):self.source,self.entry,self.models=sources();self.assets={};self.data={};self.readers={}
    def get(self,ds,seed):
        if ds not in self.assets:self.assets[ds]=Assets(ds)
        asset=self.assets[ds];key=(ds,seed)
        if key not in self.data:
            folder=ROOT/'results/replay_inputs'/f'{ds}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);data=prepare_image(asset,seed,PLAN,folder)
            for g in data['groups']:g['first_step']=1+sum(PLAN['stage_steps'][:g['stage']])
            group=next(g for g in data['groups'] if g['name']=='S1');ids=group['roles']['probe-test'];x=asset.read(0,ids,0);labels=asset.labels[ids];self.data[key]=(data,group,x,labels)
        return asset,self.data[key]

def input_control(ctx,ds,seed):
    folder=ROOT/'results/input_control'/f'{ds}_seed{seed}';summary=folder/'summary.json'
    if summary.exists():return
    folder.mkdir(parents=True,exist_ok=True);asset,(data,group,_,_)=ctx.get(ds,seed);reader=ctx.readers.setdefault(ds,Reader(ds,PLAN)) if ds not in ctx.readers else ctx.readers[ds];features={};labels={}
    for role,ids in group['roles'].items():features[role]=torch.from_numpy(reader(asset.read(0,ids,0))).double();labels[role]=asset.labels[ids]
    result=[];heads={}
    for task in group['tasks']:
        y={role:torch.from_numpy(v[:,task['column']]) for role,v in labels.items()};m,h=fit_head(features,y,list(range(task['classes'])),PLAN['image_probe_head']);m.pop('candidates');result.append(dict(dataset=ds,seed=seed,task=task['name'],classes=task['classes'],chance=1/task['classes'],readout='supervised_pretrained_ResNet18_hx',**m));heads[task['name']]=h
    append(folder/'scores.csv',result);torch.save(heads,folder/'heads.pt');reader.verify();write(summary,dict(status='complete',dataset=ds,seed=seed,diagnostic_only=True,pretrained_external_supervision=True,same_probe_roles_and_fitter=True,no_main_model_feedback=True,readout_model_sha256=reader.frozen_hash))

def replay(ctx,case,meta,fixture=False):
    ds,seed,dim=meta['dataset'],meta['seed'],meta['latent_dim'];name=case.name;out=ROOT/'results'/('frozen_fixture' if fixture else 'probe_replay')/name
    if (out/'summary.json').exists():return
    out.mkdir(parents=True,exist_ok=False);pins={};anchor=PLAN['frozen_head_anchor_arrival'];asset,(data,group,x,y)=ctx.get(ds,seed)
    with np.load(case/'prepared/source_ids.npz',allow_pickle=False) as a,np.load(ROOT/'results/replay_inputs'/f'{ds}_seed{seed}'/'source_ids.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Replay source roles differ')
    source=rows(pin(pins,case/'changing/probe_history.csv'));expected={(r['task'],r['readout']):r for r in source if int(r['step'])==anchor and r['probe_group']=='S1'}
    checkpoint=pin(pins,case/f'changing/snapshots/pre_{anchor:04d}.pt');state=torch.load(checkpoint,map_location='cpu',weights_only=True);heads={k:v for k,v in state['heads'].items() if k.startswith('S1:')};frozen_hash=head_hash(heads)
    if len(heads)!=2*len(group['tasks']):raise ValueError('Incomplete S1 frozen head set')
    model,_=model_for(ctx.entry,ds,seed,dim,ds);model.eval();epsilon=torch.from_numpy(np.random.default_rng(seed_for('probe-noise',ds,seed,'S1','probe-test')).normal(size=(len(x),dim)).astype(np.float32));records=[];checks=0;started=time.monotonic()
    try:
        changing_prefix=[]
        for branch in ['changing','stable']:
            if branch=='stable':
                for r in changing_prefix:records.append(dict(r,branch='stable',shared_prefix=True))
            snapshots=sorted((case/branch/'snapshots').glob('pre_*.pt'))
            for file in snapshots:
                t=int(file.stem.split('_')[1])
                if t<anchor or fixture and t not in [anchor,1000,1001,2000,2001,3000]:continue
                snap=torch.load(pin(pins,file),map_location='cpu',weights_only=True);model.load_state_dict(snap['model']);pre=tensor_digest(model)
                if pre!=snap['pre_model_sha256']:raise ValueError('Snapshot model digest differs')
                rng=torch.get_rng_state().clone();mus=[];lvs=[]
                with torch.no_grad():
                    for start in range(0,len(x),64):mu,lv=model.encode(x[start:start+64]);mus.append(mu);lvs.append(lv)
                    mu,lv=torch.cat(mus),torch.cat(lvs);features={'mu':mu.double(),'sampled_z':(mu+torch.exp(.5*lv)*epsilon).double()}
                    for task in group['tasks']:
                        labels=torch.from_numpy(y[:,task['column']]).long()
                        for readout in ['mu','sampled_z']:
                            h=heads[f'S1:{task["name"]}:{readout}'];logits=(features[readout]-h['mean'])/h['std']@h['weight']+h['bias'];m=metrics(logits,labels,len(h['classes']))
                            if branch=='changing' and t==anchor:
                                e=expected[task['name'],readout]
                                if abs(m['balanced_accuracy']-float(e['balanced_accuracy']))>1e-12 or abs(m['cross_entropy']-float(e['cross_entropy']))>1e-12:raise ValueError('Frozen anchor fails exact original-probe replay')
                                checks+=1
                            r=dict(step=t,updates_completed=t-1,branch=branch,shared_prefix=False,probe_group='S1',probe_stage=1,task=task['name'],readout=readout,head_kind='frozen_pre900',head_anchor_step=anchor,classes=len(h['classes']),samples_test=len(labels),head_sha256=frozen_hash,pre_model_sha256=pre,mu_variance=float(mu.var(0,unbiased=False).mean()),posterior_noise_variance=float(lv.exp().mean()),**m);records.append(r)
                            if branch=='changing' and t<=1000:changing_prefix.append(r)
                if tensor_digest(model)!=pre or not torch.equal(rng,torch.get_rng_state()) or head_hash(heads)!=frozen_hash:raise ValueError('Frozen replay changed model/head/RNG')
                write(out/'progress.json',dict(status='running',case=name,branch=branch,step=t,updated_at=time.time()))
        append(out/'frozen_probe.csv',records)
        if checks!=len(heads):raise ValueError('Missing anchor replay checks')
        if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Read-only source changed')
        write(out/'source_inputs.json',pins);write(out/'summary.json',dict(status='fixture_complete' if fixture else 'complete',dataset=ds,seed=seed,latent_dim=dim,anchor=anchor,causal_first_step=anchor,head_sha256=frozen_hash,original_anchor_metrics_exact=checks,rows=len(records),main_model_updates=0,source_immutable=True,source_case=str(case),wall_seconds=time.monotonic()-started));print('frozen replay complete '+name,flush=True)
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

def symmetry_audit(ctx):
    target=ROOT/'reports/dsprites_label_identifiability.json'
    if target.exists():return
    asset,_=ctx.get('dsprites',10);ids=np.asarray(asset.identities);labels=asset.labels;result=[]
    for task,col,k,mask in [('raw40_all_shapes',4,40,np.ones(len(labels),bool)),('heart_bin8',5,8,labels[:,3]==2)]:
        a=ids[mask];y=labels[mask,col].astype(np.int64);_,inverse=np.unique(a,return_inverse=True);pairs,count=np.unique(inverse*k+y,return_counts=True);support=np.bincount(y,minlength=k);best=np.zeros(int(inverse.max())+1);np.maximum.at(best,pairs//k,count/support[pairs%k]);per_group=np.bincount(pairs//k,minlength=len(best))
        result.append(dict(task=task,rows=len(y),classes=k,pixel_identity_groups=len(best),groups_with_conflicting_labels=int((per_group>1).sum()),ideal_pixel_classifier_BA_upper_bound=float(best.sum()/k),chance=1/k))
    write(target,dict(status='complete',offline_full_dataset_label_identifiability_audit=True,no_training_feedback=True,results=result))

def main(fixture=False):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);verify();ctx=Context()
    if fixture:
        case,meta=next((p,m) for p,m in available() if m['dataset']=='cifar' and m['latent_dim']==16 and m['seed']==10);replay(ctx,case,meta,True);write(ROOT/'reports/frozen_probe_qualification.json',dict(status='pass',original_anchor_metrics_exact=True,causal_no_pre_anchor_points=True,no_model_or_rng_update=True));return
    done=set();controls=set()
    try:
        symmetry_audit(ctx)
        while True:
            candidates=available()
            for case,meta in candidates:
                if case.name in done:continue
                replay(ctx,case,meta);done.add(case.name);key=(meta['dataset'],meta['seed'])
                if key not in controls:input_control(ctx,*key);controls.add(key)
                write(ROOT/'reports/frozen_worker.json',dict(status='running',completed=len(done),planned=45,input_controls=len(controls),updated_at=time.time()))
            if len(done)==45:break
            time.sleep(20)
        ctx.source.verify();verify();write(ROOT/'reports/frozen_worker.json',dict(status='complete',completed=45,planned=45,input_controls=len(controls)))
    except BaseException as e:write(ROOT/'reports/frozen_worker.json',dict(status='failed',completed=len(done),planned=45,error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',action='store_true');a=p.parse_args();main(a.fixture)
