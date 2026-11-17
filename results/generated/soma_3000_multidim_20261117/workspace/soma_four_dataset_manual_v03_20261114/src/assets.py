"""Stage-gated immutable data access and fixed physical-source probe roles."""
from pathlib import Path
import copy,hashlib,json
import numpy as np
import torch
from common import ROOT,OLD,ROLE,ROLES,TASKS,seed_for,hash_array,sample_indices,write_json
from prepare import pools

class Assets:
    def __init__(self,dataset):
        self.dataset=dataset;cache=ROOT/'data_cache'
        self.labels=np.load(cache/f'{dataset}_labels.npy',allow_pickle=False)
        self.identities=np.load(cache/f'{dataset}_hashes.npy',mmap_mode='r',allow_pickle=False)
        self.pools=pools(dataset,self.labels);self.images=None
        paths=json.loads((cache/'numeric_paths.json').read_text())
        if dataset in ['dsprites','shapes3d']:
            self.images=np.load(paths[dataset+'_images'],mmap_mode='r',allow_pickle=False)
        else:
            from torchvision.datasets import CIFAR10
            source=CIFAR10(str(OLD/'data/cifar10'),train=True,download=False)
            self.images=[source.data,np.load(OLD/'data/CIFAR-C/motion_blur.npy',mmap_mode='r',allow_pickle=False),np.load(OLD/'data/CIFAR-C/frost.npy',mmap_mode='r',allow_pickle=False)]
            self.observation_hashes=[np.load(cache/f'cifar_observation_hashes_{i}.npy',mmap_mode='r',allow_pickle=False) for i in range(3)]

    def read(self,stage,ids,arrived):
        if stage>arrived:raise ValueError('Future image access is forbidden')
        ids=np.asarray(ids,dtype=np.int64)
        if self.dataset=='cifar':
            array=self.images[stage][ids if stage==0 else 20000+ids-50000]
        else:array=self.images[ids]
        x=torch.from_numpy(np.array(array,copy=True)).float()
        if self.dataset=='dsprites':return x.unsqueeze(1)
        return x.permute(0,3,1,2).contiguous()/255

    def obs_keys(self,stage,ids,arrived):
        if stage>arrived:raise ValueError('Future observation cache access')
        ids=np.asarray(ids,dtype=np.int64)
        if self.dataset=='cifar':return self.observation_hashes[stage][ids if stage==0 else ids-50000]
        return self.identities[ids]

    def identity_strings(self,ids):return [self.dataset+':'+h.tobytes().hex() for h in self.identities[ids]]

def prepare_image(assets,seed,p,out):
    ds=assets.dataset;roles=np.load(ROOT/f'data_cache/{ds}_seed{seed}_roles.npy',allow_pickle=False)
    sizes=p['image_probe_rows'][ds];prepared=dict(dataset=ds,seed=seed,train={},validation={},calibration={},groups=[],manifest={})
    manifest=dict(dataset=ds,seed=seed,stage_roles=[],calibration={},physical_source_roles_disjoint=True,identity_assignment_sha256=hashlib.sha256(roles.tobytes()).hexdigest())
    def select(stage,role,count,tasks,mask=None,unique=True):
        pool=assets.pools[stage];pool=pool[roles[pool]==ROLE[role]]
        if mask is not None:pool=pool[mask[pool]]
        rng=np.random.default_rng(seed_for('image-role-sample',ds,seed,stage,role,'none' if mask is None else 'conditional',tuple(tasks)))
        return sample_indices(pool,count,rng,assets.labels,tasks,assets.identities if unique else None)
    for stage in range(3):
        rec={}
        for role,key in [('train','train'),('validation','validation')]:
            pool=assets.pools[stage];pool=pool[roles[pool]==ROLE[role]]
            rng=np.random.default_rng(seed_for('main-data',ds,seed,stage,role));count=p['training_rows_per_stage' if role=='train' else 'validation_rows_per_stage']
            ids=rng.choice(pool,count,replace=count>len(pool));prepared[key][stage]=ids
            rec[role]=dict(rows=len(ids),distinct_source_images=len(np.unique(assets.identities[ids])),pool_original_rows=len(pool))
        tasks=TASKS[ds][stage]
        group=dict(name=f'S{stage+1}',stage=stage,first_step=[1,2001,3001][stage],tasks=[dict(name=n,column=c,classes=k,primary=True) for n,c,k in tasks],roles={})
        for role,count in zip(['probe-train','probe-val','probe-test'],sizes):group['roles'][role]=select(stage,role,count,tasks)
        prepared['groups'].append(group);manifest['stage_roles'].append(dict(stage=stage+1,**rec))
    if ds in ['dsprites','shapes3d']:
        stage=2;shape_col=3 if ds=='dsprites' else 4;shape=2 if ds=='dsprites' else 0
        col,classes,name=(5,8,'orientation_shape2_bin8') if ds=='dsprites' else (5,15,'orientation_shape0')
        condition=assets.labels[:,shape_col]==shape
        group=dict(name='S3_observable_orientation',stage=2,first_step=3001,tasks=[dict(name=name,column=col,classes=classes,primary=True)],roles={})
        for role,count in zip(['probe-train','probe-val','probe-test'],sizes):group['roles'][role]=select(2,role,count,[(name,col,classes)],mask=condition)
        prepared['groups'].append(group)
        col,classes=(4,40) if ds=='dsprites' else (5,15)
        aux=dict(name='S3_raw_orientation',stage=2,first_step=3001,tasks=[dict(name='orientation_raw',column=col,classes=classes,primary=False)],roles={})
        for role,count in zip(['probe-train','probe-val','probe-test'],sizes):aux['roles'][role]=select(2,role,count,[('orientation_raw',col,classes)],unique=False)
        prepared['groups'].append(aux)
    for role,count in [('white',p['white_rows']),('edge',p['edge_windows']*64),('edge-cover',p['edge_cover_windows']*64)]:
        ids=select(0,role,count,[],unique=True);prepared['calibration'][role]=ids
        manifest['calibration'][role]=dict(rows=count,distinct_physical_sources=len(np.unique(assets.identities[ids])),ids=assets.identity_strings(ids))
    all_sets={}
    for role in ['train','validation']:
        all_sets[role]=set(h.tobytes() for ids in prepared[role].values() for h in assets.identities[ids])
    for role in ['probe-train','probe-val','probe-test']:
        all_sets[role]=set(h.tobytes() for group in prepared['groups'] for h in assets.identities[group['roles'][role]])
    for role,ids in prepared['calibration'].items():all_sets[role]=set(h.tobytes() for h in assets.identities[ids])
    for i,(name,a) in enumerate(all_sets.items()):
        for other,b in list(all_sets.items())[i+1:]:
            if a&b:raise ValueError('Physical image leakage '+name+' '+other)
    manifest['probe_groups']=[dict(name=g['name'],stage=g['stage']+1,tasks=g['tasks'],roles={r:dict(rows=len(ids),unique_physical=len(np.unique(assets.identities[ids])),ids_sha256=hash_array(ids),class_counts={t['name']:np.bincount(assets.labels[ids,t['column']],minlength=t['classes']).tolist() for t in g['tasks']}) for r,ids in g['roles'].items()}) for g in prepared['groups']]
    prepared['manifest']=manifest;arrays={}
    for key in ['train','validation']:
        for stage,ids in prepared[key].items():arrays[f'{key}_{stage}']=ids
    for role,ids in prepared['calibration'].items():arrays['cal_'+role]=ids
    for group in prepared['groups']:
        for role,ids in group['roles'].items():arrays[group['name']+'_'+role]=ids
    np.savez_compressed(out/'source_ids.npz',**arrays);write_json(out/'roles.json',manifest)
    return prepared

def prepare_controlled(generator,seed,p,out):
    prepared=dict(dataset='controlled',family=generator.family,seed=seed,train={},validation={},calibration={},groups=[],manifest={})
    identity={}
    for stage in range(3):
        for key,role in [('train','train'),('validation','validation')]:
            data=generator.role(seed,role,stage,p['training_rows_per_stage']);prepared[key][stage]=data;identity[role+str(stage)]=data['ids']
    for task,stage,branch in [('theta',0,False),('phi',1,False),('branch',2,True)]:
        group=dict(name=f'S{stage+1}',stage=stage,first_step=[1,2001,3001][stage],tasks=[dict(name=task,column={'theta':0,'phi':1,'branch':2}[task],classes=4 if branch else 8,primary=True)],roles={})
        for role,count in zip(['probe-train','probe-val','probe-test'],p['controlled_probe_rows']):
            data=generator.role(seed,task+'-'+role,stage,count,branch_only=branch);group['roles'][role]=data;identity[task+'-'+role]=data['ids']
        prepared['groups'].append(group)
    for role,count in [('white',p['white_rows']),('edge',p['edge_windows']*64),('edge-cover',p['edge_cover_windows']*64)]:
        data=generator.role(seed,role,0,count);prepared['calibration'][role]=data;identity[role]=data['ids']
    seen=set()
    for key,ids in identity.items():
        if seen.intersection(ids) or len(ids)!=len(set(ids)):raise ValueError('Controlled role overlap')
        seen.update(ids)
    prepared['manifest']=dict(dataset='controlled',family=generator.family,seed=seed,roles=identity,disjoint=True,
        training_hashes={str(s):d['observation_sha256'] for s,d in prepared['train'].items()},generator=generator.metadata)
    write_json(out/'roles.json',prepared['manifest']);return prepared
