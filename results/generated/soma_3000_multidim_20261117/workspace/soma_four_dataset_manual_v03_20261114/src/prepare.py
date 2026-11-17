"""Offline identity/role preflight. No model outcomes choose data or parameters."""
import hashlib,json,time,zipfile,zlib
import numpy as np
from common import ROOT,OLD,plan,sha,write_json,seed_for,ROLES,ROLE,TASKS

def hashes(images):
    result=np.empty(len(images),dtype='V32')
    for i,img in enumerate(images):result[i]=np.void(hashlib.sha256(np.asarray(img,dtype=np.uint8).tobytes()).digest())
    return result

def source_inputs(p):
    old=json.loads((OLD/'configs/data_hashes.json').read_text());rows=[]
    for row in old['files']:
        path=OLD/row['path'];actual=sha(path)
        if actual!=row['sha256']:raise ValueError('Changed source dataset '+str(path))
        rows.append(dict(path=str(path),sha256=actual,bytes=path.stat().st_size))
    weight=ROOT/p['readout']['weights_path']
    if sha(weight)!=p['readout']['sha256']:raise ValueError('Readout weights changed')
    write_json(ROOT/'reports/source_inputs.json',dict(status='passed',files=rows,readout_sha256=sha(weight)))

def prepare_numeric():
    cache=ROOT/'data_cache';metadata={}
    from torchvision.datasets import CIFAR10
    source=CIFAR10(str(OLD/'data/cifar10'),train=True,download=False);test=CIFAR10(str(OLD/'data/cifar10'),train=False,download=False)
    paths={'dsprites':OLD/'results/assets/numeric_cache/dsprites_imgs.npy','shapes3d':cache/'shapes3d_imgs.npy'}
    if not (cache/'dsprites_hashes.npy').exists():
        with np.load(OLD/'data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',allow_pickle=False) as a:
            c=np.array(a['latents_classes'],dtype=np.uint8);angles=np.array(a['latents_values'][:,3])
        labels=np.column_stack([c[:,4],c[:,5],c[:,2],c[:,1],c[:,3],np.floor(np.mod(angles,2*np.pi)*8/(2*np.pi)).astype(np.uint8)])
        np.save(cache/'dsprites_labels.npy',labels);np.save(cache/'dsprites_factors.npy',c)
        images=np.load(paths['dsprites'],mmap_mode='r',allow_pickle=False)
        target=np.lib.format.open_memmap(cache/'dsprites_hashes.npy',mode='w+',dtype='V32',shape=(len(images),))
        for start in range(0,len(images),4096):target[start:start+4096]=hashes(images[start:start+4096])
        target.flush();del target
        # Reuse is allowed only after checking the numeric cache against the archive.
        with zipfile.ZipFile(OLD/'data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz') as z:expected=z.getinfo('imgs.npy').CRC
        crc=0
        with paths['dsprites'].open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):crc=zlib.crc32(b,crc)
        if crc!=expected:raise ValueError('dSprites numeric cache CRC mismatch')
        print('dSprites numeric identity cache complete',flush=True)
    if not (cache/'shapes3d_hashes.npy').exists():
        import h5py
        with h5py.File(OLD/'data/shapes3d/3dshapes.h5','r') as f:
            raw=np.asarray(f['labels']);c=np.stack([np.searchsorted(np.unique(raw[:,i]),raw[:,i]) for i in range(6)],1).astype(np.uint8)
            np.save(cache/'shapes3d_labels.npy',c)
            images=np.lib.format.open_memmap(paths['shapes3d'],mode='w+',dtype=np.uint8,shape=f['images'].shape)
            identities=np.lib.format.open_memmap(cache/'shapes3d_hashes.npy',mode='w+',dtype='V32',shape=(len(c),))
            for start in range(0,len(c),4096):
                block=np.array(f['images'][start:start+4096]);images[start:start+4096]=block;identities[start:start+4096]=hashes(block)
                if start%81920==0:print('Shapes3D numeric cache',start,'/',len(c),flush=True)
            images.flush();identities.flush();del images,identities
        print('Shapes3D numeric identity cache complete',flush=True)
    if not (cache/'cifar_hashes.npy').exists():
        base=np.concatenate([hashes(source.data),hashes(test.data)]);labels=np.concatenate([source.targets,test.targets]).astype(np.uint8)[:,None]
        np.save(cache/'cifar_labels.npy',labels)
        obs=[base[:50000]]
        for name in ['motion_blur','frost']:
            a=np.load(OLD/f'data/CIFAR-C/{name}.npy',mmap_mode='r',allow_pickle=False);obs.append(hashes(a[20000:30000]))
        lab=np.load(OLD/'data/CIFAR-C/labels.npy',mmap_mode='r',allow_pickle=False)
        if not np.array_equal(lab[20000:30000],np.asarray(test.targets)):raise ValueError('CIFAR-C original identity/labels mismatch')
        # Union original-image identity and exact pixel aliases over every domain.
        parent=np.arange(60000)
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
            return i
        def union(i,j):
            a,b=find(i),find(j)
            if a!=b:parent[max(a,b)]=min(a,b)
        seen={}
        for array,offset in [(base,0),(obs[1],50000),(obs[2],50000)]:
            for j,h in enumerate(array):
                key=h.tobytes();i=j+offset
                if key in seen:union(i,seen[key])
                else:seen[key]=i
        identity=np.array([base[find(i)] for i in range(60000)],dtype='V32')
        np.save(cache/'cifar_hashes.npy',identity)
        for i,h in enumerate(obs):np.save(cache/f'cifar_observation_hashes_{i}.npy',h)
        print('CIFAR identity aliases complete',flush=True)
    del source,test
    metadata['dsprites_images']=str(paths['dsprites']);metadata['shapes3d_images']=str(paths['shapes3d'])
    write_json(cache/'numeric_paths.json',metadata)

def pools(ds,labels):
    if ds=='dsprites':return [np.flatnonzero((labels[:,3]==0)&(labels[:,4]==0)),np.flatnonzero(labels[:,4]==0),np.arange(len(labels))]
    if ds=='shapes3d':return [np.flatnonzero((labels[:,4]==0)&(labels[:,5]==0)),np.flatnonzero(labels[:,5]==0),np.arange(len(labels))]
    return [np.arange(50000),np.arange(50000,60000),np.arange(50000,60000)]

def assign(ds,seed,p):
    cache=ROOT/'data_cache';labels=np.load(cache/f'{ds}_labels.npy',allow_pickle=False);identities=np.load(cache/f'{ds}_hashes.npy',mmap_mode='r',allow_pickle=False)
    pp=pools(ds,labels);_,first=np.unique(identities[pp[0]],return_index=True);representatives=pp[0][first]
    policy=dict(p['S1_source_roles'][ds]);fraction=policy.pop('remaining_train_fraction',None)
    if fraction is not None:
        remaining=len(representatives)-sum(policy.values());policy['train']=int(fraction*remaining);policy['validation']=remaining-policy['train']
    if sum(policy.values())>len(representatives):raise ValueError('Insufficient distinct S1 images: '+ds)
    # Rejection uses only pre-existing label support, never model/probe outcomes.
    for attempt in range(1000):
        rng=np.random.default_rng(seed_for('S1-global-roles',ds,seed,attempt));permutation=rng.permutation(representatives);rolemap={};offset=0;valid=True
        for role,count in policy.items():
            selected=permutation[offset:offset+count];offset+=count
            if role.startswith('probe-') and not all(len(np.unique(labels[selected,c]))==n for _,c,n in TASKS[ds][0]):valid=False;break
            rolemap.update((identities[int(i)].tobytes(),ROLE[role]) for i in selected)
        if valid:break
    else:raise ValueError('Cannot allocate S1 class-complete roles')
    fractions=p['new_image_roles_fractions'];names=list(fractions);cuts=np.cumsum(list(fractions.values()))
    code=np.zeros(len(labels),dtype=np.uint8);computed={}
    for i,h in enumerate(identities):
        key=h.tobytes()
        if key not in computed:
            if key in rolemap:value=rolemap[key]
            else:
                u=int.from_bytes(hashlib.sha256(key+str(seed).encode()).digest()[:8],'little')/2**64
                value=ROLE[names[min(int(np.searchsorted(cuts,u,side='right')),len(names)-1)]]
            computed[key]=value
        code[i]=computed[key]
    # Unused S1 identities remain unused even when aliased in later phases.
    unassigned={identities[int(i)].tobytes() for i in representatives}-set(rolemap)
    if unassigned:
        for i,h in enumerate(identities):
            if h.tobytes() in unassigned:code[i]=0
    np.save(cache/f'{ds}_seed{seed}_roles.npy',code)
    record=dict(dataset=ds,seed=seed,S1_original_rows=len(pp[0]),S1_distinct_pixel_sources=len(representatives),S1_exact_role_counts=policy,
        role_assignment_sha256=hashlib.sha256(code.tobytes()).hexdigest(),S1_class_support_rejection_attempt=attempt,
        stage_rows=list(map(len,pp)),cross_role_physical_image_overlap=0,stage_role_counts=[])
    for stage,pool in enumerate(pp):
        record['stage_role_counts'].append(dict(stage=stage+1,counts={role:dict(rows=int(np.sum(code[pool]==number)),unique_pixel_sources=int(len(np.unique(identities[pool[code[pool]==number]])))) for role,number in ROLE.items()}))
    write_json(ROOT/f'reports/{ds}_seed{seed}_roles.json',record)
    print(ds,'seed',seed,'roles ready; S1 unique',len(representatives),flush=True)
    return record

def main():
    p=plan();started=time.monotonic();source_inputs(p);prepare_numeric();records=[]
    for ds in ['dsprites','shapes3d','cifar']:
        for seed in p['seeds']:records.append(assign(ds,seed,p))
    cache_files=[f for f in sorted((ROOT/'data_cache').glob('*.npy')) if f.name!='shapes3d_imgs.npy']
    manifest={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in cache_files}
    # The large derived image cache has a full recorded digest and is opened read-only.
    f=ROOT/'data_cache/shapes3d_imgs.npy';manifest[f.name]=dict(bytes=f.stat().st_size,sha256=sha(f))
    write_json(ROOT/'reports/data_preflight.json',dict(status='passed',records=records,numeric_cache_manifest=manifest,wall_seconds=time.monotonic()-started,
        labels_choose_signal=False,physical_identity_role_isolation=True,final_test_unlocked=False))
    print('All image identity/role preflight checks passed',flush=True)

if __name__=='__main__':main()
