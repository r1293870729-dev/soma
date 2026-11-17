"""Full-size role/class/observability checks before freezing the training matrix."""
import json,time
import numpy as np
import torch
from common import ROOT,plan,write_json,Generator
from assets import Assets,prepare_image
from probes import fit_head

def main():
    torch.set_num_threads(2);p=plan();started=time.monotonic();records=[];ambiguity=[];generator_checks=[]
    for ds in ['dsprites','shapes3d','cifar']:
        assets=Assets(ds)
        for seed in p['seeds']:
            out=ROOT/'reports/prepared'/f'{ds}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
            data=prepare_image(assets,seed,p,out);records.append(data['manifest'])
            print('Prepared all full-size probe and calibration roles:',ds,seed,flush=True)
        if ds in ['dsprites','shapes3d']:
            shape_col=3 if ds=='dsprites' else 4;shape=2 if ds=='dsprites' else 0;labelcol=5
            ids=np.flatnonzero(assets.labels[:,shape_col]==shape)
            keys,inv=np.unique(assets.identities[ids],return_inverse=True)
            minimum=np.full(len(keys),255,dtype=np.uint8);maximum=np.zeros(len(keys),dtype=np.uint8)
            np.minimum.at(minimum,inv,assets.labels[ids,labelcol]);np.maximum.at(maximum,inv,assets.labels[ids,labelcol])
            ambiguity.append(dict(dataset=ds,conditional_shape=shape,target='modulo_2pi_bin8' if ds=='dsprites' else 'original_orientation15',source_rows=len(ids),distinct_pixels=len(keys),ambiguous_pixel_groups=int(np.sum(minimum!=maximum)),
                policy='all planned probe outcomes retained; aliases are disclosed, no outcome-selected task replacement'))
        del assets
    for family in p['controlled_families']:
        gen=Generator(family,p['controlled_generator']);mom=gen.moment_check(p['controlled_generator']['moment_check_grid'])
        if mom['mean_relative_error']>p['controlled_generator']['mean_error_limit'] or mom['covariance_relative_error']>p['controlled_generator']['covariance_relative_error_limit']:raise ValueError('Generator moment mismatch')
        for seed in p['seeds']:
            features={};labels={}
            for role,n in [('probe-train',2048),('probe-val',1024),('probe-test',1024)]:
                data=gen.role(seed,'raw-qualification-'+role,2,n,branch_only=True);features[role]=torch.from_numpy(data['x']).double();labels[role]=torch.from_numpy(data['labels'][:,2])
            result,_=fit_head(features,labels,[1,2,3,4],p['controlled_probe_head'])
            if not .5<=result['balanced_accuracy']<=1:raise ValueError('Raw branch task qualification failed')
            generator_checks.append(dict(family=family,seed=seed,moment=mom,raw_branch_BA=result['balanced_accuracy'],offline_generator_qualification_only=True))
            print('Generator moment/branch qualification:',family,seed,result['balanced_accuracy'],flush=True)
    write_json(ROOT/'reports/asset_qualification.json',dict(status='passed',image_roles=records,orientation_ambiguity=ambiguity,generator_checks=generator_checks,wall_seconds=time.monotonic()-started,model_outcomes_used_to_select_protocol=False))

if __name__=='__main__':main()
