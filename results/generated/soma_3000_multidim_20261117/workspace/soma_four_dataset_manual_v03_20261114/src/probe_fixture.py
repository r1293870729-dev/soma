import argparse,json
import torch
from common import ROOT,plan,sources,tensor_digest,write_json,Generator
from assets import Assets,prepare_image,prepare_controlled
from engine import model_for,probe

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);args=parser.parse_args();torch.set_num_threads(2)
    p=plan();_,entry,_=sources();assets=None if args.dataset=='controlled' else Assets(args.dataset)
    family='product_fourier' if assets is None else args.dataset;dim=p['primary_latent_dims'][args.dataset]
    root=ROOT/'reports/probe_fixture'/args.dataset;root.mkdir(parents=True,exist_ok=True)
    data=prepare_controlled(Generator(family,p['controlled_generator']),10,p,root) if assets is None else prepare_image(assets,10,p,root)
    model,_=model_for(entry,args.dataset,10,dim,family)
    file=ROOT/'results/generated/fixture_001'/(args.dataset+'_monitor_on')/f'{family}_z{dim}_seed10/changing/checkpoint_final.pt'
    model.load_state_dict(torch.load(file,map_location='cpu',weights_only=True)['model']);before=tensor_digest(model);rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):rows,heads=probe(model,assets,data,p,10,family,100,0)
    if before!=tensor_digest(model) or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Probe mutated main model/RNG')
    if any(row['probe_stage']!=1 for row in rows):raise ValueError('Future probe leaked')
    write_json(ROOT/f'reports/probe_fixture_{args.dataset}.json',dict(status='passed',no_model_or_rng_mutation=True,no_future_tasks=True,rows=rows))
    print('Probe path fixture passed:',args.dataset,len(rows),flush=True)

if __name__=='__main__':main()
