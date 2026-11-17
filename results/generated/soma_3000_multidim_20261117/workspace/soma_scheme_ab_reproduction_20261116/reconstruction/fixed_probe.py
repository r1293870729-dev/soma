"""Offline paired data-view diagnostic; never feeds back into historical training."""
import argparse,csv,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from runtime import ROOT,load_runner,write,digest

def collect(runner,args,model,phase_index):
    embedding=runner.make_orthogonal_embedding(args.observation_dim,args.intrinsic_dim,args.seed+123)
    data=runner.ControlledPhaseDataset(phase=['s1','t2','local_split'][phase_index],embedding=embedding,size=args.controlled_stage_size,noise_std=args.controlled_noise_std,seed=args.seed+phase_index*101,factor_bins=args.factor_bins,split_offset=args.split_offset,local_probability=args.local_probability,split_mode=args.split_mode)
    loader=DataLoader(data,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=0,generator=runner.make_torch_generator(888888))
    features=[];labels=[];seen=0;model.eval()
    with torch.no_grad():
        for x,y in loader:
            if seen>=args.online_probe_max_samples:break
            take=min(len(x),args.online_probe_max_samples-seen);mu,_=model.encode(x[:take]);features.append(mu);labels.append(y[:take,0]);seen+=take
    return torch.cat(features),torch.cat(labels)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dim',type=int,required=True);p.add_argument('--mode',required=True);p.add_argument('--run-id',default='reconstructed_B_001');p.add_argument('--fixture',action='store_true');a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    runner=load_runner();out=ROOT/'results'/a.run_id/f'B_z{a.dim}_{a.mode}';execution=json.loads((out/'execution.json').read_text());checkpoints=[5] if a.fixture else [900,1100,1900,2100,2900];anchor=5 if a.fixture else 900;fixed_seed=10000+anchor;rows=[];started=time.monotonic()
    history=list(csv.DictReader((out/'probe_history.csv').open()))
    for t in checkpoints:
        state=torch.load(out/'snapshots'/f'step_{t:04d}.pt',map_location='cpu',weights_only=True);args=argparse.Namespace(**state['args'])
        with torch.random.fork_rng(devices=[]):model=runner.MLPVAE(args.observation_dim,args.controlled_latent_dim,args.controlled_hidden_dim,args.controlled_hidden_layers)
        model.load_state_dict(state['model_state']);before=digest(model);stage_index=['s1','t2','local_split'].index(state['stage']);scores={}
        for view,index in [('fixed_S1',0),('current_stage',stage_index)]:
            x,y=collect(runner,args,model,index);acc,losses=runner.train_online_linear_probe(x,y,args.factor_bins,seed=fixed_seed,train_steps=args.online_probe_train_steps,patience=args.online_probe_early_stopping_patience,min_delta=args.online_probe_early_stopping_min_delta)
            row=dict(step=t,view=view,actual_input_stage=['s1','t2','local_split'][index],accuracy=acc,fixed_head_seed=fixed_seed,samples=len(x),epochs=len(losses),evaluation='historical first2048 training-pool rows, fixed random80/20 and historical fitter; not independent final test')
            rows.append(row);scores[view]=acc;write(out/'fixed_probe_progress.json',dict(status='running',step=t,view=view,rows=len(rows),planned_rows=len(checkpoints)*2,updated_at=time.time()))
            with (out/'fixed_probe_loss_history.jsonl').open('a') as f:
                for loss in losses:f.write(json.dumps(dict(step=t,view=view,**loss))+'\n')
        if digest(model)!=before:raise ValueError('Offline probe changed encoder')
        if t==anchor:
            old=next(float(r['accuracy']) for r in history if int(r['step'])==anchor and r['task']=='theta')
            if abs(scores['fixed_S1']-old)>1e-7 or scores['fixed_S1']!=scores['current_stage']:raise ValueError('Pre-transition probe round-trip failed')
        print(f'fixed-view probes z={a.dim} {a.mode} at{t}: '+str(scores),flush=True)
    with (out/'fixed_probe_diagnostic.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'fixed_probe_summary.json',dict(status='complete',rows=rows,anchor_roundtrip_passed=True,offline_only=True,same_classifier_and_seed_between_views=True,untouched_test=False,wall_seconds=time.monotonic()-started))
    write(out/'fixed_probe_progress.json',dict(status='complete',rows=len(rows),updated_at=time.time()))

if __name__=='__main__':main()
