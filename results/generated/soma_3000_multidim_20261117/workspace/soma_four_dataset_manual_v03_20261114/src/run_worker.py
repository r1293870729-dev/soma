"""Launch directly in a terminal; never spawn from a torch-importing Python parent."""
import argparse,copy,json,os,time,traceback
import numpy as np
import torch
from common import ROOT,plan,sources,sha,write_json,Generator
from assets import Assets
from engine import Reader,Bank,run_parent,probe,model_for

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=['controlled','dsprites','shapes3d','cifar']);parser.add_argument('--run-id',required=True);parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):raise ValueError('Unsafe run id')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    p=plan();root=ROOT/'results/generated'/args.run_id
    root.mkdir(parents=True,exist_ok=True)
    source,entry,models=sources();assets=None if args.dataset=='controlled' else Assets(args.dataset)
    reader=Reader(args.dataset,p);bank=None if assets is None else Bank(assets,reader,ROOT/'data_cache'/f'{args.dataset}_observation_features.npz')
    if not args.fixture:
        frozen=json.loads((root/'frozen.json').read_text())
        if any(sha(ROOT/name)!=h for name,h in frozen['files'].items()):raise ValueError('Frozen source changed')
        if json.loads((ROOT/'reports/fixture_summary.json').read_text())['status']!='passed':raise ValueError('Fixtures required')
    results=[];failures=[];started=time.monotonic()
    if args.fixture:
        pp=copy.deepcopy(p);pp.update(reference_updates=20,white_rows=64,edge_windows=2,edge_cover_windows=2)
        generator=Generator('product_fourier',p['controlled_generator']) if assets is None else None
        dim=p['primary_latent_dims'][args.dataset];hashes=[]
        for enabled in [True,False]:
            folder=root/(args.dataset+('_monitor_on' if enabled else '_monitor_off'));folder.mkdir(exist_ok=False)
            result=run_parent(entry,models,assets,generator,10,dim,pp,folder,reader,bank,fixture=True,monitor_enabled=enabled)
            results.append(result);hashes.append(result['branches'][0]['final_model_sha256'])
        if hashes[0]!=hashes[1]:raise ValueError('Monitoring changed model trajectory')
        write_json(root/f'fixture_{args.dataset}.json',dict(status='passed',dataset=args.dataset,updates_per_arm=100,monitor_on_off_final_model_sha256=hashes,rows=results))
        print('Monitor on/off fixture PASSED:',args.dataset,hashes[0],flush=True)
        return
    families=p['controlled_families'] if assets is None else [args.dataset]
    for family in families:
        generator=Generator(family,p['controlled_generator']) if assets is None else None
        if generator is not None:
            moment=generator.moment_check(p['controlled_generator']['moment_check_grid'])
            if moment['mean_relative_error']>p['controlled_generator']['mean_error_limit'] or moment['covariance_relative_error']>p['controlled_generator']['covariance_relative_error_limit']:raise ValueError('Controlled moment preflight failed')
            write_json(root/f'generator_{family}.json',dict(metadata=generator.metadata,moment_check=moment))
        for seed in p['seeds']:
            dimensions=p['controlled_dims'] if assets is None else [p['primary_latent_dims'][args.dataset]]
            for dimension in dimensions:
                try:
                    result=run_parent(entry,models,assets,generator,seed,dimension,p,root,reader,bank)
                    results.append(result)
                except BaseException as exc:
                    failure=dict(dataset=args.dataset,family=family,seed=seed,latent_dim=dimension,error=repr(exc),traceback=traceback.format_exc());failures.append(failure)
                    write_json(root/f'FAILED_{family}_z{dimension}_seed{seed}.json',failure)
                    # A failed scientific unit remains in the planned denominator.
                    # Source/data integrity errors never trigger a replacement signal.
                    if isinstance(exc,(FileExistsError,KeyboardInterrupt)):raise
                write_json(root/f'worker_{args.dataset}.json',dict(status='running',runs=results,failures=failures,wall_seconds=time.monotonic()-started))
    source.verify();reader.verify()
    if any(sha(ROOT/name)!=h for name,h in frozen['files'].items()):raise ValueError('Frozen source changed')
    result=dict(status='complete' if not failures else 'complete_with_failed_units',runs=results,failures=failures,wall_seconds=time.monotonic()-started)
    write_json(root/f'worker_{args.dataset}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','failures']}),flush=True)

if __name__=='__main__':main()
