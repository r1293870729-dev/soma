"""One fresh process per case. No alteration to the parallel image project."""
import argparse,json,sys,time,traceback
from pathlib import Path
import torch
from runtime import ROOT,load_runner,case_args,sha,digest,write

def run(dim,mode,run_id,fixture=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);runner=load_runner();out=ROOT/'results'/run_id/f'B_z{dim}_{mode}';out.mkdir(parents=True,exist_ok=False);(out/'snapshots').mkdir()
    args,archived,extra=case_args(runner,dim,mode,out)
    if fixture:
        args.controlled_stage_steps=[25,25,25];args.pw_calibration_warmup_steps=10;args.pw_calibration_steps=10;args.online_probe_start_offset=5;args.online_probe_every=20;args.online_probe_train_steps=20;args.online_probe_early_stopping_patience=3
    else:
        frozen=json.loads((ROOT/'results'/run_id/'frozen.json').read_text())
        if any(sha(ROOT/n)!=h for n,h in frozen['files'].items()):raise ValueError('Frozen reconstruction source changed')
    write(out/'reconstruction_args.json',dict(actual=runner.vars_for_json(args),archived=archived,added_later_defaults=runner.vars_for_json(argparse.Namespace(**extra)),restored_math=['un-normalized whitened covariance','absolute largest-eigenvalue excess','MP denominator W'],provenance='protocol reconstruction from report and archived args, original July source unavailable'))
    runner.set_seed(args.seed);device=torch.device('cpu');started=time.monotonic();phase='calibration';step=0;calibration_rows=0
    def status(message):write(out/'progress.json',dict(status='running',phase=phase,step=step,calibration_rows=calibration_rows,message=message,updated_at=time.time(),wall_seconds=time.monotonic()-started))
    original_append=runner.append_csv
    def observed_append(path,row):
        nonlocal step,calibration_rows
        original_append(path,row)
        path=Path(path)
        if path.name.startswith('calibration_'):
            calibration_rows+=1
            if int(row['step'])%100==0:status(f'{path.stem} {row["step"]}')
        elif path.name=='train_history.csv':
            step=int(row['step'])
            if step%100==0 or step in [1,1001,2001]:
                msg=f'B z={dim} {mode} {step}: rec={row["reconstruction"]:.6g} Xi={row["xi_pw"]:.6g}'
                status(msg);print(msg,flush=True)
    runner.append_csv=observed_append
    original_probe=runner.run_online_controlled_probe
    def observed_probe(*pa,**kw):
        result=original_probe(*pa,**kw);model=kw['model'];t=kw['global_step'];before=torch.get_rng_state().clone();h=digest(model)
        torch.save(dict(model_state=model.state_dict(),step=t,stage=kw['stage'].name,args=runner.vars_for_json(args),model_sha256=h),out/'snapshots'/f'step_{t:04d}.pt')
        if not torch.equal(before,torch.get_rng_state()) or digest(model)!=h:raise ValueError('Snapshot changed model/RNG')
        return result
    runner.run_online_controlled_probe=observed_probe
    try:
        status('starting archived 3-stage calibration workflow');cal=runner.run_calibration(args,device,out);write(out/'calibration_summary.json',cal)
        model,stages,bounds=runner.build_experiment(args,device);model=model.to(device);initial=digest(model);torch.save(dict(model_state=model.state_dict(),args=runner.vars_for_json(args),model_sha256=initial),out/'initial.pt')
        phase='formal_training';status('fresh formal model after separate calibration');runner.train_formal(args,device,out,model,stages,bounds,cal)
        torch.save(dict(model_state=model.state_dict(),args=runner.vars_for_json(args),calibration=cal),out/'last.pt');runner.plot_curves(out,bounds)
        result=dict(status='fixture_complete' if fixture else 'complete',dimension=dim,mode=mode,seed=args.seed,formal_model_updates=sum(args.controlled_stage_steps),separate_calibration_model_updates=calibration_rows,initial_model_sha256=initial,final_model_sha256=digest(model),snapshots=len(list((out/'snapshots').glob('*.pt'))),wall_seconds=time.monotonic()-started,device='cpu',threads=1,original_environment_unknown=True,protocol_reconstruction=True,not_original_source_byte_reproduction=True)
        if not fixture and any(sha(ROOT/n)!=h for n,h in frozen['files'].items()):raise ValueError('Frozen source changed')
        write(out/'execution.json',result);write(out/'progress.json',dict(status=result['status'],step=sum(args.controlled_stage_steps),calibration_rows=calibration_rows,updated_at=time.time()));print(json.dumps(result),flush=True)
        return out
    except BaseException as exc:write(out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),phase=phase,step=step,calibration_rows=calibration_rows));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dim',type=int,required=True,choices=[2,4,8,16,32]);p.add_argument('--mode',required=True,choices=['loss_only','loss_plus_xi']);p.add_argument('--run-id',default='reconstructed_B_001');p.add_argument('--fixture',action='store_true');a=p.parse_args();run(a.dim,a.mode,a.run_id,a.fixture)
