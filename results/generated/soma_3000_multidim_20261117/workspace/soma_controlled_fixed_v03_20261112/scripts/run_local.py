"""Run the approved controlled pipeline and retain stdout plus exit receipt."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--experiment-id',required=True);a=p.parse_args()
if not a.experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in a.experiment_id):raise ValueError('Unsafe experiment ID')
lock=(ROOT/'reports/local_pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=ROOT/'reports'/('exit_'+a.experiment_id+'.json')
if receipt.exists():raise FileExistsError('Preserve prior run receipt; use a new experiment ID')
start=time.time();log=ROOT/'reports'/(a.experiment_id+'.log')
meta=dict(experiment_id=a.experiment_id,pid=os.getpid(),started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),log=str(log),planned_runs=3,planned_model_updates=9000)
(ROOT/'reports/latest_execution.json').write_text(json.dumps(meta,indent=2)+'\n')
env=dict(os.environ,BWS_CPU_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
code=1
try:
    with log.open('x') as handle:
        print('Controlled pipeline log: '+str(log),flush=True)
        code=subprocess.run([sys.executable,'-u','scripts/run_manual.py','all','--experiment-id',a.experiment_id,'--device','cpu'],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
finally:
    receipt.write_text(json.dumps(dict(status='complete' if code==0 else 'failed',exit_code=code,experiment_id=a.experiment_id,wall_seconds=time.time()-start),indent=2)+'\n')
raise SystemExit(code)
