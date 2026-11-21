"""Stdlib-only supervisor: qualify first cases, then continue all original conditions."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from dense_common import ROOT,verify,write,check_sources,sha
FAMILIES=['controlled','dsprites','shapes3d','cifar']
def state(ds):
    p=ROOT/'reports'/f'worker_{ds}.json';return json.loads(p.read_text()) if p.exists() else {}
def main():
    verify();write(ROOT/'reports/SUPERVISOR.json',dict(status='waiting_qualification',datasets=FAMILIES))
    while True:
        ss={ds:state(ds) for ds in FAMILIES}
        errors=[(ds,v.get('error')) for ds,v in ss.items() if v.get('status')=='failed']
        if errors:raise RuntimeError('First-case qualification failed '+str(errors))
        if all(v.get('status')=='qualified' for v in ss.values()):break
        time.sleep(10)
    qualified={}
    for ds,v in ss.items():
        assert v['completed']==1;s=v['cases'][0];assert s['status']=='complete'
        if ds=='controlled':assert s['original_log_rows_verified']==303 and s['original_snapshots_verified']==15 and s['final_model_exact']
        else:assert s['model_hashes_and_losses_verified_each_step']==3060 and s['original_probe_rows_verified']>0 and s['original_signal_rows_verified']>0 and s['original_frozen_rows_verified']>0
        check_sources(s['source_files']);qualified[ds]=s
    verify();write(ROOT/'reports/qualification.json',dict(status='pass',first_complete_cases=qualified,all_four_families_exact=True,first_cases_reused_in_full_batch=True))
    children={};handles=[]
    for ds in FAMILIES:
        env=dict(os.environ,OMP_NUM_THREADS='1' if ds=='controlled' else '2',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/soma_dense_mpl',PYTHONDONTWRITEBYTECODE='1')
        command=[str(ROOT/'.venv/bin/python'),str(ROOT/'src'/('dense_controlled.py' if ds=='controlled' else 'dense_images.py'))]
        if ds!='controlled':command+=['--dataset',ds]
        log=(ROOT/'logs'/f'full_{ds}.log').open('w');handles.append(log);children[ds]=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
    write(ROOT/'reports/SUPERVISOR.json',dict(status='running',pids={ds:c.pid for ds,c in children.items()},parent_imports_torch=False))
    while any(c.poll() is None for c in children.values()):time.sleep(10)
    codes={ds:c.returncode for ds,c in children.items()}
    for h in handles:h.close()
    if any(codes.values()):raise RuntimeError('Full worker exited unsuccessfully '+str(codes))
    verify();write(ROOT/'reports/SUPERVISOR.json',dict(status='complete',return_codes=codes))
if __name__=='__main__':
    try:main()
    except BaseException as e:write(ROOT/'reports/SUPERVISOR.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise
