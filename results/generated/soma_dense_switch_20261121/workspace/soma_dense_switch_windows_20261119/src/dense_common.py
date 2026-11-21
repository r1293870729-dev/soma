from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
LOW=BASE/'soma_lowdim_switch_probe_20261117';OLD=BASE/'soma_all_datasets_3000_multidim_20261116';PARENT=BASE/'soma_four_dataset_manual_v03_20261114';HIST=BASE/'soma_scheme_ab_reproduction_20261116'
PLAN=json.loads((ROOT/'configs/plan.json').read_text());SOURCE_PLAN=json.loads((LOW/'configs/plan.json').read_text())
sys.path.insert(0,str(LOW/'src'))
from study_paths import sha,write,rows,append
def selected(t):return any(lo<=t<=hi for lo,hi in PLAN['windows'])
def wanted():return [t for lo,hi in PLAN['windows'] for t in range(lo,hi+1)]
def pin(pins,p):p=Path(p).resolve();pins[str(p)]=sha(p);return p
def read(pins,p):return rows(pin(pins,p))
def jread(pins,p):return json.loads(pin(pins,p).read_text())
def verify():
    from study_paths import verify as source_verify
    source_verify()
    frozen=ROOT/'configs/frozen.json'
    if frozen.exists():
        for p,h in json.loads(frozen.read_text())['files'].items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen dense source changed '+p)
def cases(ds):
    result=[]
    homes=[OLD/'results/all_3000_001'] if ds=='controlled' else [OLD/'results/all_3000_001',LOW/'results'/SOURCE_PLAN['run_id']]
    for home in homes:
        for f in home.glob('*/summary.json'):
            s=json.loads(f.read_text())
            if s.get('dataset')==ds and s['status']=='complete':result.append((f.parent,s))
    result.sort(key=lambda pair:(pair[1]['latent_dim'],pair[1]['seed']))
    if len(result)!=15:raise ValueError('Expected15 original cases '+ds)
    return result
def check_sources(pins):
    for file,h in pins.items():
        if sha(file)!=h:raise ValueError('Readonly source changed '+file)
def compare(original,new,keys,tol=0.):
    for k in keys:
        if abs(float(original[k])-float(new[k]))>tol:raise ValueError('Archived value differs '+k+': '+str((original[k],new[k])))
def reference(folder,age,pins):
    # Imported only after the appropriate runtime was selected in each process.
    from experiments.manual_v03.signal import FixedPsi,SignalConfig
    import numpy as np
    top=folder/f'calibration_ref{age}.json'
    if top.exists():meta=jread(pins,top)
    else:meta=jread(pins,folder/f'ref{age}/calibration.json')
    candidates=[folder/f'ref{age}/calibration_residuals.npz']
    inputs=folder/'source_inputs.json'
    if inputs.exists():
        for file in jread(pins,inputs):
            path=Path(file)
            if path.name=='calibration.json' and path.exists():
                v=jread(pins,path)
                if v.get('reference_sha256')==meta['reference_sha256']:candidates.append(path.with_name('calibration_residuals.npz'))
    data=next((f for f in candidates if f.exists()),None)
    if data is None:raise FileNotFoundError('Archived residual reference '+str(folder)+' '+str(age))
    with np.load(pin(pins,data),allow_pickle=False) as a:s=FixedPsi.fit(a['white'],a['edge'],SignalConfig(**meta['config']),meta['roles'])
    if s.reference_hash()!=meta['reference_sha256']:raise ValueError('Reference hash mismatch '+str(age))
    return s,meta
