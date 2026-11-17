from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'soma_vscode_mac_20261111'
sys.path.insert(0,str(ROOT/'vendor/fixed_v03'))
from experiments.b_weight_sensitivity.historical_loader import HistoricalSource
from experiments.b_weight_sensitivity.formal_runner import tensor_digest
from experiments.manual_v03.signal import FixedPsi,SignalConfig
from experiments.manual_v03.readout import FrozenReadout
from generator import Generator,seed_for,hash_array

def plan():return json.loads((ROOT/'configs/plan.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def append(path,rows):
    if not rows:return
    path=Path(path);first=not path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        if first:w.writeheader()
        w.writerows(rows)
def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def sources():
    source=HistoricalSource(ROOT/'vendor/fixed_v03/legacy/03_psi_xib_unified')
    return source,source.load('train_xib_unified'),source.load('models')
ROLES=['unused','white','edge','edge-cover','train','validation','probe-train','probe-val','probe-test']
ROLE={name:i for i,name in enumerate(ROLES)}
TASKS={
 'dsprites':[[('x',0,32),('y',1,32),('scale',2,6)],[('x',0,32),('y',1,32),('scale',2,6),('shape',3,3)],[('x',0,32),('y',1,32),('scale',2,6),('shape',3,3)]],
 'shapes3d':[[('floor_hue',0,10),('wall_hue',1,10),('object_hue',2,10),('scale',3,8)],[('floor_hue',0,10),('wall_hue',1,10),('object_hue',2,10),('scale',3,8),('shape',4,4)],[('floor_hue',0,10),('wall_hue',1,10),('object_hue',2,10),('scale',3,8),('shape',4,4)]],
 'cifar':[[('class',0,10)],[('class',0,10)],[('class',0,10)]]}

def sample_indices(pool,count,rng,labels=None,tasks=None,unique_keys=None):
    pool=pool.copy()
    if unique_keys is not None:
        _,first=__import__('numpy').unique(unique_keys[pool],return_index=True);pool=pool[first]
    if count>len(pool):raise ValueError(f'Need {count} distinct probe images, only {len(pool)}')
    for attempt in range(1000):
        ids=rng.choice(pool,count,replace=False)
        if labels is None or all(len(__import__('numpy').unique(labels[ids,col]))==classes for _,col,classes in tasks):return ids
    raise ValueError('Cannot obtain all declared probe classes without overlap')
