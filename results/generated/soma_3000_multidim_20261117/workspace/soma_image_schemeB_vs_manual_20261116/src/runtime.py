from pathlib import Path
import sys,json,hashlib,csv
ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'configs/plan.json').read_text())
PARENT=ROOT.parent/PLAN['parent_project'];PARENT_RUN=PARENT/'results/generated'/PLAN['parent_run']
REFS=ROOT.parent/PLAN['manual_ref_project']/'results/generated'/PLAN['manual_ref_run']/'observers'
sys.path.insert(0,str(ROOT/'vendor/report_scheme_B'))
sys.path.insert(0,str(PARENT/'src'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,v):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def verify():
    for n,h in json.loads((PARENT_RUN/'frozen.json').read_text())['files'].items():
        if sha(PARENT/n)!=h:raise ValueError('Parent code changed: '+n)
    file=ROOT/'configs/frozen.json'
    if file.exists():
        for n,h in json.loads(file.read_text())['files'].items():
            if sha(ROOT/n)!=h:raise ValueError('Frozen comparison code changed: '+n)

