from pathlib import Path
import csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
PLAN=json.loads((ROOT/'configs/plan.json').read_text())
OLD=BASE/'soma_all_datasets_3000_multidim_20261116';PARENT=BASE/'soma_four_dataset_manual_v03_20261114';PAIR=BASE/'soma_image_schemeB_vs_manual_20261116'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def append(p,records):
    if not records:return
    p=Path(p);first=not p.exists()
    with p.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader() if first else None;w.writerows(records)
def stage_at(t):
    if t<1 or t>3000:raise ValueError('Arrival outside3000 steps')
    return (t-1)//1000
def verify():
    p=ROOT/'configs/frozen.json'
    if p.exists():
        for name,h in json.loads(p.read_text())['files'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen source changed '+name)
    original=json.loads((OLD/'configs/frozen.json').read_text())
    for p,h in original['dependencies'].items():
        if sha(p)!=h:raise ValueError('Original dependency changed '+p)
