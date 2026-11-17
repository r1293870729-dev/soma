from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
PLAN=json.loads((ROOT/'configs/plan.json').read_text())
PARENT=BASE/'soma_four_dataset_manual_v03_20261114';IMAGE=BASE/'soma_image_dimensions_ref0_ref1000_20261116';PAIR=BASE/'soma_image_schemeB_vs_manual_20261116';HIST=BASE/'soma_scheme_ab_reproduction_20261116';ANCHOR=BASE/'soma_four_dataset_historical_anchor_20261116'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def stage_at(t):
    if not 1<=t<=sum(PLAN['stage_steps']):raise ValueError('Arrival outside planned3000 steps')
    return min(2,(t-1)//1000)
def verify():
    file=ROOT/'configs/frozen.json'
    if file.exists():
        f=json.loads(file.read_text())
        for p,h in f['files'].items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen local source changed '+p)
        for p,h in f['dependencies'].items():
            if sha(p)!=h:raise ValueError('Frozen dependency changed '+p)
