# Source integrity checks only; no PyTorch import and no model execution.
from pathlib import Path
import argparse,ast,hashlib,json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--legacy-root',type=Path)
p.add_argument('--out',type=Path)
a=p.parse_args()
root=a.root.resolve();legacy=a.legacy_root or root/'legacy/03_psi_xib_unified'
pin=json.loads((root/'experiments/b_weight_sensitivity/configs/source_pin.json').read_text())
for name,expected in pin['files'].items():
    source=legacy/name
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit('历史源码哈希不匹配：'+str(source))
    ast.parse(source.read_text())
new_files=list((root/'experiments/b_weight_sensitivity').rglob('*.py'))
for source in new_files:ast.parse(source.read_text())
result={'status':'passed_source_integrity_only','original_files_verified':len(pin['files']),
        'experiment_python_files_parsed':len(new_files),'legacy_root':str(legacy.resolve()),
        'numerical_validation_performed':False,'source_hashes':pin['files']}
if a.out:
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
