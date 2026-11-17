import sys,json,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reconstruction'))
from runtime import ROOT,sha,write
import torch,numpy,ignite
p=json.loads((ROOT/'configs/reconstruction_plan.json').read_text());out=ROOT/'results'/p['run_id'];out.mkdir(exist_ok=False)
if '\nOK\n' not in (ROOT/'logs/restored_tests.log').read_text():raise ValueError('Math tests have not passed')
fixture=ROOT/'results/engineering_fixture_001/B_z2_loss_plus_xi'
if json.loads((fixture/'execution.json').read_text())['status']!='fixture_complete':raise ValueError('Pipeline fixture missing')
if json.loads((fixture/'fixed_probe_summary.json').read_text())['status']!='complete':raise ValueError('Fixed-view round-trip fixture missing')
files={}
for folder in ['candidate_handoff_20260903','reconstruction','tests','configs','vendor_deps']:
    for f in sorted((ROOT/folder).rglob('*')):
        if f.is_file() and '__pycache__' not in f.parts:files[str(f.relative_to(ROOT))]=sha(f)
for name in ['docs/RECONSTRUCTION_PROTOCOL.md','scripts/freeze_reconstruction.py']:files[name]=sha(ROOT/name)
for f in (ROOT/'archive_logs').glob('*/resolved_args.json'):files[str(f.relative_to(ROOT))]=sha(f)
write(out/'frozen.json',dict(files=files,frozen_at=time.time(),protocol_reconstruction=True));write(out/'plan.json',p);write(out/'environment.json',dict(torch=torch.__version__,numpy=numpy.__version__,ignite=ignite.__version__,python=sys.version,platform=platform.platform(),device='cpu',threads=1,original_environment_unknown=True))
print('Frozen:',out,flush=True)
