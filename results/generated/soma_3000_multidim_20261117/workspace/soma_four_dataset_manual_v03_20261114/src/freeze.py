import argparse,json,platform,sys,time
import torch,numpy as np
from common import ROOT,plan,sha,write_json

def main():
    a=argparse.ArgumentParser();a.add_argument('--run-id',required=True);a.add_argument('--fixture-id',required=True);args=a.parse_args()
    if json.loads((ROOT/'reports/data_preflight.json').read_text())['status']!='passed':raise ValueError('Data preflight not passed')
    if json.loads((ROOT/'reports/asset_qualification.json').read_text())['status']!='passed':raise ValueError('Asset qualification required')
    if '\nOK\n' not in (ROOT/'logs/unit_tests.log').read_text():raise ValueError('Pipeline contracts must pass')
    fixtures=[]
    for ds in ['controlled','dsprites','shapes3d','cifar']:
        path=ROOT/'results/generated'/args.fixture_id/f'fixture_{ds}.json'
        d=json.loads(path.read_text())
        if d['status']!='passed':raise ValueError('Fixture not passed '+ds)
        probe_check=json.loads((ROOT/f'reports/probe_fixture_{ds}.json').read_text())
        if probe_check['status']!='passed':raise ValueError('Probe path fixture required '+ds)
        fixtures.append(d)
    write_json(ROOT/'reports/fixture_summary.json',dict(status='passed',fixture_id=args.fixture_id,records=fixtures,model_updates=800))
    root=ROOT/'results/generated'/args.run_id;root.mkdir(exist_ok=False)
    files={}
    for folder in ['src','tests','configs','vendor']:
        for file in sorted((ROOT/folder).rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts:files[str(file.relative_to(ROOT))]=sha(file)
    for name in ['docs/design_rationale.md','README.md']:
        file=ROOT/name;files[name]=sha(file)
    audits={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'reports/source_inputs.json',ROOT/'reports/data_preflight.json',ROOT/'reports/asset_qualification.json',ROOT/'logs/unit_tests.log',*[ROOT/f'reports/probe_fixture_{ds}.json' for ds in ['controlled','dsprites','shapes3d','cifar']]]}
    write_json(root/'frozen.json',dict(files=files,started_at=time.time(),audits=audits))
    write_json(root/'plan.json',plan());write_json(root/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),device='cpu',MPS_available=torch.backends.mps.is_available(),threads_per_worker=2,workers=4))
    print(str(root),flush=True)

if __name__=='__main__':main()
