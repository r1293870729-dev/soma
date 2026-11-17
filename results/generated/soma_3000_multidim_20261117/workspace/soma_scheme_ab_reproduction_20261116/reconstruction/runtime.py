import importlib.util,json,sys,hashlib
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
for path in [ROOT/'vendor_deps',ROOT/'candidate_handoff_20260903']:
    if str(path) not in sys.path:sys.path.insert(0,str(path))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):h.update(name.encode());h.update(str((tuple(t.shape),str(t.dtype))).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def load_runner():
    from legacy_signal import RestoredAbsoluteMPSignal
    name='restored_scheme_b_candidate';file=ROOT/'candidate_handoff_20260903/run_pw_signal_experiment.py'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.PrecisionWeightedResidualSignal=RestoredAbsoluteMPSignal
    return module

def case_args(runner,dim,mode,out):
    original=json.loads((ROOT/f'archive_logs/B_z{dim}_{mode}/resolved_args.json').read_text());argv=sys.argv
    try:sys.argv=['run_pw_signal_experiment.py','--dataset','controlled'];args=runner.parse_args()
    finally:sys.argv=argv
    defaults=dict(vars(args))
    for key,value in original.items():setattr(args,key,value)
    args.output_dir=Path(out);args.device='cpu';args.xi_signal='pw';args.pw_spectral_readout='edge';args.pw_readout_mode='current'
    # Legacy args contain server-only image paths; this task is controlled only.
    for name in ['cifar10_root','cifarc_root']:setattr(args,name,Path(getattr(args,name)))
    extra={k:v for k,v in defaults.items() if k not in original}
    return args,original,extra
