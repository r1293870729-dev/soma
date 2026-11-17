"""Relocate immutable archived paths; verify, replot, or explicitly start a fresh run."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,runpy,shutil,sys,time
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[2];ARCHIVE=REPO/'results/generated/soma_3000_multidim_20261117';WORKSPACE=ARCHIVE/'workspace';NAME='soma_all_datasets_3000_multidim_20261116'
MANIFEST=json.loads((ARCHIVE/'asset_manifest.json').read_text());ORIGINAL=Path(MANIFEST['original_outputs_root'])
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()
def relocate(value,*parts):
    p=Path(value,*parts)
    try:return WORKSPACE/p.relative_to(ORIGINAL)
    except ValueError:return p
def bootstrap(root):
    sys.path.insert(0,str(root/'src'));import paths
    native=paths.sha;paths.sha=lambda p:native(relocate(p));paths.verify();return paths
def verify_all():
    started=time.monotonic();total=0
    for i,row in enumerate(MANIFEST['files'],1):
        p=REPO/row['path']
        if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Missing/changed asset: '+row['path']+'; complete git lfs pull first')
        total+=row['bytes']
        if i%500==0:print(f'verified {i}/{len(MANIFEST["files"])} files',flush=True)
    print(json.dumps(dict(status='pass',files=len(MANIFEST['files']),bytes=total,seconds=time.monotonic()-started)))
def replot(output):
    root=WORKSPACE/NAME;bootstrap(root);sys.path.insert(0,str(root/'tools'));import render
    render.Path=relocate
    spec=importlib.util.spec_from_file_location('portable_combined_plot',root/'tools/plot_combined_all.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if output.exists():raise ValueError('Output must be a new directory to preserve previous results')
    m.OUT=output/'figures';m.REPORT=output/'reports';m.OUT.mkdir(parents=True);m.REPORT.mkdir()
    cases=render.load_cases();assert len(cases)==42
    for ds in m.ORDER:
        selected=[c for c in cases if c['dataset']==ds];m.dataset_figure(ds,selected);m.dataset_figure(ds,[c for c in selected if c['seed']==(0 if ds=='controlled' else 10)],single=True)
    m.master_figure(cases);values=[r for rows in m.TABLE.values() for r in rows];path=m.REPORT/'plotted_values.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    original=root/'reports/combined_20261117/plotted_values.csv';assert sha(path)==sha(original),'Replotted numerical data differs from original'
    receipt=dict(status='pass',parents=42,pngs=9,svgs=9,plotted_rows=len(values),plotted_values_sha256=sha(path),plotted_values_byte_identical=True,new_training_updates=0,output=str(output))
    (m.REPORT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
def train(args):
    # Only explicit --train enters optimizer code. Archived results are never erased.
    fresh=Path(args.workdir).resolve()
    if fresh.exists():raise ValueError('--workdir must be a new directory')
    fresh.mkdir(parents=True)
    root=fresh/NAME;root.mkdir()
    for name in ['src','tools','configs','docs']:shutil.copytree(WORKSPACE/NAME/name,root/name)
    for name in ['data_cache','results','reports','logs']:(root/name).mkdir()
    for p in WORKSPACE.iterdir():
        if p.name!=NAME and p.is_dir():(fresh/p.name).symlink_to(p,target_is_directory=True)
    bootstrap(root)
    if args.train=='controlled':file=root/'src/controlled_worker.py';sys.argv=[str(file)];runpy.run_path(str(file),run_name='__main__');return
    if not args.data_root:raise ValueError('Image training requires --data-root, with original data/cifar10 and data/CIFAR-C layout')
    import numpy as np
    parent=WORKSPACE/'soma_four_dataset_manual_v03_20261114';numeric=json.loads((parent/'data_cache/numeric_paths.json').read_text());overrides={}
    for key,value in [('dsprites_images',args.dsprites_images),('shapes3d_images',args.shapes_images)]:
        if value:overrides[numeric[key]]=str(Path(value).resolve())
    if args.train in ['dsprites','shapes3d']:
        key=args.train+'_images';p=Path(overrides.get(numeric[key],numeric[key]))
        if not p.is_file():raise ValueError('Provide --dsprites-images or --shapes-images for the omitted raw numeric image input')
    native=np.load
    def relocated_numpy_load(path,*a,**kw):
        return native(overrides.get(str(path),path),*a,**kw)
    np.load=relocated_numpy_load
    sys.path.insert(0,str(WORKSPACE/'soma_image_schemeB_vs_manual_20261116/src'));import runtime;import common
    common.OLD=Path(args.data_root).resolve()
    file=root/'src/image_worker.py';sys.argv=[str(file),'--dataset',args.train];runpy.run_path(str(file),run_name='__main__')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--verify',action='store_true');g.add_argument('--plot',action='store_true');g.add_argument('--train',choices=['controlled','dsprites','shapes3d','cifar']);p.add_argument('--output');p.add_argument('--workdir');p.add_argument('--data-root');p.add_argument('--dsprites-images');p.add_argument('--shapes-images');a=p.parse_args()
    if a.verify:verify_all()
    elif a.plot:replot(Path(a.output).resolve() if a.output else REPO/'reports/local'/('3000_multidim_replot_'+str(int(time.time()))))
    else:
        if not a.workdir:p.error('--train requires a new --workdir')
        train(a)
