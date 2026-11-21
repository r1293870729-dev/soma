"""Verify immutable published assets or replot their actual dense measurements."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, sys, time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO/'results/generated/soma_dense_switch_20261121'
PRIOR = REPO/'results/generated/soma_3000_multidim_20261117/workspace'
LOW = 'soma_lowdim_switch_probe_20261117'
DENSE = 'soma_dense_switch_windows_20261119'
MANIFEST = json.loads((ARCHIVE/'asset_manifest.json').read_text()) if (ARCHIVE/'asset_manifest.json').exists() else None

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def verify():
    total = 0
    for i, r in enumerate(MANIFEST['files'], 1):
        p = REPO/r['path']
        if not p.is_file() or p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
            raise ValueError('Missing/changed archive asset; complete git lfs pull: '+r['path'])
        total += r['bytes']
        if i % 500 == 0: print(f'Verified {i}/{len(MANIFEST["files"])} files', flush=True)
    aliases = json.loads((Path(__file__).parent/'figure_alias_manifest.json').read_text())
    for name, r in aliases.items():
        if sha(REPO/name) != r['sha256']: raise ValueError('Changed figure alias: '+name)
    prior = REPO/MANIFEST['inherited_dependency_manifest']
    if sha(prior) != MANIFEST['inherited_dependency_manifest_sha256']: raise ValueError('Changed prior dependency manifest')
    receipt = dict(status='pass',archive_files=len(MANIFEST['files']),archive_bytes=total,figure_aliases=len(aliases),training_updates=0)
    print(json.dumps(receipt)); return receipt

def replot(output, prior, points_only):
    if output.exists(): raise ValueError('--output must be a new directory')
    if not prior.is_dir(): raise FileNotFoundError('Prior 42-condition archive is required; run git lfs pull first')
    workspace = output/'workspace'
    workspace.mkdir(parents=True)
    roots = [*prior.iterdir(), ARCHIVE/'workspace'/LOW, ARCHIVE/'workspace'/DENSE]
    for original in roots:
        if not original.is_dir(): continue
        dst = workspace/original.name
        dst.mkdir()
        for child in original.iterdir():
            target = dst/child.name
            if child.name in {'src','tools','configs','docs'} and child.is_dir():
                shutil.copytree(child,target)
            elif original.name == DENSE and child.name in {'reports','figures'}:
                target.mkdir()
            else:
                target.symlink_to(child,target_is_directory=child.is_dir())
    # Original source/config/result bytes remain untouched. Only deserialized
    # path strings are relocated in this independent read-only plotting process.
    original_root = Path(MANIFEST['original_outputs_root'])
    native_loads = json.loads
    def relocate(v):
        if isinstance(v,str):
            try: return str(workspace/Path(v).relative_to(original_root))
            except ValueError: return v
        if isinstance(v,list): return [relocate(x) for x in v]
        if isinstance(v,dict): return {relocate(k):relocate(x) for k,x in v.items()}
        return v
    def relocated_loads(s,*a,**kw): return relocate(native_loads(s,*a,**kw))
    json.loads = relocated_loads
    sys.path.insert(0,str(workspace/DENSE/'src'))
    import render_dense as render
    cases = render.load()
    if len(cases) != 60 or any(c['dense'] is None for c in cases): raise ValueError('Expected all 60 verified conditions')
    render.point_table(cases)
    table = workspace/DENSE/'reports/plot_points_by_seed.csv'
    original_table = ARCHIVE/'workspace'/DENSE/'reports/plot_points_by_seed.csv'
    def values(p):
        result={}
        with p.open() as f:
            for row in csv.DictReader(f):
                key=tuple(row[k] for k in row if k!='value')
                if key in result: raise ValueError('Duplicate plot point')
                result[key]=row['value']
        return result
    expected, observed = values(original_table), values(table)
    if expected != observed: raise ValueError('Replotted point identities or exact numerical strings differ')
    if not points_only:
        for ds in render.DATASETS:
            render.panel(ds,cases,render.PLAN['cuts'])
            for cut in render.PLAN['cuts']: render.panel(ds,cases,[cut])
            for dim in (render.PLAN['controlled_dims'] if ds=='controlled' else render.PLAN['image_dims'][ds]):
                render.detail(ds,dim,cases)
    receipt=dict(status='pass',conditions=60,rows=len(expected),all_point_keys_and_values_exact=True,
                 csv_byte_identical=sha(original_table)==sha(table),original_table_sha256=sha(original_table),
                 output_table_sha256=sha(table),training_updates=0,interpolation=False,points_only=points_only,
                 pngs=len(list((workspace/DENSE/'figures').glob('*.png'))),
                 svgs=len(list((workspace/DENSE/'figures').glob('*.svg'))))
    (output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--verify',action='store_true');g.add_argument('--plot',action='store_true');g.add_argument('--points-only',action='store_true')
    p.add_argument('--output');p.add_argument('--prior-workspace',default=str(PRIOR))
    a=p.parse_args()
    if a.verify: verify()
    else: replot(Path(a.output).resolve() if a.output else REPO/'reports/local'/('dense_replot_'+str(int(time.time()))),Path(a.prior_workspace).resolve(),a.points_only)
