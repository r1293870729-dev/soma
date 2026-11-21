"""Export completed experiments without changing their bytes or executing training."""
from pathlib import Path
import argparse, collections, ctypes, hashlib, json, os, re, shutil, time

PROJECTS = ['soma_lowdim_switch_probe_20261117', 'soma_dense_switch_windows_20261119']
ID = 'soma_dense_switch_20261121'
TASK = 'experiments/dense_switch_20261121'
SKIP = {'.venv', '.git', '__pycache__', '.pytest_cache', '.cache', '.DS_Store'}
SECRET = re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')

def main(a):
    source, repo, status = Path(a.source).resolve(), Path(a.repo).resolve(), Path(a.status).resolve()
    archive = repo/'results/generated'/ID
    entry = repo/TASK
    status.mkdir(parents=True, exist_ok=True)
    entry.mkdir(parents=True, exist_ok=True)
    low = json.loads((source/PROJECTS[0]/'reports/monitor_complete.json').read_text())
    dense = json.loads((source/PROJECTS[1]/'reports/LIVE_PROGRESS.json').read_text())
    assert low['status'] == dense['status'] == 'complete'
    assert low['new_training_complete'] == 18 and low['frozen_replay_complete'] == 45 and not low['errors']
    assert dense['complete_conditions'] == 60 and not dense['errors']
    selected = {}
    excluded = []
    for name in PROJECTS:
        for p in (source/name).rglob('*'):
            if any(part in SKIP for part in p.relative_to(source/name).parts): continue
            if p.is_symlink():
                excluded.append(str(p.relative_to(source))); continue
            if p.is_file():
                if p.name == '.env' or p.name.startswith('.env.'): raise ValueError('Unexpected environment file')
                selected[p.relative_to(source)] = p
    for name in ['controlled_all_dims.png', 'controlled_all_dims.svg', 'controlled_all_dimensions.zip']:
        p = source/'downloads'/name
        selected[p.relative_to(source)] = p
    clone = None
    if os.uname().sysname == 'Darwin':
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        clone = lib.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
    records = []
    started = time.monotonic()
    for i, (rel, p) in enumerate(sorted(selected.items()), 1):
        dst = archive/'workspace'/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in {'.py','.md','.json','.jsonl','.log','.csv','.toml','.yaml','.yml','.sh','.txt'}:
            with p.open('rb') as f:
                previous = b''
                for block in iter(lambda: f.read(1024*1024), b''):
                    if SECRET.search(previous+block): raise ValueError('Possible credential in '+str(rel))
                    previous = block[-256:]
        if not dst.exists():
            if clone is None or clone(os.fsencode(p), os.fsencode(dst), 0): shutil.copy2(p, dst)
            else: shutil.copystat(p, dst)
        before, after = sha(p), sha(dst)
        assert before == after, str(rel)
        records.append(dict(path=str(dst.relative_to(repo)), original_path=str(p), bytes=p.stat().st_size, sha256=after,
                            role='download_alias' if rel.parts[0]=='downloads' else 'complete_experiment_output'))
        if i % 300 == 0:
            print(json.dumps(dict(files=i,total=len(selected),seconds=round(time.monotonic()-started,1))), flush=True)
    prior = repo/'results/generated/soma_3000_multidim_20261117/asset_manifest.json'
    manifest = dict(schema=1,experiment_id=ID,original_outputs_root=str(source),files=records,
                    original_bytes_preserved=True,raw_datasets_included=False,excluded=sorted(SKIP),excluded_symlinks=excluded,
                    inherited_dependency_manifest=str(prior.relative_to(repo)),inherited_dependency_manifest_sha256=sha(prior),
                    inherited_base_commit='2d012be74ecdcf35b4f8ae99bc34e987bbe1bba9')
    write(archive/'asset_manifest.json',manifest)
    # Human-friendly ASCII aliases preserve exactly the published original image bytes.
    figure_map = {}
    for ds in ['controlled','dsprites','shapes3d','cifar']:
        for ext in ['png','svg']:
            src = source/PROJECTS[1]/'figures'/f'{ds}_逐步切换_全部维数.{ext}'
            dst = entry/'figures'/f'{ds}_all_dimensions.{ext}'
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src,dst)
            figure_map[str(dst.relative_to(repo))] = dict(source=str(src),sha256=sha(src),bytes=src.stat().st_size)
    for ext in ['png','svg']:
        src=source/PROJECTS[0]/'figures'/f'总览_四数据集_全部维数_新旧信号_MSE_探针.{ext}'
        dst=entry/'figures'/f'all_datasets_overview.{ext}'
        shutil.copy2(src,dst)
        figure_map[str(dst.relative_to(repo))]=dict(source=str(src),sha256=sha(src),bytes=src.stat().st_size)
    write(entry/'figure_alias_manifest.json',figure_map)
    attrs=repo/'.gitattributes'
    block='\n# Low-dimensional results and dense switch-window publication, 2026-11-21\n'
    for r in records:
        if r['path'].endswith('.csv') and r['bytes']>=5*1024**2:
            block+=f'"{r["path"]}" filter=lfs diff=lfs merge=lfs -text\n'
    text=attrs.read_text()
    if block not in text: attrs.write_text(text+block)
    summary=dict(status='prepared',experiment_id=ID,branch='experiment/lowdim-dense-results-20261121',
                 base_commit=manifest['inherited_base_commit'],archived_files=len(records),archived_bytes=sum(r['bytes'] for r in records),
                 archived_projects={name:sum(Path(r['original_path']).is_relative_to(source/name) for r in records) for name in PROJECTS},
                 new_training_conditions=18,reused_training_conditions=42,total_conditions=60,dataset_dimension_combinations=20,
                 frozen_probe_replays=45,pretrained_input_controls=9,new_training_updates=90000,dense_replayed_updates=227700,
                 new_treatment_updates_in_dense_replay=0,stage_steps=[1000,1000,1000],
                 dense_windows=[[970,1030],[1970,2030]],dense_points_per_window=61,raw_datasets_included=False,
                 scientific_status='exploratory_development_not_Q1_Q3_acceptance',upload_status='not_yet_uploaded',
                 asset_manifest=str((archive/'asset_manifest.json').relative_to(repo)))
    write(entry/'publication_summary.json',summary)
    write(status/'prepare_receipt.json',dict(**summary,secret_scan='pass',seconds=time.monotonic()-started))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--repo',required=True);p.add_argument('--status',required=True)
    main(p.parse_args())
