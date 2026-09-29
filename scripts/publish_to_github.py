#!/usr/bin/env python3
"""Transfer the FULL local bundle using the user's local Git/Git LFS login.
Default is a dry run. Never force-push, merge, or request credentials in code.
An explicit expected remote commit prevents overwriting concurrent changes.
"""
import argparse,json,re,shutil,subprocess,sys
from pathlib import Path
from verify_assets import verify
REPOSITORY='https://github.com/r1293870729-dev/soma.git'
BRANCH='audit/restructure-20260929'
SKIP={'.git','.venv','__pycache__','.pytest_cache','publish_worktree'}

def run(*args,cwd=None,capture=False):
    p=subprocess.run(list(args),cwd=cwd,text=True,check=True,
        stdout=subprocess.PIPE if capture else None,stderr=subprocess.PIPE if capture else None)
    return p.stdout.strip() if capture else ''

def paths(root):
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if any(part in SKIP or part.endswith('.egg-info') for part in rel.parts):continue
        if rel.parts[:2] in {('reports','local'),('results','generated'),('data','raw'),('data','cache')}:continue
        if p.is_symlink():raise ValueError(f'Symlink not allowed: {rel}')
        if p.is_file():
            if p.name=='.env' or p.name.startswith('.env.'):raise ValueError(f'Environment secret file: {rel}')
            yield p,rel

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--workdir',type=Path,required=True,help='A new path outside the full bundle')
    ap.add_argument('--expected-head',required=True,help='Reviewed full 40-character remote commit SHA')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--push',action='store_true')
    a=ap.parse_args();src=a.source.resolve();dest=a.workdir.resolve()
    if not re.fullmatch(r'[0-9a-f]{40}',a.expected_head):raise SystemExit('Full expected SHA required')
    if src==dest or src in dest.parents or dest in src.parents:raise SystemExit('Workdir must be outside and not contain source')
    if a.push and not a.execute:raise SystemExit('--push requires --execute')
    checked=verify(src,full=True)
    if not checked['passed']:print(json.dumps(checked,ensure_ascii=False,indent=2));raise SystemExit(2)
    selected=list(paths(src))
    secret=re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}')
    for p,rel in selected:
        if p.suffix.lower() in {'.py','.md','.json','.toml','.yaml','.yml','.sh','.ps1','.txt'} and p.stat().st_size<2_000_000:
            if secret.search(p.read_bytes()):raise SystemExit(f'Possible secret: {rel}; investigate first')
    print(json.dumps({'repository':REPOSITORY,'branch':BRANCH,'files':len(selected),
        'bytes':sum(p.stat().st_size for p,_ in selected),'mode':'execute' if a.execute else 'dry-run',
        'notice':'LFS provider storage/transfer quota and charges may apply.'},ensure_ascii=False,indent=2))
    if not a.execute:return
    if dest.exists():raise SystemExit('Workdir exists; refuse overwrite or blind resume')
    if not shutil.which('git'):raise SystemExit('Install Git and Git LFS locally')
    run('git','lfs','version')
    observed=run('git','ls-remote',REPOSITORY,f'refs/heads/{BRANCH}',capture=True).split()[0]
    if observed!=a.expected_head:raise SystemExit(f'Remote changed: {observed}; review before proceeding')
    run('git','clone','--single-branch','--branch',BRANCH,REPOSITORY,str(dest))
    if run('git','rev-parse','HEAD',cwd=dest,capture=True)!=a.expected_head:raise SystemExit('Remote moved during clone')
    for setting in ('user.name','user.email'):
        if not run('git','config','--get',setting,cwd=dest,capture=True):raise SystemExit(f'Configure Git {setting}')
    run('git','lfs','install','--local',cwd=dest)
    for p,rel in selected:
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    run('git','add','--all',cwd=dest)
    if not run('git','diff','--cached','--name-only',cwd=dest,capture=True):raise SystemExit('Nothing to commit')
    run('git','diff','--cached','--stat',cwd=dest)
    run('git','commit','-m','chore: import original SOMA assets and audited experiment workspace',cwd=dest)
    run('git','lfs','fsck',cwd=dest)
    local=run('git','rev-parse','HEAD',cwd=dest,capture=True)
    receipt={'branch':BRANCH,'base':a.expected_head,'local_commit':local,'pushed':False,'files':len(selected)}
    if a.push:
        latest=run('git','ls-remote',REPOSITORY,f'refs/heads/{BRANCH}',capture=True).split()[0]
        if latest!=a.expected_head:raise SystemExit('Remote changed before push; no force push attempted')
        run('git','lfs','push','origin',BRANCH,cwd=dest)
        run('git','push','origin',f'HEAD:refs/heads/{BRANCH}',cwd=dest)
        remote=run('git','ls-remote',REPOSITORY,f'refs/heads/{BRANCH}',capture=True).split()[0]
        if remote!=local:raise SystemExit('Post-push verification failed')
        receipt.update(pushed=True,verified_remote_commit=remote,lfs_upload_command='successful',remote_binary_redownload='not-performed')
    report=src/'reports/local/transfer_receipt.json';report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except (subprocess.CalledProcessError,ValueError) as e:
        print(f'Publish stopped: {e}',file=sys.stderr);raise SystemExit(2)
