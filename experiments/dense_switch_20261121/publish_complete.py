"""Finish an authorized full publication; never publish incomplete LFS pointers."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,subprocess,time,traceback
BRANCH='experiment/lowdim-dense-results-20261121';REPOSITORY='https://github.com/r1293870729-dev/soma.git';TASK='experiments/dense_switch_20261121';EXPECTED_BASE='2d012be74ecdcf35b4f8ae99bc34e987bbe1bba9'
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def run(args,repo,env,log=None):
    if log:
        with Path(log).open('ab') as f:p=subprocess.run(args,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'{args[0]} {args[1]} exited {p.returncode}; see local log {Path(log).name}')
        return ''
    p=subprocess.run(args,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode:raise RuntimeError(f'{args[0]} {args[1]} exited {p.returncode}; no credentials logged')
    return p.stdout.strip()
def tail(p):
    p=Path(p)
    if not p.exists():return ''
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8000));s=f.read().decode(errors='replace')
    lines=[v.strip() for v in s.replace('\r','\n').splitlines() if 'Uploading LFS objects:' in v];return lines[-1] if lines else ''
def alive(pid):
    if not pid:return False
    try:
        os.kill(pid,0);p=subprocess.run(['ps','-p',str(pid),'-o','stat=','-o','comm='],capture_output=True,text=True);s=p.stdout.strip();return p.returncode==0 and not s.startswith('Z') and 'git-lfs' in s
    except ProcessLookupError:return False
def report(work,phase,detail='',**extra):
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds');v=dict(phase=phase,detail=detail,updated_at=now,branch=BRANCH);v.update(extra);write(work/'upload_progress.json',v)
    (work/'UPLOAD_PROGRESS.md').write_text('# GitHub完整上传进度\n\n更新时间：'+now+'\n\n阶段：**'+phase+'**\n\n'+detail+'\n\n'+('GitHub分支：https://github.com/r1293870729-dev/soma/tree/'+BRANCH if phase=='complete' else '完整LFS对象校验成功后才推送Git分支，尚未完成时不声称全部结果已在GitHub。')+'\n')
def metadata_verify(repo,env,work):
    raw=json.loads(run(['git','lfs','ls-files','--json'],repo,env));files=[f for f in raw['files'] if f['name'].startswith('results/generated/soma_dense_switch_20261121/')];objects={f['oid']:dict(oid=f['oid'],size=f['size']) for f in files};items=list(objects.values())
    credential=subprocess.run(['git','-c','credential.interactive=false','credential','fill'],cwd=repo,env=env,input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,check=True)
    values=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line);assert values.get('password'),'No credential available'
    # Normal credential-helper authentication is passed via stdin, never argv/logs.
    user=values['username']+':'+values['password'];checked=0
    for start in range(0,len(items),100):
        payload=work/'lfs_presence_batch.json';write(payload,dict(operation='download',transfers=['basic'],ref=dict(name='refs/heads/'+BRANCH),objects=items[start:start+100]))
        config='\n'.join(['url = '+json.dumps(REPOSITORY+'/info/lfs/objects/batch'),'request = "POST"','silent','show-error','fail-with-body','connect-timeout = 20','max-time = 120','header = "Accept: application/vnd.git-lfs+json"','header = "Content-Type: application/vnd.git-lfs+json"','user = '+json.dumps(user),'data-binary = '+json.dumps('@'+str(payload))])+'\n'
        p=subprocess.run(['curl','--config','-'],input=config,capture_output=True,text=True,env=env,timeout=135)
        if p.returncode:raise RuntimeError('Remote LFS presence check failed; sensitive response not logged')
        response=json.loads(p.stdout);found=response.get('objects',[])
        expected={o['oid']:o['size'] for o in items[start:start+100]}
        if len(found)!=len(expected) or {o['oid'] for o in found}!=set(expected) or any(o.get('error') or 'download' not in o.get('actions',{}) or int(o['size'])!=expected[o['oid']] for o in found):raise RuntimeError('Remote LFS object missing or metadata differs')
        checked+=len(found);report(work,'verifying_remote',f'已确认远端LFS对象 {checked}/{len(items)}',remote_lfs_objects_verified=checked,remote_lfs_objects_planned=len(items))
    return dict(lfs_paths=len(files),unique_lfs_objects=len(items),all_remote_lfs_objects_present=True,lfs_bytes=sum(f['size'] for f in files))
def main(args):
    repo=Path(args.repo).resolve();work=Path(args.status_dir).resolve();work.mkdir(parents=True,exist_ok=True);env=dict(os.environ);env['PATH']=str(Path(args.lfs_dir).resolve())+os.pathsep+str(Path(args.gh_dir).resolve())+os.pathsep+env.get('PATH','');env.update(GH_CONFIG_DIR=str(Path(args.auth_dir).resolve()),GIT_TERMINAL_PROMPT='0',GIT_LFS_FORCE_PROGRESS='1',GIT_LFS_PROGRESS=str(work/'lfs_final_progress.txt'))
    phase='waiting_for_current_transfer'
    try:
        while alive(args.wait_pid):report(work,phase,tail(work/'lfs_upload_v2.log') or '正在上传全部模型与二进制结果');time.sleep(20)
        # Idempotent verification/resume skips already stored objects. Exit0 is required.
        report(work,'uploading_lfs','正在补齐并核验全部LFS对象')
        for attempt in range(1,4):
            log=work/f'lfs_finalize_attempt{attempt}.log'
            with log.open('wb') as f:p=subprocess.Popen(['git','-c','lfs.concurrenttransfers=16','lfs','push','--all','origin',BRANCH],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
            while p.poll() is None:report(work,'uploading_lfs',tail(log) or 'GitHub正在核对LFS对象',attempt=attempt);time.sleep(20)
            if p.returncode==0:break
            if attempt==3:raise RuntimeError('LFS upload failed after3 attempts; Git branch was not published')
            time.sleep(20)
        run(['git','lfs','fsck'],repo,env,work/'lfs_fsck.log')
        artifact=run(['git','rev-parse','HEAD'],repo,env);remote=run(['git','ls-remote','origin','HEAD','refs/heads/'+BRANCH],repo,env);heads={line.split()[1]:line.split()[0] for line in remote.splitlines() if line.split()}
        if heads.get('HEAD')!=EXPECTED_BASE:raise RuntimeError('Default branch changed since preflight; inspect before publishing')
        target=heads.get('refs/heads/'+BRANCH)
        if target and target!=artifact:
            ancestor=subprocess.run(['git','merge-base','--is-ancestor',target,artifact],cwd=repo,env=env,capture_output=True)
            if ancestor.returncode:raise RuntimeError('Target branch changed; refusing overwrite or force push')
        summary=repo/TASK/'publication_summary.json';v=json.loads(summary.read_text());v.update(status='publication_payload_verified',upload_status='LFS_objects_uploaded_and_checked',lfs_upload_exit_code=0,artifact_commit=artifact,observed_default_head_before_push=heads.get('HEAD'));write(summary,v)
        run(['git','add',TASK+'/publication_summary.json'],repo,env);run(['git','commit','--quiet','-m','Record successful full LFS asset transfer'],repo,env)
        local=run(['git','rev-parse','HEAD'],repo,env);report(work,'pushing_git','全部LFS对象已上传，正在发布实验分支')
        run(['git','push','--set-upstream','origin','HEAD:refs/heads/'+BRANCH],repo,env,work/'git_push.log')
        observed=run(['git','ls-remote','origin','refs/heads/'+BRANCH],repo,env).split()[0]
        if observed!=local:raise RuntimeError('Remote Git HEAD does not match publication commit')
        presence=metadata_verify(repo,env,work)
        # Independently download three representative binary artifacts to a fresh LFS store.
        verification=work/('remote_sample_checkout_'+str(int(time.time())));sample_env=dict(env,GIT_LFS_SKIP_SMUDGE='1')
        # A full Git object clone avoids Apple Git promisor/commit-graph failures.
        # Skip LFS smudging, then download only the three selected binary samples.
        run(['git','clone','--no-checkout','--single-branch','--branch',BRANCH,REPOSITORY,str(verification)],repo,sample_env,work/'remote_clone.log')
        manifest=json.loads((repo/'results/generated/soma_dense_switch_20261121/asset_manifest.json').read_text());byname={r['path']:r for r in manifest['files']};base='results/generated/soma_dense_switch_20261121/workspace/'
        names=[base+'soma_lowdim_switch_probe_20261117/results/lowdim_switch_001/shapes3d_z4_seed12/changing/checkpoint_final.pt',base+'soma_lowdim_switch_probe_20261117/results/lowdim_switch_001/dsprites_z2_seed10/ref1000/calibration_residuals.npz',base+'soma_dense_switch_windows_20261119/四数据集_逐步切换窗口图.zip']
        run(['git','config','--local','credential.https://github.com.helper',''],verification,env);run(['git','config','--local','--add','credential.https://github.com.helper','!gh auth git-credential'],verification,env)
        run(['git','lfs','install','--local'],verification,env);run(['git','lfs','fetch','--include='+','.join(names),'--exclude=','origin',BRANCH],verification,env,work/'remote_sample_download.log')
        samples=[]
        for name in names:
            row=byname[name];oid=row['sha256'];p=verification/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid
            if not p.is_file() or sha(p)!=oid:raise RuntimeError('Remote sample bytes differ from original manifest')
            samples.append(dict(path=name,bytes=row['bytes'],sha256=oid,remote_download_matches=True))
        pr=''  # Created with the GitHub connector after complete transfer verification.
        receipt=dict(status='complete',repository=REPOSITORY,branch=BRANCH,commit=local,verified_remote_commit=observed,pull_request_url=pr,archived_files=3863,archived_bytes=10243920790,remote_samples=samples,**presence)
        write(work/'upload_receipt.json',receipt);report(work,'complete','全部代码与实验结果已上传并核验。\n\n'+pr,**receipt)
    except BaseException as error:
        progress=work/'upload_progress.json';phase=json.loads(progress.read_text()).get('phase',phase) if progress.exists() else phase
        write(work/'upload_failure.json',dict(status='incomplete',error=str(error),phase=phase,traceback=traceback.format_exc()));report(work,'incomplete','上传或核验尚未完成，详见本地upload_failure.json；不声称全部已提交。');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--status-dir',required=True);p.add_argument('--lfs-dir',required=True);p.add_argument('--gh-dir',required=True);p.add_argument('--auth-dir',required=True);p.add_argument('--wait-pid',type=int,default=0);main(p.parse_args())
