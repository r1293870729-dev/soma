"""Pure-analysis background finalizer; never launches torch from a torch parent."""
from pathlib import Path
import json,time,traceback,zipfile
from datetime import datetime
from dense_common import ROOT,PLAN,cases,write,verify,sha
def progress():
    rr=[];errors=[];terminal=0;updates=0
    for ds in ['controlled','dsprites','shapes3d','cifar']:
        file=ROOT/'reports'/f'worker_{ds}.json';worker=json.loads(file.read_text()) if file.exists() else {};terminal+=worker.get('status') in ['complete','failed']
        if worker.get('status')=='failed':errors.append(dict(dataset=ds,error=worker.get('error')))
        for src,meta in cases(ds):
            folder=ROOT/'results'/src.name;summary=folder/'summary.json';pfile=folder/'progress.json';failed=folder/'FAILED.json';done=summary.exists();v=json.loads((summary if done else pfile).read_text()) if done or pfile.exists() else {}
            if failed.exists() and not any(e['dataset']==ds for e in errors):errors.append(dict(dataset=ds,error=json.loads(failed.read_text())['error']))
            n=int(v.get('replayed_updates',0));updates+=n
            rr.append(dict(case=src.name,dataset=ds,dim=meta['latent_dim'],seed=meta['seed'],status='complete' if done else 'failed' if failed.exists() else 'running' if pfile.exists() else 'pending',step=v.get('step',''),branch=v.get('branch',v.get('phase','')),replayed_updates=n))
    count=sum(r['status']=='complete' for r in rr);status='complete' if count==60 else 'failed' if errors and terminal==4 else 'running_with_error' if errors else 'running'
    supervisor=ROOT/'reports/SUPERVISOR.json'
    if supervisor.exists():
        s=json.loads(supervisor.read_text())
        if s.get('status')=='failed':errors.append(dict(dataset='supervisor',error=s['error']));status='failed'
    v=dict(status=status,updated_at=datetime.now().astimezone().isoformat(),complete_conditions=count,planned_conditions=60,replayed_updates=updates,planned_replayed_updates=227700,new_treatment_updates=0,errors=errors,cases=rr);write(ROOT/'reports/LIVE_PROGRESS.json',v)
    lines=['# 四数据集逐步补测进度','',f"更新时间：{v['updated_at']}；状态：{status}",'',f'完整复现并补测：**{count}/60条件**。原模型更新重放：**{updates:,}/227,700**；新增实验条件/权重调整为0。','', '每条件两次切换，各前后30步，共122个真实时点。图像另外补同父稳定分叉，不能与新的正式训练混称。','', '[四数据集图片](GALLERY.md)','', '| 条件 | 状态 | 分支/阶段 | 进度步 | 已重放更新 |','| --- | --- | --- | --- | --- |']
    lines+=['| '+' | '.join(str(r[k]) for k in ['case','status','branch','step','replayed_updates'])+' |' for r in rr]
    if errors:lines+=['','错误（不跳过）：',json.dumps(errors,ensure_ascii=False,indent=2)]
    (ROOT/'reports/LIVE_PROGRESS.md').write_text('\n'.join(lines)+'\n');return v
def pack():
    target=ROOT/'四数据集_逐步切换窗口图.zip';temporary=target.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['figures','reports']:
            for file in sorted((ROOT/folder).glob('*')):
                if file.is_file() and file.suffix in ['.png','.svg','.md','.csv','.json']:z.write(file,str(file.relative_to(ROOT)))
        z.write(ROOT/'configs/plan.json','configs/plan.json');z.write(ROOT/'docs/PROTOCOL.md','docs/PROTOCOL.md')
    with zipfile.ZipFile(temporary) as z:assert z.testzip() is None
    temporary.replace(target)
def main():
    verify();import render_dense
    last=None;rendered=0
    while True:
        p=progress();count=p['complete_conditions'];terminal=p['status'] in ['complete','failed']
        if last is None or count!=last and time.monotonic()-rendered>=180 or terminal:
            render_dense.main();last=count;rendered=time.monotonic();print(json.dumps(dict(event='render',conditions=count,status=p['status'])),flush=True)
        if terminal:
            verify();write(ROOT/'reports/completion_receipt.json',p);pack();return
        time.sleep(20)
if __name__=='__main__':
    try:main()
    except BaseException as e:write(ROOT/'reports/MONITOR_FAILED.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
