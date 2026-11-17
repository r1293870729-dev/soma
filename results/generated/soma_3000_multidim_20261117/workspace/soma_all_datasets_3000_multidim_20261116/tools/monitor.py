"""Persistent progress and automatic plotting, no training subprocesses."""
from pathlib import Path
import sys,json,time,datetime,hashlib,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from paths import PLAN,write,sha
import render
DS=['controlled','dsprites','shapes3d','cifar'];RUN=ROOT/'results'/PLAN['run_id']
def js(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def snapshot():
    entries=[];done=0;updates=0;failures=[]
    for ds in DS:
        dims=PLAN['controlled_dimensions'] if ds=='controlled' else PLAN['image_dimensions'][ds];seeds=PLAN['controlled_seeds'] if ds=='controlled' else PLAN['image_seeds'];dones=0;active=[]
        for seed in seeds:
            for dim in dims:
                name=f'{ds}_z{dim}_seed{seed}';folder=RUN/name;s=js(folder/'summary.json');p=s or js(folder/'progress.json');failure=js(folder/'FAILED.json')
                if s.get('status')=='complete':done+=1;dones+=1
                if ds=='controlled':
                    if s:updates+=0 if s.get('reuse') else s.get('formal_updates',0)+s.get('separate_calibration_updates',0)
                    else:updates+=p.get('step',0)+p.get('calibration_updates',0)
                else:updates+=p.get('actual_optimizer_updates',p.get('updates',0))
                if p and not s:active.append(f'z{dim}/seed{seed} {p.get("phase",p.get("branch",""))} step{p.get("step",0)}')
                if failure:failures.append(dict(unit=name,**failure))
        worker=js(ROOT/'reports'/f'worker_{ds}.json');state=worker.get('status','running' if active else 'pending')
        if state not in ['complete','failed']:
            log=ROOT/'logs'/f'train_{ds}.log'
            if log.exists() and 'Traceback (most recent call last)' in log.read_text()[-15000:]:state='failed';failures.append(dict(unit=ds,error='See worker log traceback'))
        entries.append(dict(dataset=ds,complete=dones,planned=len(dims)*len(seeds),status=state,active=active))
    status='complete' if done==42 and not failures else 'failed' if all(e['status'] in ['complete','failed'] for e in entries) else 'running'
    result=dict(status=status,completed_parents=done,planned_parents=42,new_optimizer_updates=updates,planned_new_optimizer_updates=213000,entries=entries,failures=failures,updated_at=time.time())
    write(ROOT/'reports/progress.json',result)
    now=datetime.datetime.now().astimezone().isoformat(timespec='seconds');lines=['# 3000步多维度实验实时进度','',f'更新时间：{now}',f'父条件：**{done}/42 完成**；新增实际更新：{updates:,}/213,000（不含资格试跑300步）。','',
        '| 数据集 | 完成 | 状态 | 当前任务 |','|---|---:|---|---|']
    for e in entries:lines.append(f'|{e["dataset"]}|{e["complete"]}/{e["planned"]}|{e["status"]}|'+('; '.join(e['active']) or '—')+'|')
    lines+=['','阶段为1000/1000/1000，图像S1稳定对照从同一1000步模型与Adam状态分叉。共42个父条件，稳定分支不是独立seed。','controlled seed0的2/4维为已核验结果复用，其余维数只读重放补充新信号；新增seed1/2条件独立运行。','',
       '- [目前结果与图像目录](RESULTS.md)','- [四数据集多维度总览](../figures/00_四数据集_全部维度总览.png)','- [协议与维数](../docs/PROTOCOL.md)','',
       '自动出图进程每20秒刷新进度，完成条件增加后更新所有图。图中n不足3或空列表示尚未完成；不补零、不挑seed。最终全量结果完成后自动打包代码、配置、图表和表格。']
    if failures:lines+=['','## 错误（保留计划分母）','',json.dumps(failures,ensure_ascii=False,indent=2)]
    (ROOT/'reports/LIVE_PROGRESS.md').write_text('\n'.join(lines)+'\n');return result
def main():
    manifest=js(ROOT/'configs/analysis_frozen.json')
    for name,h in manifest.items():
        if sha(ROOT/name)!=h:raise ValueError('Analysis code changed '+name)
    last=-1
    while True:
        status=snapshot();n=status['completed_parents']
        if n!=last or status['status'] in ['complete','failed']:
            render.main();last=n;print(json.dumps(status,ensure_ascii=False),flush=True)
        if status['status'] in ['complete','failed']:write(ROOT/'reports/FINAL_STATUS.json',status);break
        time.sleep(20)
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:write(ROOT/'reports/MONITOR_STOPPED.json',dict(status='interrupted',updated_at=time.time()));raise
    except BaseException as exc:write(ROOT/'reports/ANALYSIS_FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
