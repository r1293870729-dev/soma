"""Persistent pure-analysis monitor; does not import torch or start train subprocesses."""
import json,sys,time,traceback,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from study_paths import PLAN,write,sha,verify
def optional(file,default=None):return json.loads(file.read_text()) if file.exists() else default
def progress():
    done=0;updates=0;rr=[];errors=[];terminal=0
    for ds in PLAN['image_dimensions']:
        worker=optional(ROOT/'reports'/f'worker_{ds}.json',{})
        terminal+=worker.get('status') in ['complete','failed']
        if worker.get('status')=='failed':errors.append({'dataset':ds,'error':worker['error']})
        for seed in PLAN['image_seeds']:
            for dim in PLAN['image_dimensions'][ds]:
                case=ROOT/'results'/PLAN['run_id']/f'{ds}_z{dim}_seed{seed}';s=optional(case/'summary.json',{});p=optional(case/'progress.json',{});failed=optional(case/'FAILED.json',{})
                complete=s.get('status')=='complete';done+=complete
                n=5000 if complete else int(p.get('updates',0));updates+=n
                rr.append(dict(dataset=ds,dim=dim,seed=seed,status='complete' if complete else 'failed' if failed else 'running' if p else 'pending',branch=p.get('branch',''),step=p.get('step',''),updates=n))
    frozen=optional(ROOT/'reports/frozen_worker.json',{});count=sum(optional(f,{}) .get('status')=='complete' for f in (ROOT/'results/probe_replay').glob('*/summary.json'));controls=sum(optional(f,{}).get('status')=='complete' for f in (ROOT/'results/input_control').glob('*/summary.json'))
    if frozen.get('status')=='failed':errors.append({'worker':'frozen','error':frozen.get('error')})
    complete=done==18 and count==45 and controls==9
    status='complete' if complete else 'failed' if terminal==3 and (done<18 or frozen.get('status')=='failed') else 'running_with_error' if errors else 'running'
    v=dict(status=status,updated_at=datetime.now().astimezone().isoformat(),new_training_complete=done,new_training_planned=18,new_optimizer_updates=updates,planned_optimizer_updates=90000,update_fraction=updates/90000,reused_parent_conditions=42,complete_parent_conditions=42+done,planned_parent_conditions=60,frozen_replay_complete=count,frozen_replay_planned=45,input_controls=controls,input_controls_planned=9,errors=errors,cases=rr)
    write(ROOT/'reports/LIVE_PROGRESS.json',v);bar='█'*int(25*updates/90000)+'░'*(25-int(25*updates/90000))
    text=['# 2/4维补充与切换探针：实时进度','',f"更新时间：{v['updated_at']}；状态：{status}",'',f'新增模型更新：`{bar}` **{updates:,}/90,000（{100*updates/90000:.1f}%）**。',f'新增完整父条件：**{done}/18**；只读复用42；全维数共**{42+done}/60**。',f'冻结头重算：**{count}/45**；预训练图像特征控制：**{controls}/9**。','', '每个父条件包含3000步变化流和同父1000步分叉后的2000步S1稳定流；90,000统计实际模型更新，非只统计横轴步数。训练显示仍为三个1000步阶段。','', '图只纳入完整父条件；完成数变化后自动更新，最多数分钟延迟。PNG/SVG均保留，未完成不补零。','', '[查看清晰图片](GALLERY.md) · [实验解释与平衡准确率](RESULTS_AND_EXPLANATION.md)','', '| 数据集 | 维数 | seed | 状态 | 分支 | 到达步 | 已更新/5000 |','| --- | --- | --- | --- | --- | --- | --- |']
    text += ['| '+' | '.join(str(r[k]) for k in ['dataset','dim','seed','status','branch','step','updates'])+' |' for r in rr]
    if errors:text+=['','## 错误（保留，不跳过）','',json.dumps(errors,ensure_ascii=False,indent=2)]
    (ROOT/'reports/LIVE_PROGRESS.md').write_text('\n'.join(text)+'\n');return v
def main():
    verify();manifest=optional(ROOT/'configs/frozen_postprocessing.json',{})
    for file,h in manifest.get('files',{}).items():
        if sha(ROOT/file)!=h:raise ValueError('Postprocessing code changed '+file)
    import plots_switch,explain_results
    last=None;last_render=0
    while True:
        v=progress();signature=(v['new_training_complete'],v['frozen_replay_complete'],v['input_controls']);terminal=v['status'] in ['complete','failed']
        if last is None or signature!=last and time.monotonic()-last_render>=180 or terminal:
            cases=plots_switch.main();explain_results.main(cases);last=signature;last_render=time.monotonic();print(json.dumps(dict(event='rendered',**{k:v[k] for k in ['status','new_training_complete','frozen_replay_complete','new_optimizer_updates']})),flush=True)
        if terminal:
            verify();write(ROOT/'reports/monitor_complete.json',v);return
        time.sleep(20)
if __name__=='__main__':
    try:main()
    except BaseException as e:write(ROOT/'reports/monitor_failed.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
