"""Read-only progress from worker artifacts; imports no training libraries."""
import json,datetime,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/reconstructed_B_001'
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}
rows=[];formal=cal=done=failed=0
for d in [2,4,8,16,32]:
    for mode in ['loss_plus_xi','loss_only']:
        name=f'B_z{d}_{mode}';out=RUN/name;p=read(out/'progress.json');e=read(out/'execution.json');q=read(out/'fixed_probe_progress.json')
        f=int(e.get('formal_model_updates',p.get('step',0)));c=int(e.get('separate_calibration_model_updates',p.get('calibration_rows',0)));formal+=f;cal+=c
        status='待运行'
        if (out/'FAILED.json').exists():status='训练失败';failed+=1
        elif e.get('status')=='complete':
            if q.get('status')=='complete':status='完成';done+=1
            else:status=f'训练完成，离线探针 {q.get("rows",0)}/10'
        elif p:status=f'{p.get("phase","running")}：{p.get("message","")}'
        rows.append(f'| {d} | {mode} | {f}/3000 | {c}/3000 | {status} |')
text='\n'.join(['# 方案 B 历史协议重建进度','',f'更新时间：{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}',f'已完成 {done}/10，训练失败 {failed}。正式模型更新 {formal}/30000；独立校准模型更新 {cal}/30000。','',
'单 worker、单 torch 线程；另一个图像实验继续运行。此处是报告协议重建，原始 7 月源码不可得。',
'','| 维数 | 目标 | 正式更新 | 校准更新 | 状态 |','|---|---|---|---|---|',*rows,'',
'每组固定 seed 0。loss_only 与 loss_plus_xi 使用相同历史参数；不按结果调参。探针离线回测不更新编码器。',''])
target=ROOT/'reports/RECONSTRUCTION_PROGRESS.md';tmp=target.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(target)
print(text)
