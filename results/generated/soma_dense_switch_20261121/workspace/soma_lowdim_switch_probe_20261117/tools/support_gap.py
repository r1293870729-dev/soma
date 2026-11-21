"""Compare fixed/current support at identical old checkpoints, not temporal forgetting."""
from pathlib import Path
import csv
import numpy as np
import plots_switch as p
ROOT=Path(__file__).resolve().parents[1]
def main():
    output=[]
    for c in p.load():
        if c['dataset']=='controlled' or c['summary'].get('new_lowdim'):continue
        fixed=dict(p.get(c,'probe',kind='fixed'));current=dict(p.get(c,'probe',kind='current'))
        for t in [1001,2001]:
            if t not in fixed or t not in current:continue
            output.append(dict(dataset=c['dataset'],dim=c['dim'],seed=c['seed'],step=t,fixed_S1_BA=fixed[t],current_support_BA=current[t],current_minus_fixed_pp=100*(current[t]-fixed[t]),same_preupdate_model=True,tasks='same original S1 tasks averaged; refitted heads on different group train/val/test'))
    with (ROOT/'reports/existing_same_model_support_gap.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    text=['# 同一检查点：换评估支持就会改变分数','', '本表只读原27条完整图像轨迹；新低维不纳入。每个时点比较同一更新前模型的固定S1测试池与刚到达S2/S3测试池，任务只聚合相同的原S1任务。各组分别用各自独立train/val拟合头、在test测BA。这里不是同一固定测试集的时间变化，不能称旧任务遗忘。','', '| 数据集 | 维数 | 到达步 | seed数 | 固定S1 BA | 当前支持BA | 当前减固定（百分点） |','| --- | --- | --- | --- | --- | --- | --- |']
    keys=sorted({(r['dataset'],r['dim'],r['step']) for r in output})
    for ds,d,t in keys:
        rr=[r for r in output if (r['dataset'],r['dim'],r['step'])==(ds,d,t)];a,b,g=[float(np.mean([r[k] for r in rr])) for k in ['fixed_S1_BA','current_support_BA','current_minus_fixed_pp']]
        text.append(f'| {ds} | {d} | {t} | {len(rr)} | {100*a:.2f}% | {100*b:.2f}% | {g:+.2f} |')
    text += ['', '例如dSprites32维在第一次切换同一模型下，固定S1约56.95%、当前S2约36.34%，差约20.62个百分点。模型尚未用S2完成第一次更新，差异来自输入支持与头拟合/评估角色变化，不是这一瞬间训练造成了遗忘。CIFAR32维的当前支持分数还略高于S1；所有正负结果均保留。', '', '可见结构改变对当前任务难度确实有影响；论文若预测旧任务遗忘，需要另外看固定S1随训练的变化、稳定流校正和原重拟合探针，而非只选择更容易下降的评估支持。']
    (ROOT/'reports/existing_same_model_support_gap_note.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':main()
