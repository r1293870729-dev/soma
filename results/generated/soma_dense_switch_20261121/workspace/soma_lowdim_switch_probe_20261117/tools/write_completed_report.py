"""Readable synthesis with complete tables, source links, and limitations."""
from pathlib import Path
import csv,json
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/final_analysis_20261119'
def rr(name):return list(csv.DictReader((OUT/name).open()))
def one(rows,**keys):
    matches=[r for r in rows if all(str(r[k])==str(v) for k,v in keys.items())]
    if len(matches)!=1:raise ValueError(str(keys))
    return matches[0]
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    f=rr('final_by_dimension.csv');e=rr('transition_macro_by_dimension.csv');s=rr('signals_by_dimension.csv');v=rr('S1_validation_by_dimension.csv');di=rr('numerical_diagnostics.csv');co=rr('controlled_by_dimension.csv');fa=rr('S1_factor_by_dimension.csv');dims={'dsprites':[2,4,8,16,32],'shapes3d':[2,4,16,32,64],'cifar':[2,4,16,32,64]};names={'dsprites':'dSprites','shapes3d':'Shapes3D','cifar':'CIFAR/CIFAR-C'}
    def fv(ds,dim,task,key='changing_3000_mean'):return f"{100*float(one(fa,dataset=ds,dim=dim,readout='mu',task=task)[key]):.2f}"
    status=json.loads((ROOT/'reports/LIVE_PROGRESS.json').read_text());assert status['status']=='complete'
    mainrows=[]
    for ds in ['controlled','dsprites','shapes3d','cifar']:
        row=['controlled θ普通准确率' if ds=='controlled' else names[ds]+' 固定S1 BA（μ）']
        for dim in [2,4,8,16,32,64]:
            found=[r for r in (co if ds=='controlled' else f) if int(r['dim'])==dim and (ds=='controlled' or r['dataset']==ds and r['readout']=='mu')]
            row.append(f"{100*float(found[0]['theta_final_accuracy_mean' if ds=='controlled' else 'S1_refit_3000_mean']):.2f}%" if found else '—')
        mainrows.append(row)
    transition=[];final=[]
    for ds,dd in dims.items():
        for d in dd:
            a=one(e,dataset=ds,dim=d,readout='mu',kind='fixed',cut=1000);b=one(e,dataset=ds,dim=d,readout='mu',kind='fixed',cut=2000);z=one(f,dataset=ds,dim=d,readout='mu')
            fmt=lambda r:f"{float(r['paired_increment_pp_mean']):+.2f} [{float(r['paired_increment_pp_min']):+.2f}, {float(r['paired_increment_pp_max']):+.2f}]"
            transition.append([names[ds],d,fmt(a),fmt(b)])
            final.append([names[ds],d,f"{100*float(z['S1_refit_3000_mean']):.2f}%",f"{100*float(z['stable_refit_3000_mean']):.2f}%",f"{float(z['refit_final_changing_minus_stable_pp_mean']):+.2f}",f"{100*float(z['S1_frozen_3000_mean']):.2f}%",f"{100*float(z['stable_frozen_3000_mean']):.2f}%"])
    flags=[]
    for r in di:
        if float(r['max_clip'])>0:flags.append([names[r['dataset']],r['dim'],r['seed'],r['branch'],f"{100*float(r['max_clip']):.3f}%",r['max_pixel_mse_step'],f"{float(r['max_pixel_mse']):.5f}"])
    coverage=rr('reference_coverage.csv');ex=sum(int(r['exceedances']) for r in coverage);total=sum(int(r['windows']) for r in coverage)
    content=['# 完整训练结果分析：低维补充、结构切换与下游探针','',
    '分析日期：2026年11月19日。训练完成于2026年11月18日03:13（北京时间）。本次只读分析，没有重新训练或改动历史结果。','',
    '**主要结论：容量对“能否学会/线性读出任务”的影响很清楚；“每次结构切换都造成明显旧任务准确率下降”没有在四数据集上普遍出现。dSprites 4维有较晚的相对性能损失，Shapes3D中高维首次切换有小幅短期下降，CIFAR变化很小。冻结头的大幅下降含有表示漂移，旧Ξ的尖峰还受已知stage基线切换机制影响。**','',
    '## 完成与分析口径','',
    '- 新增18/18个2/4维父条件完成，新增90,000次模型更新；原42个完整条件只读复用，共60个条件、20个数据集×维数组合，各3个seed。',
    '- 图像45/45个父条件均有1000步同模型/Adam分叉的S1稳定流；冻结探针45/45完成，预训练图像特征控制9/9完成。没有运行失败记录；数值稳定性问题另列。',
    '- 每条变化流仍为1000+1000+1000步。图像重构权重1、KL0.0001；controlled保留历史rec+0.001KL+0.001Xi。图像新旧信号均旁路记录。',
    '- 主表确定性读出使用μ，并完整保留采样z结果；不按维数或seed挑选“最好”的读出。图像末次探针在到达步3000、更新2999次时测量；验证在更新3000次后测量。',
    '- 固定S1主表对原S1任务等权平均各任务BA：dSprites为x/y/scale，Shapes3D为三种颜色/scale，CIFAR为类别。其机会水平分别约7.64%、10.625%、10%。这不是各数据集的同一个任务，不能跨行直接排名。',
    '- controlled仍为历史当前阶段θ普通准确率、训练池80/20回测，不能与图像独立固定S1探针混为同一遗忘指标。',
    '- 所有图表采用全部3seed均值与最小—最大范围；不宣称置信区间、显著性或确认性因果结论。','',
    '## 一、补2/4维以后发生了什么','',table(['末次读出','2维','4维','8维','16维','32维','64维'],mainrows),'',
    '“—”是原协议没有设置的维数，不是失败或缺失运行。图像用固定S1重拟合线性头BA；controlled用历史当前θ普通准确率。全部逐任务μ/z结果见 ../../figures/final_analysis_20261119/02_各维数_全部主任务_mu与z.png。','',
    '**controlled：** 2维末期θ准确率均值74.23%，seed范围65.37–90.98%；4维94.96%，8/16/32维约95%。这符合当前合成任务中低维瓶颈更明显的现象，但历史回测口径不能独立证明旧任务遗忘。','',
    '**dSprites：** 2维固定S1仅14.91%，4维41.18%，8/32维约50%。提高维数帮助部分因子，但并非单调提升：16维低于8/32，且有一条changing轨迹出现logvar裁剪。新方向任务在所有维数仍弱，不能把高维S1分数当成方向已学好。','',
    f'**Shapes3D：** μ的固定S1平均随容量从54.08%升至95.90%，容量影响最清楚。2维并非四种旧因子都学会：末期地面颜色{fv("shapes3d",2,"floor_hue")}%、墙面颜色{fv("shapes3d",2,"wall_hue")}%，但物体颜色9.73%（随机10%）、尺度11.90%（随机12.5%）。在1000步S1末，物体颜色约9.8%、尺度12.1%，已经弱。这更接近“优先编码背景颜色、其他因子未有效读出”，不能描述成这些因子后来才被遗忘。','',
    '**CIFAR：** 2/4维固定S1类别BA为17.27%/19.95%，16/32/64维约27%。模型优化重构+KL，类别不是训练目标。相同角色/头拟合器的监督预训练ResNet18特征控制为76.24%，支持VAE表示与语义读出不匹配这一解释；其有额外监督，不能当公平算法排名。32维只在本批μ均值略高，不宣称统计上优于16/64维。','',
    '## 二、老师要求的“结构切换后任务变化”有多大','',
    '按原冻结窗口计算：第一次预点900/950/1000，后点1001/1050/1100；第二次预点1900/1950/2000，后点2001/2050/2100。效应（百分点）=100×[(变化流后−前)−(稳定流后−前)]，使用同一固定S1任务、μ、每次重拟合线性头。负数表示切换流相对下降。','',
    table(['数据集','维数','首次切换：均值 [seed范围]，pp','第二次切换：均值 [seed范围]，pp'],transition),'',
    'CIFAR所有维数、两次切换的均值都在1个百分点以内；Shapes3D 4/16/32/64维首次切换分别约−0.78/−1.85/−0.94/−2.24个百分点，2维则约+0.59。第二次切换未出现一致下降。dSprites多数情况变化小或正向；8维是相对改善，32维第二次切换是很小的下降。','',
    '这批结果没有普遍复现“信号一跳，固定旧任务就立即大跌”。探针在当批更新前测量：1001步模型尚未对新阶段做第一次更新，信号已经读到新阶段输入，因此两者本来也不必同一步跳变。','',
    '### dSprites 4维：较晚的相对损失值得单独看','',
    f'末期固定S1重拟合BA：变化流41.18%、稳定流47.38%，配对差−6.21个百分点，三个seed分别约−5.86/−3.76/−9.00，均低于对应稳定流。主要来自尺度（{fv("dsprites",4,"scale")}% vs{fv("dsprites",4,"scale","stable_3000_mean")}%，−14.14pp）和y（{fv("dsprites",4,"y")}% vs{fv("dsprites",4,"y","stable_3000_mean")}%，−7.52pp），x反而约+3.04pp。','',
    '这是本批较清楚的“长期变化流相对稳定流劣化”，但与即时切换峰不同：原定短窗口首次约+0.98pp、第二次约−0.02pp。相对于自身1000步，末期平均约−3.48pp，三个seed的自身时间变化并非全为下降。应报告其延迟、因子选择性和seed差异。','',
    '## 三、冻结探针为何掉得很明显，原探针却没有','',
    '冻结头固定了900步时的标准化均值/标准差与线性权重。若编码坐标旋转、平移或缩放，即使信息仍在，旧头也会失配；原重拟合头能适应这些变化。因此冻结头主要诊断“原读出器还可否直接用”，不能单独当信息丢失或遗忘证据。','',
    table(['数据集','维数','重拟合/变化','重拟合/稳定','配对差pp','冻结/变化','冻结/稳定'],final),'',
    '例：Shapes3D 4维，重拟合末期76.93% vs稳定77.26%，仅−0.33pp；冻结头却40.84% vs稳定55.69%，约−14.85pp。可以确认原读出器失配更明显，但旧任务线性信息仍大体可读出。','',
    'Shapes3D 64维更能说明时间漂移：冻结μ头从900步平均89.18%到1000步59.84%，在第一次结构切换前就下降约29.33pp。末期稳定流冻结头也只约18.13%。所以不能把冻结头整条下降都归因于结构切换。','',
    '## 四、新任务未学会，与旧任务被遗忘要分开','',
    'Shapes3D S3条件方向任务（形状0、15类，随机6.67%）μ末期：2维5.87%、4维7.73%、16维71.97%、32维80.48%、64维75.93%。2/4维近机会水平，中高维明显可读，说明容量对新因子学习存在强影响。这是“新任务是否学会”的证据，不是旧任务从高分下降的证据；S3到达前该任务不定义，不能补零来造学习曲线。','',
    'dSprites心形方向8类（随机12.5%）μ在2/4/8/16/32维约13.81/15.56/14.63/15.57/17.17%，所有维数都偏弱。真实像素标签审计未发现此条件主任务的同像素冲突，上限为100%，因此不能单纯归咎于图像方向不可识别。重构训练目标、有限S3更新和线性可读性是候选机制，当前结果未逐项识别原因。','',
    'μ和z还可能给出不同维数排序：Shapes3D固定S1平均μ在64维约95.90%，采样z却46.84%；16维z约83.87%。z包含后验噪声，维数与噪声分配也影响可读性。报告必须同时保留两者，不能挑其中更符合预期的曲线。','',
    '## 五、新旧信号反映什么，这次能支持论文哪一部分','',
    '新Ψ ref1000在独立S1角色上校准并保持参考固定。它在dSprites/Shapes3D切换时响应明显，例如Shapes3D首次切换后窗口的变化流−稳定流Ψ约5.40–6.80，而固定S1重拟合BA只小幅变化；它测量的是固定参考下的残差谱偏离，不能直接换算成旧任务准确率下降。不同维数拥有自己的白化参考，信号幅度也不是统一容量刻度。','',
    '旧Ξ的尖峰包含历史stage机制。检查全部45条图像轨迹的两次切换（90次事件）发现：每次最初20批使用旧stage基线fallback；到1021/2021步切为当前stage基线。该实现接收训练阶段stage，尖峰和衰减受状态路由及基线适应影响。它是有效的历史机制对照，但不能仅凭尖峰漂亮认定它更能盲检测未知切换，或更能预测下游下降。源码见 [observer.py](../../../soma_image_schemeB_vs_manual_20261116/src/observer.py)。','',
    '此外本批比较的是两条整管线：新信号使用冻结ResNet18的512维h(x)−h(x̂)，旧信号使用128维投影像素残差及历史stage基线。虽然模型轨迹相同，差异仍包括残差读出空间和参考更新机制，不能全归因于某一个谱公式。','',
    '描述性散点图保留所有seed：横轴为新Ψ后窗口变化流−稳定流，纵轴为固定S1 BA配对变化。不合并跨数据集拟合相关性，不把同一轨迹的两个事件当独立统计样本，也没有新增AUROC或在线误报率声明。','',
    f'校准补充：90个图像参考各有32个不重叠edge-cover窗口，其原图角色与拟合角色独立，共{ex}/{total}（{100*ex/total:.2f}%）越界，单参考最大7/32（21.875%）。这仅为固定快照覆盖检查；跨参考窗口复用了部分原图，不能把总计当独立在线误报保证。图像信号日志8505条均标为valid。','',
    '## 六、数值质量：运行完成不等于所有对照都稳定','',
    '原模型logvar裁剪范围为[−12,12]。下表是观测日志点出现过裁剪的分支；不删除这些seed。裁剪按每50步等诊断点记录，不能推断其余时刻完全无裁剪。MSE最大值按完整分支统计，changing的初始峰可能只是未训练模型，不能单凭这个峰判异常。','',
    table(['数据集','维数','seed','分支','观测最大裁剪比例','MSE最大时点','分支最大MSE'],flags),'',
    '尤其Shapes3D16维seed12稳定流：约1900步有100% logvar裁剪，训练MSE峰约0.366（1869步）；64维seed11稳定流：约2800步有100%裁剪，MSE峰约0.326（2740步），末期固定S1验证MSE约0.0581，另外两个seed约0.0015–0.0016。','',
    '因此高维Shapes3D中“变化流优于稳定流”的某些均值不能直接解释为正迁移或因果优势。完整表仍保留全部seed及范围，数值异常单独标注。本轮没有为改善图形修改梯度裁剪、学习率、KL或历史logvar实现。','',
    '## 可对负责人汇报的结论','',
    '1. 已完成全部维数与配对对照；容量不足会让一些因子从一开始就难以读出，2/4维并不会自动产生“先高分、切换后大幅遗忘”的轨迹。',
    '2. Shapes3D新方向任务最清楚地体现容量门槛；dSprites4维有延迟且因子选择性的相对劣化；CIFAR主要表现为总体语义探针较弱、切换影响小。',
    '3. 固定S1、当前阶段、冻结头是三种不同问题。当前分布变难、冻结头失配、旧信息不可读必须分别论证。',
    '4. 新信号确有残差结构响应，但本批未建立“信号幅度普遍预测旧任务准确率损失”的关系；旧信号还包含已知stage路由，不能凭尖峰形状选择胜者。',
    '5. 当前属于开发/探索结果，且有对照分支数值异常。尚未完成手册Q1–Q3的确认性验证、匹配在线误报率或足够重复数的统计推断。','',
    '## 图表与可复现数据','',
    '- [四数据集容量、探针与同S1重构](../../figures/final_analysis_20261119/01_四数据集_维数_探针_重构.png) · 同名SVG。',
    '- [全部主任务μ/z热图](../../figures/final_analysis_20261119/02_各维数_全部主任务_mu与z.png) · 同名SVG。',
    '- [两次切换的配对探针变化](../../figures/final_analysis_20261119/03_两次切换_稳定流校正_探针变化.png) · 同名SVG。',
    '- [新信号与固定S1探针变化](../../figures/final_analysis_20261119/04_新信号与固定S1探针变化.png) · 同名SVG。',
    '- [原全部时序图](../GALLERY.md)：信号/MSE/各维数/各因子及稳定流均保留。',
    '- 本目录CSV包含逐seed、聚合、各因子、S1验证MSE、信号、stage路由、裁剪与覆盖；input_manifest.json记录原始依赖哈希。',
    '- 分析源码：../../tools/analyze_completed.py、../../tools/plot_completed_analysis.py、../../tools/write_completed_report.py。原训练/回放/绘图冻结清单不变。','']
    (OUT/'REPORT.md').write_text('\n'.join(content))
    print('Wrote final scientific interpretation report.')
if __name__=='__main__':main()
