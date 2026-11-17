"""Corrected four-dataset view: keep screenshot-controlled primary probes exactly."""
from pathlib import Path
import sys,json,csv,importlib.util,hashlib,shutil,zipfile,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;HIST=BASE/'soma_scheme_ab_reproduction_20261116';FULL=BASE/'soma_four_dataset_full_signals_20261116';FIG=ROOT/'figures';REPORT=ROOT/'reports';inputs={}
sys.path.insert(0,str(FULL/'src'))
spec=importlib.util.spec_from_file_location('prior_plot_helpers',FULL/'tools/plot_full.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
    p=Path(p);h=sha(p)
    if str(p) in inputs and inputs[str(p)]!=h:raise ValueError('Input changed')
    inputs[str(p)]=h;return p
def js(p):return json.loads(pin(p).read_text())
def numeric(p):return helpers.numeric(pin(p))
def records(p):return [json.loads(s) for s in pin(p).read_text().splitlines()]
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def export(fig,name):fig.savefig(FIG/(name+'.png'),dpi=260);fig.savefig(FIG/(name+'.svg'));plt.close(fig)

def controlled(dim):
    unit=ROOT/'results'/f'B_z{dim}';meta=js(unit/'summary.json');assert meta['status']=='complete' and meta['original_primary_theta_preserved'] and meta['original_checkpoint_hashes_verified']==15 and meta['initial_model_exact'] and meta['final_model_exact'];source=HIST/'results/reconstructed_B_001'/f'B_z{dim}_loss_plus_xi';old=numeric(source/'train_history.csv');source_probes=[r for r in numeric(source/'probe_history.csv') if r['task']=='theta'];replayed=[r for r in numeric(unit/'native/probe_history.csv') if r['task']=='theta'];assert len(source_probes)==len(replayed)==15
    for a,b in zip(source_probes,replayed):assert a['step']==b['step'] and a['accuracy']==b['accuracy']
    extended=numeric(unit/'online_theta_extensions.csv');primary=sorted([dict(step=r['step'],accuracy=r['accuracy'],origin='unchanged_original_online_point') for r in source_probes]+[dict(step=r['step'],accuracy=r['accuracy'],origin=r['origin']) for r in extended],key=lambda r:r['step']);assert len(primary)==18
    fixed=numeric(FULL/'results/historical'/f'B_z{dim}_loss_plus_xi'/'fixed_probe_full.csv');archive=[r for r in numeric(HIST/'archive_logs'/f'B_z{dim}_loss_plus_xi'/'probe_history.csv') if r['task']=='theta'];signals={age:records(unit/f'ref{age}/signals.jsonl') for age in [0,1000]};assert signals[0][0]['step']==1 and signals[1000][0]['step']==1001
    for rows in signals.values():
        for r in rows:assert r['updates_completed']==r['step']-1
    with (REPORT/f'controlled_z{dim}_original_theta_plus_extensions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(primary[0]));w.writeheader();w.writerows(primary)
    return dict(kind='historical',dim=dim,old=old,primary=primary,fixed=fixed,archive=archive,signals=signals,summary=meta)

def decorate(ax,old):
    ax.set_xlim(0,3000 if old else 4000)
    for t in [1000,2000] if old else [2000,3000]:ax.axvline(t,color='#444444',ls=':',lw=.8)
    ax.grid(alpha=.15)

def draw_signal(ax,item):
    right=ax.twinx();right.spines['top'].set_visible(False)
    if item['kind']=='historical':
        a=ax.plot([r['step'] for r in item['old']],[r['xi_pw'] if r['baseline_ready'] and r['window_ready'] else np.nan for r in item['old']],color='#dc2626',lw=1.3,label='原旧 Ξ（左轴）')[0];hs=[a]
        for age,color,marker in [(0,'#d97706','o'),(1000,'#2563eb','s')]:
            rr=item['signals'][age];hs.append(right.plot([r['step'] for r in rr],[r['psi'] for r in rr],color=color,marker=marker,ms=1.5,lw=1.3,label=f'新 Ψ ref{age}（右轴）')[0])
    else:
        cases=item['cases'];a=helpers.meanline(ax,[[(r['step'],r['xi'] if r['eligible'] else None) for r in c['changing']['old']] for c in cases],'#dc2626','原旧 Ξ（左轴）');hs=[a]
        for ref,color,marker in [('ref0','#d97706','o'),('ref1000','#2563eb','s')]:hs.append(helpers.meanline(right,[[(r['step'],r['psi']) for r in c['changing']['sig'][ref]] for c in cases],color,'新 Ψ '+ref+'（右轴）',marker=marker))
    ax.set_ylabel('旧 Ξ',color='#b91c1c');right.set_ylabel('新 Ψ',color='#1d4ed8');ax.tick_params(axis='y',colors='#b91c1c');right.tick_params(axis='y',colors='#1d4ed8');ax.set_ylim(bottom=0);right.set_ylim(bottom=0);decorate(ax,item['kind']=='historical');return hs

def draw_probe(ax,item,include_current_fixed=False):
    if item['kind']=='historical':
        p=item['primary'];original=[r for r in p if r['origin']=='unchanged_original_online_point'];extra=[r for r in p if r['origin']!='unchanged_original_online_point']
        ax.plot([r['step'] for r in p],[r['accuracy'] for r in p],color='#15803d',lw=1.5,label='原在线 θ：当前阶段池')
        ax.scatter([r['step'] for r in original],[r['accuracy'] for r in original],color='#15803d',s=18,zorder=4);ax.scatter([r['step'] for r in extra],[r['accuracy'] for r in extra],facecolors='white',edgecolors='#15803d',marker='D',s=26,zorder=5,label='补测0/1/3000：原拟合规则')
        archived=item['archive'];ax.plot([r['step'] for r in archived],[r['accuracy'] for r in archived],color='#999999',ls='--',lw=1.1,label='7月归档 θ：截图原虚线')
        rr=[r for r in item['fixed'] if r['view']=='fixed_S1'];ax.plot([r['step'] for r in rr],[r['accuracy'] for r in rr],color='#2563eb',ls='--',lw=1.3,label='固定 S1：另列补充回测')
        if include_current_fixed:
            rr=[r for r in item['fixed'] if r['view']=='current_stage'];ax.plot([r['step'] for r in rr],[r['accuracy'] for r in rr],color='#d97706',ls=':',lw=1.2,label='当前池：固定划分补测')
        ax.set_ylabel('θ八分类普通准确率')
    else:
        cases=item['cases'];tasks=sorted({r['task'] for c in cases for r in c['changing']['pr'] if r['probe_group']=='S1'})
        for i,task in enumerate(tasks):
            for readout,style in [('mu','-'),('sampled_z',':')]:helpers.meanline(ax,[[(r['step'],r['balanced_accuracy']) for r in c['changing']['pr'] if r['probe_group']=='S1' and r['task']==task and r['readout']==readout] for c in cases],plt.get_cmap('tab10')(i),helpers.TASK_NAMES[task]+' '+('μ' if readout=='mu' else 'z'),style,shade=readout=='mu')
        ax.set_ylabel('固定 S1 探针平衡准确率')
    ax.set_ylim(0,1.02);decorate(ax,item['kind']=='historical')

def panel(items,labels,name):
    n=len(items);fig,axes=plt.subplots(3,n,figsize=(6.8*n,12.2),squeeze=False,sharex=False);hs=[]
    for col,(item,label) in enumerate(zip(items,labels)):
        hs=draw_signal(axes[0,col],item);axes[0,col].set_title(label,fontsize=14,pad=10)
        if item['kind']=='historical':axes[1,col].plot([r['step'] for r in item['old']],[r['reconstruction'] for r in item['old']],color='#15803d',lw=1.3);axes[1,col].set_ylabel('原始重构 MSE（对数）')
        else:helpers.meanline(axes[1,col],[[(r['step'],r['pixel_mse']) for r in c['changing']['tr']] for c in item['cases']],'#15803d');axes[1,col].set_ylabel('像素 MSE（对数）')
        axes[1,col].set_yscale('log');decorate(axes[1,col],item['kind']=='historical');draw_probe(axes[2,col],item,include_current_fixed=n==2);axes[2,col].set_xlabel('原训练日志步数' if item['kind']=='historical' else '到达步 t（模型更新t−1次）');axes[2,col].legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('保持各自原训练轨迹：新旧信号同图，原探针口径不替换',fontsize=22,y=.985);fig.legend(hs,[h.get_label() for h in hs],loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False,fontsize=12)
    footer=('controlled严格对应截图原轨迹：z2/z4、seed0、1000/1000/1000步、rec+0.001KL+0.001旧Ξ；15个原θ点逐点一致。\n其他图像保持先前开发协议：3seed、2000/1000/1000步、rec+0.0001KL。各数据集只在自身轨迹内比较新旧信号，不作跨协议训练性能排名。\n红线读左轴，橙/蓝读右轴，不按峰高排名。controlled绿线=原当前阶段θ，蓝虚线=固定S1补充；其他三图像末排=固定S1 μ/z，阴影为seed范围。' if n>2 else '本图两列均为截图中原训练轨迹：seed0、1000/1000/1000步。绿色原15个θ测点一分不改；空心点补测0/1/3000。\n固定S1蓝虚线和固定划分当前池橙点线是补充回测，不能替代原绿线。新Ψ只旁路记录，不进入模型更新。\n信号计算于更新前，原在线探针评于更新后，沿用原日志步号；历史训练池80/20回测并非独立最终测试。')
    fig.text(.5,.022,footer,ha='center',fontsize=10,color='#555555');fig.subplots_adjust(top=.86,bottom=.22,hspace=.28,wspace=.43,left=.055,right=.95);export(fig,name)

def main():
    c2,c4=controlled(2),controlled(4);imgs=[dict(kind='image',cases=[helpers.load_unit(ds,s) for s in [10,11,12]]) for ds in ['dsprites','shapes3d','cifar']]
    panel([c2,c4],['截图原 controlled · z=2 · seed0','截图原 controlled · z=4 · seed0'],'01_controlled严格原轨迹_原绿线保留')
    panel([c2,c4]+imgs,['原 controlled z2 · 3000步','原 controlled z4 · 3000步','dSprites z32 · 4000步','Shapes3D z16 · 4000步','CIFAR z64 · 4000步'],'02_四数据集原轨迹_含controlled两种维数')
    # Existing image probes and large plots are already correct; copy their unchanged exports.
    for ds in ['dsprites','shapes3d','cifar']:
        for stem in [f'大图_{ds}_信号重构与S1探针',f'探针_{ds}_全部任务']:
            for suffix in ['.png','.svg']:
                f=pin(FULL/'figures'/(stem+suffix));shutil.copy2(f,FIG/f.name)
    for p,h in {**helpers.inputs,**inputs}.items():
        if sha(p)!=h:raise ValueError('Source changed during corrective plotting')
    write(REPORT/'input_manifest.json',{**helpers.inputs,**inputs});write(REPORT/'verification.json',dict(status='complete',controlled_dimensions=[2,4],controlled_source='same original screenshot parents reconstructed_B_001',original_primary_probe_points_unchanged=30,original_checkpoint_hashes_exact=30,original_final_models_exact=True,new_signal_enters_loss=False,controlled_manual_calibration='independent new S1 synthetic source namespaces using original generator and embedding',image_sources_unchanged=True,controlled_audits=[c2['summary'],c4['summary']],new_treatment_training_updates=0,plot_code_sha256=sha(Path(__file__))))
    lines=['# controlled为什么不同，以及修正版','',
    '先前四数据集主图用了新的controlled开发协议，和截图的历史controlled不是同一次实验。虽然都叫θ探针，但生成器、训练、评估池、分类头与指标同时不同；不能把差异归因于新旧信号。将其放在“补齐截图”上下文中容易误解，本修正版纠正该对应关系。','',
    '| 项目 | 用户截图原controlled | 先前四数据集开发controlled |','|---|---|---|',
    '| 数据流 | circle→torus→关联局部四维分支 | product/mixed Fourier、矩匹配独立分支 |',
    '| 阶段 | 1000/1000/1000 | 2000/1000/1000 |',
    '| 目标 | rec+0.001KL+0.001旧Ξ | rec+0.0001KL，信号只旁路 |',
    '| 模型与seed | z2/z4、seed0 | 主z4、seed10/11/12均值 |',
    '| 下排探针 | 当前阶段θ，80/20每次重新划分 | 固定S1独立角色θ，μ/z分别读出 |',
    '| 指标/分类头 | 普通准确率，历史AdamW与early-stop | 平衡准确率，验证选L2、LBFGS线性头 |','',
    '## 本修正版保持什么','',
    '严格使用截图的两条原训练轨迹；原始15个在线θ点、所有原日志标量、15个保存模型及最终模型全部匹配。新信号只在模型更新前的同一批残差上旁路计算，权重、数据、seed、阶段和原探针均不改变。绿色主曲线仍是原在线当前阶段θ；固定S1与固定划分当前池补测另列，不替代绿线。','',
    '原口径初始0、首次更新后1、最终3000另外补测，使用原分类器与原seed规则；原15点没有重拟合替换。图中空心点标记新增离线评价。固定S1补充使用此前完整18点，仍是历史训练池80/20，不当独立最终测试。','',
    '## 新手册信号怎样加入','',
    '不改变原训练数据。使用同一个原S1生成器与相同embedding，按固定独立seed命名空间生成white512、edge2048、edge-cover2048；三角色互斥，和原训练seed命名空间不同。0/1000参考共享这些独立源与固定潜噪声。采用已确认手册W64/alpha0.1/beta0.05/float64/floor1e-12/负谱容差1e-10；相对谱质量和、固定白化、经验edge。校准前后强制模型和全局Torch RNG不变。','',
    '窗口取原batch128最后64个残差，与旧current-window读数取样一致；测点为原日志实际记录点。ref0从1批，ref1000从1001批，之前不倒填。解码器输出是在本次潜变量样本条件下的重构均值，校准与在线一致。新独立校准源和工程数值设置见configs/plan.json，不能把它们宣称为丢失7月代码中的原始设置。','',
    '原7月源码缺失限制仍在：这里保留的是本次已经重建并展示给用户的相同父轨迹，不声称恢复未知原版逐字源码。','',
    '## 四数据集如何展示','',
    '主图仍包含四个数据集，controlled的两种维数各一列，三图像各一列。各自使用原有阶段长度和训练协议，横轴明确区分3000与4000步，不再为了排版替换controlled父轨迹。跨数据集不能把权重、阶段、探针和MSE直接作公平方法排名。','',
    '- figures/01_controlled严格原轨迹_原绿线保留.png：原controlled两维，新旧信号及原绿线。',
    '- figures/02_四数据集原轨迹_含controlled两种维数.png：四数据集原轨迹总览。',
    '- figures/大图_*、探针_*：三个图像数据集不变的清晰单图与全部任务。',
    '- reports/controlled_z*_original_theta_plus_extensions.csv：原θ点和新增测点逐条注明来源。',
    '- reports/verification.json：原模型、训练日志与原探针的完全一致性。','']
    (REPORT/'修正说明.md').write_text('\n'.join(lines));(ROOT/'README.md').write_text('# 四数据集原轨迹修正版\n\ncontrolled严格对应用户截图的原父轨迹；不以新开发协议替换。详见reports/修正说明.md。PNG/SVG与核验、补充测量、代码均已保存。\n')
    files=[f for folder in ['src','tools','configs','reports','figures','results'] for f in (ROOT/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.partial']]+[ROOT/'README.md'];dest=ROOT/'四数据集原轨迹修正版.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in files:z.write(f,f.relative_to(ROOT))
    with zipfile.ZipFile(dest) as z:
        if z.testzip():raise ValueError('Zip CRC failed')
    print(json.dumps(dict(status='complete',controlled_original_theta_points_unchanged=30,zip=str(dest),pngs=len(list(FIG.glob('*.png')))),ensure_ascii=False))
if __name__=='__main__':main()
