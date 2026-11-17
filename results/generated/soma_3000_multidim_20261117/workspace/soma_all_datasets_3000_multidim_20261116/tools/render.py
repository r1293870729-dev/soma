"""Read completed cases only; retain all planned dimensions and seeds in figures."""
from pathlib import Path
import sys,json,csv,hashlib,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from paths import PLAN,HIST,read,sha,write,verify
font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']='Arial Unicode MS'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path','axes.unicode_minus':False})
FIG=ROOT/'figures';REPORT=ROOT/'reports';RUN=ROOT/'results'/PLAN['run_id'];inputs={};flat=[]
NAMES={'controlled':'controlled','dsprites':'dSprites','shapes3d':'Shapes3D','cifar':'CIFAR / CIFAR-C'}
TASKS={'x':'位置 x','y':'位置 y','scale':'尺度','shape':'形状','orientation':'方向（条件主任务）','orientation_shape2_bin8':'方向（shape2，八分类）','orientation_shape0':'方向（shape0，十五分类）','orientation_raw':'方向（原始辅助）','floor_hue':'地面颜色','wall_hue':'墙面颜色','object_hue':'物体颜色','class':'类别','theta':'角度 θ'}
DIMS={'controlled':PLAN['controlled_dimensions'],**PLAN['image_dimensions']}
def pin(p):
    p=Path(p);h=sha(p)
    if str(p) in inputs and inputs[str(p)]!=h:raise ValueError('Completed input changed '+str(p))
    inputs[str(p)]=h;return p
def rows(p):return read(pin(p)) if Path(p).exists() else []
def records(p):return [json.loads(s) for s in pin(p).read_text().splitlines() if s] if Path(p).exists() else []
def truth(v):return str(v).lower() in ['true','1','1.0']
def xy(rr,key,pred=lambda r:True):return [(int(r['step']),float(r[key])) for r in rr if pred(r) and r.get(key,'') not in ['',None]]
def probe_points(rr,task,stage,readout):
    # Conditional and raw orientation have custom group names, but both belong to S3.
    return xy(rr,'balanced_accuracy',lambda r:int(r['probe_stage'])==stage and r['task']==task and r['readout']==readout)
def metric(c,b,k,v):
    for t,y in v:flat.append(dict(dataset=c['dataset'],latent_dim=c['dim'],seed=c['seed'],branch=b,metric=k,step=t,value=y))
def load_cases():
    cases=[]
    for f in sorted(RUN.glob('*/summary.json')):
        s=json.loads(pin(f).read_text())
        if s.get('status')!='complete':continue
        ds=s.get('dataset','controlled');dim=s.get('latent_dim',s.get('dimension'));seed=s['seed'];c=dict(dataset=ds,dim=dim,seed=seed,summary=s,branches={})
        if ds=='controlled':
            base=Path(s['source_dir']);tr=rows(base/'native/train_history.csv');pr=[r for r in rows(base/'native/probe_history.csv') if r['task']=='theta'];ext=rows(base/'online_theta_extensions.csv');pr=sorted(pr+ext,key=lambda r:int(r['step']))
            assert len(pr)==18 and int(tr[-1]['step'])==3000
            if seed==0:
                original=rows(HIST/'results/reconstructed_B_001'/f'B_z{dim}_loss_plus_xi'/'probe_history.csv');original=[r for r in original if r['task']=='theta']
                native=[r for r in pr if int(r['step']) not in [0,1,3000]]
                assert [(int(r['step']),float(r['accuracy'])) for r in native]==[(int(r['step']),float(r['accuracy'])) for r in original]
            sig={age:records(base/f'ref{age}/signals.jsonl') for age in [0,1000]};assert sig[0][0]['step']==1 and sig[1000][0]['step']==1001
            series=dict(mse=xy(tr,'reconstruction'),rec=xy(tr,'reconstruction'),kl=xy(tr,'kl'),old=xy(tr,'xi_pw',lambda r:truth(r['baseline_ready']) and truth(r['window_ready'])),ref0=xy(sig[0],'psi'),ref1000=xy(sig[1000],'psi'),probe_mu=xy(pr,'accuracy'),probe_z=[])
            c['branches']['changing']=dict(series=series,probes=pr,trajectory=tr,signals=sig)
        else:
            assert s['prefix_exact_checks']==1000 and s['stage_steps']==[1000]*3 and s['actual_optimizer_updates']==5000
            for branch in ['changing','stable']:
                folder=f.parent/branch;tr=rows(folder/'trajectory.csv');pr=rows(folder/'probe_history.csv');legacy=rows(folder/'legacy_every_batch.csv');sig={age:records(folder/f'psi_ref{age}.jsonl') for age in [0,1000]};diagnostics=rows(folder/'diagnostics.csv')
                assert len(tr)==(3000 if branch=='changing' else 2000) and int(tr[-1]['step'])==3000
                if branch=='changing':assert sig[0][0]['step']==1 and sig[1000][0]['step']==1001
                for r in pr:
                    assert int(r['probe_stage'])<=(1 if branch=='stable' else 1+(int(r['step'])-1)//1000)
                series=dict(mse=xy(tr,'pixel_mse'),rec=xy(tr,'reconstruction'),kl=xy(tr,'kl'),old=xy(legacy,'xi',lambda r:truth(r['eligible'])),ref0=xy(sig[0],'psi'),ref1000=xy(sig[1000],'psi'),clip=xy(diagnostics,'logvar_clipped_fraction'))
                for readout,key in [('mu','probe_mu'),('sampled_z','probe_z')]:
                    groups={}
                    for r in pr:
                        if r['probe_group']=='S1' and r['readout']==readout and truth(r['primary_task']):groups.setdefault(int(r['step']),[]).append(float(r['balanced_accuracy']))
                    series[key]=sorted((t,float(np.mean(v))) for t,v in groups.items())
                c['branches'][branch]=dict(series=series,probes=pr,trajectory=tr,signals=sig,validation=rows(folder/'validation.csv'),diagnostics=diagnostics)
        for b,v in c['branches'].items():
            for key,points in v['series'].items():metric(c,b,key,points)
        cases.append(c)
    return cases
def meanline(ax,values,color,label=None,style='-',shade=True,marker=None):
    if not any(values):return None
    d={}
    for points in values:
        for t,y in points:d.setdefault(t,[]).append(y)
    t=sorted(d);mean=np.asarray([np.mean(d[x]) for x in t]);low=[min(d[x]) for x in t];high=[max(d[x]) for x in t]
    line=ax.plot(t,mean,color=color,label=label,ls=style,lw=1.45,marker=marker,ms=2.3,markevery=max(1,len(t)//18))[0]
    if shade and len(values)>1:ax.fill_between(t,low,high,color=color,alpha=.12,lw=0)
    return line
def series(c,b,key):
    if b=='stable':return [(t,y) for t,y in c['branches']['changing']['series'].get(key,[]) if t<=1000]+c['branches'][b]['series'].get(key,[])
    return c['branches'][b]['series'].get(key,[])
def decorate(ax,prob=False):
    ax.set_xlim(0,3000);ax.set_xticks([0,1000,2000,3000]);ax.grid(alpha=.17)
    for t in [1000,2000]:ax.axvline(t,color='#555555',ls=':',lw=.85)
    if prob:ax.set_ylim(0,1.02)
def export(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=220,facecolor='white');fig.savefig(FIG/(name+'.svg'));plt.close(fig)
def note(fig,text,top=.91,bottom=.13):
    fig.text(.5,.025,text,ha='center',va='bottom',fontsize=10,color='#555555');fig.subplots_adjust(left=.065,right=.94,top=top,bottom=bottom,hspace=.32,wspace=.52)
def signal(ax,cs):
    right=ax.twinx();right.spines['right'].set_visible(True)
    hs=[]
    for key,color,label in [('ref0','#d97706','新 Ψ：ref0（左）'),('ref1000','#2563eb','新 Ψ：ref1000（左）')]:
        line=meanline(ax,[series(c,'changing',key) for c in cs],color,label,marker='o')
        if line is not None:hs.append(line)
    line=meanline(right,[series(c,'changing','old') for c in cs],'#dc2626','旧 Ξ：MP超额（右）',shade=False)
    if line is not None:hs.append(line)
    ax.set_ylabel('新 Ψ',color='#2563eb');right.set_ylabel('旧 Ξ',color='#dc2626');ax.set_ylim(bottom=0);right.set_ylim(bottom=0);decorate(ax)
    return hs
def dimension_panel(ds,cs,mean=False,individual=None):
    dims=DIMS[ds];nr=3 if ds=='controlled' else 4;fig,axes=plt.subplots(nr,len(dims),figsize=(5.1*len(dims),3.15*nr+1.6),squeeze=False);hs=[]
    for j,dim in enumerate(dims):
        selected=[c for c in cs if c['dim']==dim]
        for ax in axes[:,j]:decorate(ax)
        axes[0,j].set_title(f'z={dim} · n={len(selected)}/'+('3 seeds' if mean else '1 seed'),fontsize=14,pad=9)
        if not selected:
            for ax in axes[:,j]:ax.text(.5,.5,'等待此条件完成',ha='center',transform=ax.transAxes,color='#777777')
            continue
        meanline(axes[0,j],[series(c,'changing','mse') for c in selected],'#15803d');axes[0,j].set_ylabel('原始重构 MSE（对数）' if ds=='controlled' else '像素 MSE（对数）');axes[0,j].set_yscale('log');hs=signal(axes[1,j],selected)
        if ds=='controlled':
            meanline(axes[2,j],[series(c,'changing','probe_mu') for c in selected],'#15803d','原当前阶段 θ 准确率',marker='o');axes[2,j].set_ylabel('原 θ 普通准确率');decorate(axes[2,j],True)
            axes[2,j].axhline(1/8,color='#999999',ls='--',lw=.7)
        else:
            tasks=sorted({r['task'] for c in selected for r in c['branches']['changing']['probes'] if r['probe_group']=='S1' and truth(r['primary_task'])})
            for row,readout in [(2,'mu'),(3,'sampled_z')]:
                for i,task in enumerate(tasks):
                    points=[xy(c['branches']['changing']['probes'],'balanced_accuracy',lambda r:r['probe_group']=='S1' and r['task']==task and r['readout']==readout) for c in selected]
                    meanline(axes[row,j],points,plt.get_cmap('tab10')(i),TASKS.get(task,task))
                axes[row,j].set_ylabel(f'固定 S1 · {"μ" if row==2 else "z"} 探针 BA');decorate(axes[row,j],True);axes[row,j].legend(loc='lower left',fontsize=8,ncol=2)
        axes[-1,j].set_xlabel('训练到达步 t（阶段边界 1000 / 2000）')
    title=(f'{NAMES[ds]}：全部维数 · 三 seed 均值与范围' if mean else f'{NAMES[ds]}：全部维数 · seed{individual} 原轨迹')
    fig.suptitle(title,fontsize=20,y=.985)
    if hs:fig.legend(hs,[h.get_label() for h in hs],loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False,fontsize=11)
    foot='每列同一模型轨迹，新旧信号读各自坐标轴，不能直接按峰高排名。ref0从第1批测量；ref1000从第1001批起，早期不倒填。\n'
    foot+=('controlled保留历史loss_plus_xi、circle/torus/local及当前阶段θ回测；原15测点保持，0/1/3000为离线补测。不是独立旧任务测试。' if ds=='controlled' else '图像rec+0.0001KL，信号不进入更新；固定S1探针重新拟合线性头，μ和固定潜噪声z分别读出。dSprites训练项为BCE，此处另画像素MSE。')
    foot+='\n'+('阴影=已完成seed的最小到最大范围，不是置信区间；每列n不足3表示尚未完成。' if mean else '单seed图保留原曲线；三seed均值另图，不以均值替换原曲线。')
    note(fig,foot,top=.88,bottom=.14);export(fig,f'{"汇总" if mean else "单seed"}_{ds}_'+('三seed' if mean else str(individual))+'_多维度')
def all_probes(ds,cs):
    if ds=='controlled':return
    dims=DIMS[ds];tasks=sorted({r['task'] for c in cs for r in c['branches']['changing']['probes']});fig,axes=plt.subplots(max(1,len(tasks)),len(dims),figsize=(5.1*len(dims),2.7*max(1,len(tasks))+1.9),squeeze=False)
    for i,task in enumerate(tasks):
        for j,dim in enumerate(dims):
            selected=[c for c in cs if c['dim']==dim];ax=axes[i,j]
            for g,col in [(1,'#2563eb'),(2,'#d97706'),(3,'#15803d')]:
                for readout,style in [('mu','-'),('sampled_z','--')]:
                    v=[probe_points(c['branches']['changing']['probes'],task,g,readout) for c in selected]
                    meanline(ax,v,col,f'S{g} '+('μ' if readout=='mu' else 'z'),style,shade=readout=='mu')
            classes=[int(r['classes']) for c in selected for r in c['branches']['changing']['probes'] if r['task']==task]
            if classes:ax.axhline(1/classes[0],color='#999999',ls=':',lw=.7)
            decorate(ax,True);ax.set_ylabel(TASKS.get(task,task)+' · BA')
            if i==0:ax.set_title(f'z={dim} · {len(selected)}/3 seeds',fontsize=13);ax.legend(ncol=3,fontsize=8,loc='lower left')
            if i==len(tasks)-1:ax.set_xlabel('训练到达步 t')
    fig.suptitle(f'{NAMES[ds]}：全部探针任务与全部维数',fontsize=20,y=.99)
    note(fig,'三seed均值，μ阴影为seed最小—最大范围。每组固定源图与固定划分；新阶段任务只在阶段到达后评估，未测区不补零。\n实线=μ，虚线=固定潜噪声z；灰点线=各任务机会水平。方向原始辅助与条件主任务分别保留。',top=.94,bottom=.085);export(fig,f'全部探针_{ds}')
def stable_panel(ds,cs):
    if ds=='controlled':return
    dims=DIMS[ds];fig,axes=plt.subplots(4,len(dims),figsize=(5.1*len(dims),14.4),squeeze=False)
    for j,dim in enumerate(dims):
        selected=[c for c in cs if c['dim']==dim]
        for i,(key,label) in enumerate([('mse','像素 MSE（对数）'),('ref1000','新 Ψ ref1000'),('probe_mu','S1各主任务均值 BA · μ'),('probe_z','S1各主任务均值 BA · z')]):
            ax=axes[i,j]
            for b,col in [('changing','#dc2626'),('stable','#2563eb')]:meanline(ax,[series(c,b,key) for c in selected],col,'阶段变化流' if b=='changing' else 'S1稳定对照')
            decorate(ax,i>=2);ax.set_ylabel(label)
            if i==0:ax.set_yscale('log');ax.set_title(f'z={dim} · {len(selected)}/3 seeds');ax.legend(fontsize=9)
            if i==3:ax.set_xlabel('训练到达步 t')
    fig.suptitle(f'{NAMES[ds]}：同一1000步模型分叉，稳定流与阶段变化流',fontsize=20,y=.985);note(fig,'S1前1000步共用同一模型与Adam状态；稳定分支在后2000步继续S1，变化分支依次训练S2/S3。两条横轴均为3000步。\n探针只比较同一个固定S1数据池；均值范围不作统计显著性判断。缺失条件保留，不作结果筛选。',top=.93,bottom=.1);export(fig,f'稳定流对照_{ds}')
def diagnostic_panel(ds,cs):
    dims=DIMS[ds];nrows=2 if ds=='controlled' else 3;fig,axes=plt.subplots(nrows,len(dims),figsize=(5.1*len(dims),3.1*nrows+1.9),squeeze=False)
    for j,dim in enumerate(dims):
        selected=[c for c in cs if c['dim']==dim]
        for c in selected:
            col=plt.get_cmap('tab10')(c['seed']%10);axes[0,j].plot(*zip(*series(c,'changing','kl')),color=col,lw=1,label=f'seed{c["seed"]}')
            k='rec' if ds=='controlled' else 'clip';v=series(c,'changing',k)
            if v:axes[1,j].plot(*zip(*v),color=col,lw=1,label=f'seed{c["seed"]}')
            if ds!='controlled':axes[2,j].plot(*zip(*series(c,'changing','rec')),color=col,lw=1,label=f'seed{c["seed"]}')
        axes[0,j].set_title(f'z={dim}');axes[0,j].set_ylabel('原始 KL（未乘系数）');axes[1,j].set_ylabel('原始重构损失' if ds=='controlled' else '原始 logvar 被截断比例');axes[1,j].set_xlabel('训练到达步 t')
        if selected:axes[0,j].legend(fontsize=8)
        for ax in axes[:,j]:decorate(ax)
        if ds!='controlled':axes[1,j].set_ylim(-.02,1.02)
        if ds!='controlled':axes[2,j].set_ylabel('实际训练重构项（BCE）' if ds=='dsprites' else '实际训练重构项（MSE）');axes[2,j].set_xlabel('训练到达步 t');axes[2,j].set_yscale('log')
    fig.suptitle(f'{NAMES[ds]}：每个seed的KL与数值诊断',fontsize=20,y=.985);note(fig,'全部已完成seed保留；不因后验饱和或曲线不符合预期删除条件。controlled KL系数0.001，图像为0.0001。\n原模型logvar截断[-12,12]保留；图像诊断比例是在实际信号测点记录，非所有batch连续审计。',top=.89,bottom=.17);export(fig,f'数值诊断_{ds}')
def overview(cases):
    fig,axes=plt.subplots(4,2,figsize=(15,16.8),squeeze=False)
    for i,ds in enumerate(DIMS):
        for j,dim in enumerate(DIMS[ds]):
            selected=[c for c in cases if c['dataset']==ds and c['dim']==dim];col=plt.get_cmap('tab10')(j)
            for k,key in enumerate(['mse','probe_mu']):meanline(axes[i,k],[series(c,'changing',key) for c in selected],col,f'z={dim} · n={len(selected)}/3')
        axes[i,0].set_yscale('log');axes[i,0].set_ylabel(NAMES[ds]+'\n像素/观测 MSE');axes[i,1].set_ylabel('当前阶段 θ 准确率' if ds=='controlled' else '固定S1各原任务均值 BA · μ')
        for j in range(2):
            decorate(axes[i,j],j==1)
            if axes[i,j].get_legend_handles_labels()[0]:axes[i,j].legend(ncol=3,fontsize=8)
    axes[0,0].set_title('重构误差 · 全部潜维数',fontsize=15);axes[0,1].set_title('原探针口径 · 全部潜维数',fontsize=15)
    for ax in axes[-1]:ax.set_xlabel('训练到达步 t')
    fig.suptitle('四个数据集统一3000步：全部维数、全部seed',fontsize=22,y=.99);note(fig,'实线=已完成seed均值，阴影=seed最小—最大范围；n不足3是尚未完成，绝不按结果筛seed。阶段边界为1000/2000。\ncontrolled为历史当前阶段θ普通准确率；图像为固定独立S1探针平衡准确率，各任务详细图另存。跨数据集不作MSE或准确率公平排名。',top=.94,bottom=.085);export(fig,'00_四数据集_全部维度总览')
def main():
    global flat,inputs
    flat=[];inputs={};verify();cases=load_cases();FIG.mkdir(exist_ok=True);overview(cases)
    for ds in DIMS:
        cs=[c for c in cases if c['dataset']==ds];seed=0 if ds=='controlled' else 10
        dimension_panel(ds,[c for c in cs if c['seed']==seed],individual=seed);dimension_panel(ds,cs,mean=True);diagnostic_panel(ds,cs)
        if cs:all_probes(ds,cs);stable_panel(ds,cs)
    if flat:
        with (REPORT/'plot_values.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write(REPORT/'plot_inputs.json',inputs);write(REPORT/'figure_status.json',dict(completed_parents=len(cases),planned_parents=42,status='complete' if len(cases)==42 else 'partial',pngs=len(list(FIG.glob('*.png'))),plot_code_sha256=sha(Path(__file__))))
    lines=['# 3000步多维度结果','',f'已完成 {len(cases)}/42 个父条件；图中的空列或n不足3表示尚未完成。','',
      '阶段统一为1000/1000/1000。新旧信号在同一轨迹测量；ref0从第1批开始，ref1000从1001批开始。旧信号有预热，无效区不当成有效零。',
      'controlled保留历史circle/torus/local、rec+0.001KL+0.001旧Ξ与原当前阶段θ探针。图像保持固定独立探针角色、rec+0.0001KL，新旧信号只读。三seed均值和原seed0/10图分开。跨协议不作算法公平排名。','',
      '| 数据集 | 潜维数 | 完成父条件 |','|---|---|---:|']
    for ds in DIMS:lines.append(f'|{NAMES[ds]}|'+', '.join(map(str,DIMS[ds]))+f'|{sum(c["dataset"]==ds for c in cases)}/{len(DIMS[ds])*3}|')
    lines+=['','## 图像入口','','- [四数据集全部维度总览](../figures/00_四数据集_全部维度总览.png)','- `单seed_*`：保留原首个seed曲线，所有维数放同一张图。','- `汇总_*`：三seed均值和范围，缺失seed明确标n。','- `全部探针_*`：所有已定义因子、S1/S2/S3固定池、µ/z，按到达阶段显示。','- `稳定流对照_*`：同一1000步模型分叉后比较稳定与变化训练。','- `数值诊断_*`：每个seed的KL与图像logvar截断比例。','','本轮没有沿用4000步协议的阈值和事件窗口计算AUROC。所有测点可在plot_values.csv查证，输入SHA在plot_inputs.json。范围阴影不是置信区间。','']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines))
    for p,h in inputs.items():
        if sha(p)!=h:raise ValueError('Completed input mutated while plotting')
    if len(cases)==42:
        dest=ROOT/'3000步多维度_图表与代码.zip'
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for folder in ['src','tools','configs','docs','reports','figures']:
                for f in (ROOT/folder).rglob('*'):
                    if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.partial','.pyc']:z.write(f,f.relative_to(ROOT))
            for f in RUN.rglob('*'):
                if f.is_file() and f.suffix in ['.csv','.json','.jsonl']:z.write(f,f.relative_to(ROOT))
        with zipfile.ZipFile(dest) as z:
            if z.testzip():raise ValueError('Final zip CRC failed')
    print(json.dumps(dict(rendered=len(cases),pngs=len(list(FIG.glob('*.png')))),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
