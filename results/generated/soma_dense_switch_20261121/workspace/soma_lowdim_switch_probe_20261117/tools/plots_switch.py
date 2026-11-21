"""Honest transition-focused views; fixed-S1, current support and frozen heads stay distinct."""
from pathlib import Path
import sys,json,csv,importlib.util,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from study_paths import PLAN,OLD,rows,write,sha
sys.path.insert(0,str(OLD/'tools'));import render as old
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
DIMS={'controlled':[2,4,8,16,32],**PLAN['all_image_dimensions']};NAMES=old.NAMES
COL={'fixed':'#2563eb','current':'#d97706','stable':'#64748b','frozen':'#7c3aed','old':'#dc2626','ref0':'#d97706','ref1000':'#7c3aed','mse':'#0f766e'}
FROZEN={};VALUES={};INPUTS={}
def pin(p):p=Path(p);INPUTS[str(p)]=sha(p);return p
def read(p):return rows(pin(p)) if Path(p).exists() else []
def jl(p):return [json.loads(v) for v in pin(p).read_text().splitlines() if v] if Path(p).exists() else []
def points(rr,key,predicate=lambda r:True):return [(int(r['step']),float(r[key])) for r in rr if predicate(r)]
def load():
    old.inputs={};old.flat=[];cases=old.load_cases();INPUTS.update(old.inputs)
    for f in (ROOT/'results'/PLAN['run_id']).glob('*/summary.json'):
        s=json.loads(pin(f).read_text())
        if s['status']!='complete':continue
        assert s['actual_optimizer_updates']==5000 and s['source_IDs_match_existing'];c=dict(dataset=s['dataset'],dim=s['latent_dim'],seed=s['seed'],summary=s,branches={})
        for b,n in [('changing',3000),('stable',2000)]:
            d=f.parent/b;tr=read(d/'trajectory.csv');pr=read(d/'probe_history.csv');leg=read(d/'legacy_every_batch.csv');sig={a:jl(d/f'psi_ref{a}.jsonl') for a in [0,1000]};assert len(tr)==n and int(tr[-1]['step'])==3000
            series=dict(mse=points(tr,'pixel_mse'),rec=points(tr,'reconstruction'),kl=points(tr,'kl'),old=points(leg,'xi',lambda r:old.truth(r['eligible'])),ref0=points(sig[0],'psi'),ref1000=points(sig[1000],'psi'));c['branches'][b]=dict(series=series,probes=pr,trajectory=tr,signals=sig)
        cases.append(c)
    for c in cases:
        if c['dataset']=='controlled':continue
        name=f'{c["dataset"]}_z{c["dim"]}_seed{c["seed"]}';file=ROOT/'results/probe_replay'/name/'summary.json'
        if file.exists() and json.loads(pin(file).read_text())['status']=='complete':FROZEN[name]=read(file.parent/'frozen_probe.csv')
    return cases
def get(c,metric,b='changing',readout='mu',kind='fixed',task=None):
    if metric!='probe':return old.series(c,b,metric)
    if c['dataset']=='controlled':return old.series(c,'changing','probe_mu')
    rr=c['branches']['changing']['probes'];tasks={r['task'] for r in rr if r['probe_group']=='S1'}
    if task:tasks={task}
    if kind=='frozen':rr=FROZEN.get(f'{c["dataset"]}_z{c["dim"]}_seed{c["seed"]}',[]);rr=[r for r in rr if r['branch']==b]
    else:
        rr=c['branches'][b]['probes']
        if b=='stable':rr=[r for r in c['branches']['changing']['probes'] if int(r['step'])<=1000]+rr
    group={}
    for r in rr:
        t=int(r['step']);desired='S'+str(1+(t-1)//1000)
        keep=(kind=='frozen' or r['probe_group']==('S1' if kind=='fixed' else desired))
        if keep and r['task'] in tasks and r['readout']==readout:group.setdefault(t,[]).append(float(r['balanced_accuracy']))
    return sorted((t,float(np.mean(v))) for t,v in group.items() if len(v)==len(tasks))
def line(ax,cases,metric,color,label,b='changing',readout='mu',kind='fixed',style='-',task=None):
    lo,hi=ax.get_xlim();values=[[(t,y) for t,y in get(c,metric,b,readout,kind,task) if lo<=t<=hi] for c in cases];return old.meanline(ax,values,color,label,style,shade=True)
def decorate(ax,prob=False,window=None):
    ax.set_xlim(*(window or [PLAN['figure_start'],3000]));ax.grid(alpha=.15)
    for t in [1000,2000]:ax.axvline(t,color='#666666',ls=':',lw=.8)
    if prob:ax.set_ylim(0,1.02)
    if window:ax.set_xticks(np.arange(window[0],window[1]+1,100))
    else:ax.set_xticks([700,1000,1500,2000,2500,3000])
def export(fig,name,dpi=190):
    for ext in ['png','svg']:fig.savefig(FIG/(name+'.'+ext),dpi=dpi,facecolor='white')
    plt.close(fig)
def signal(ax,cs,window=None):
    right=ax.twinx();right.spines['right'].set_visible(True)
    for k in ['ref0','ref1000']:line(ax,cs,k,COL[k],k)
    line(right,cs,'old',COL['old'],'old');decorate(ax,window=window);ax.set_ylim(bottom=0);right.set_ylim(bottom=0);ax.set_ylabel('新 Ψ（左）',color=COL['frozen']);right.set_ylabel('旧 Ξ（右）',color=COL['old']);right.tick_params(axis='y',labelsize=8,colors=COL['old'])
def chance(cs):
    c=cs[0]
    if c['dataset']=='controlled':return 1/8
    r=c['branches']['changing']['probes'];classes={x['task']:int(x['classes']) for x in r if x['probe_group']=='S1'};return np.mean([1/k for k in classes.values()])
def overview(ds,cases):
    dims=DIMS[ds];nr=3 if ds=='controlled' else 5;fig,axes=plt.subplots(nr,len(dims),figsize=(5.7*len(dims),2.7*nr+2.7),squeeze=False)
    for j,dim in enumerate(dims):
        cs=[c for c in cases if c['dim']==dim];axes[0,j].set_title(f'z={dim} · 完成{len(cs)}/3 seed',fontsize=13)
        for ax in axes[:,j]:decorate(ax)
        if not cs:
            for ax in axes[:,j]:ax.text(.5,.5,'2/4维补充实验运行中',ha='center',transform=ax.transAxes,color='#777777')
            continue
        signal(axes[0,j],cs);line(axes[1,j],cs,'mse',COL['mse'],'变化流当前批MSE');axes[1,j].set_ylabel('当前流像素/观测 MSE')
        if ds=='controlled':line(axes[2,j],cs,'probe',COL['fixed'],'原当前阶段θ');decorate(axes[2,j],True);axes[2,j].set_ylabel('原θ普通准确率')
        else:
            line(axes[1,j],cs,'mse',COL['stable'],'S1稳定流',b='stable')
            for row,readout in [(2,'mu'),(3,'sampled_z')]:
                for kind,color,label in [('fixed',COL['fixed'],'固定S1'),('current',COL['current'],'当前阶段')]:line(axes[row,j],cs,'probe',color,label,readout=readout,kind=kind)
                line(axes[row,j],cs,'probe',COL['stable'],'S1稳定对照',b='stable',readout=readout);axes[row,j].axhline(chance(cs),color='#aaaaaa',ls='--',lw=.7);decorate(axes[row,j],True);axes[row,j].set_ylabel('原任务均值 BA · '+('μ' if row==2 else 'z'));axes[row,j].legend(loc='lower left',fontsize=8)
            ready=sum(f'{c["dataset"]}_z{dim}_seed{c["seed"]}' in FROZEN for c in cs)
            for b,color in [('changing',COL['frozen']),('stable',COL['stable'])]:
                for rd,st in [('mu','-'),('sampled_z','--')]:line(axes[4,j],cs,'probe',color,('变化流' if b=='changing' else '稳定流')+' '+rd,b=b,readout=rd,kind='frozen',style=st)
            axes[4,j].set_ylabel(f'900步冻结S1头 BA · {ready}/{len(cs)}完成');decorate(axes[4,j],True)
            if ready:axes[4,j].legend(loc='lower left',fontsize=7,ncol=2)
        axes[-1,j].set_xlabel('训练到达步 t')
    fig.suptitle(NAMES[ds]+' · 结构切换聚焦 · 所有维数',fontsize=22,y=.99)
    fig.text(.5,.948,'信号：橙=新Ψ ref0，紫=新Ψ ref1000，红=旧Ξ（右轴）；探针：固定S1与当前阶段是不同评估支持。',ha='center',fontsize=11)
    foot='显示从700步起，原3000步数据全部保留。曲线=已完成seed均值，阴影=最小—最大范围；未完成不补零。\n'
    foot+=('controlled保留历史当前阶段θ训练池80/20回测，普通准确率；不将其当固定独立S1遗忘指标。' if ds=='controlled' else '原探针每次重新拟合；当前阶段只聚合同一组原S1任务，跨段换测试支持，下降不能直接当旧任务遗忘。\n冻结头只从900步起，固定均值/标准差/权重；下降可含坐标漂移，不等于信息不可恢复。稳定流从同一1000步模型分叉。')
    fig.text(.5,.018,foot,ha='center',fontsize=10,color='#555555');fig.subplots_adjust(top=.90,bottom=.13,hspace=.29,wspace=.45,left=.055,right=.945);export(fig,f'概览_{ds}_全部维数_信号_MSE_三种探针')
def effects(c,kind,cut,task=None):
    a=dict(get(c,'probe','changing',kind=kind,task=task));b=dict(get(c,'probe','stable',kind=kind,task=task));pre=[900,950,1000] if cut==1000 else [1900,1950,2000]
    if not all(t in a and t in b for t in pre):return []
    baseline=np.mean([a[t]-b[t] for t in pre]);return [(t,100*((a[t]-b[t])-baseline)) for t in sorted(set(a)&set(b))]
def window_figure(ds,cases):
    if ds=='controlled':return
    dims=DIMS[ds];fig,axes=plt.subplots(6,len(dims),figsize=(5.7*len(dims),19.5),squeeze=False)
    for j,d in enumerate(dims):
        cs=[c for c in cases if c['dim']==d];axes[0,j].set_title(f'z={d} · n={len(cs)}/3',fontsize=13)
        for k,(cut,window) in enumerate(zip([1000,2000],PLAN['display_windows'])):
            off=k*3
            for ax in axes[off:off+3,j]:decorate(ax,window=window)
            if not cs:continue
            signal(axes[off,j],cs,window);line(axes[off+1,j],cs,'mse',COL['mse'],'变化流');line(axes[off+1,j],cs,'mse',COL['stable'],'S1稳定流',b='stable');axes[off+1,j].set_ylabel('当前流 MSE')
            for kind,color,label in [('fixed',COL['current'],'原重拟合头'),('frozen',COL['frozen'],'900步冻结头')]:old.meanline(axes[off+2,j],[[(t,y) for t,y in effects(c,kind,cut) if window[0]<=t<=window[1]] for c in cs],color,label)
            axes[off+2,j].axhline(0,color='#777777',lw=.8);axes[off+2,j].set_ylabel('固定S1 BA 配对变化\n（百分点）');axes[off+2,j].legend(fontsize=8,loc='best')
    fig.suptitle(NAMES[ds]+' · 两次切换局部放大与稳定流校正',fontsize=21,y=.99);fig.text(.5,.018,'上三排=1000步切换，下三排=2000步切换；新Ψ参考1000在1001步前不可用。\n探针效应=（变化流BA−稳定流BA）再减切换前共同窗口均差，单位为百分点；不是原始BA，绝对0–1曲线见概览。\n预窗口900–1000/1900–2000，后窗口1001–1100/2001–2100，均为3个真实测点；不挑最好窗口、不做显著性声明。',ha='center',fontsize=10,color='#555555');fig.subplots_adjust(top=.94,bottom=.10,hspace=.32,wspace=.45,left=.055,right=.945);export(fig,f'切换窗口_{ds}_全部维数')
def all_tasks(ds,cases):
    if ds=='controlled':return
    dims=DIMS[ds];tasks=sorted({r['task'] for c in cases for r in c['branches']['changing']['probes']});nr=len(tasks)+2;fig,axes=plt.subplots(nr,len(dims),figsize=(5.7*len(dims),2.5*nr+2.3),squeeze=False)
    for j,d in enumerate(dims):
        cs=[c for c in cases if c['dim']==d];axes[0,j].set_title(f'z={d} · n={len(cs)}/3')
        for ax in axes[:,j]:decorate(ax)
        if not cs:
            for ax in axes[:,j]:ax.text(.5,.5,'运行中',ha='center',transform=ax.transAxes,color='#888888')
            continue
        signal(axes[0,j],cs);line(axes[1,j],cs,'mse',COL['mse'],'MSE');axes[1,j].set_ylabel('当前批像素MSE')
        for i,task in enumerate(tasks,2):
            ax=axes[i,j];classes=[]
            for stage,col in [(1,COL['fixed']),(2,COL['current']),(3,'#15803d')]:
                for rd,style in [('mu','-'),('sampled_z','--')]:old.meanline(ax,[old.probe_points(c['branches']['changing']['probes'],task,stage,rd) for c in cs],col,f'S{stage} '+rd,style)
            classes=[int(r['classes']) for c in cs for r in c['branches']['changing']['probes'] if r['task']==task]
            if classes:ax.axhline(1/classes[0],color='#999999',ls=':',lw=.7)
            decorate(ax,True);ax.set_ylabel(old.TASKS.get(task,task)+'\nBA')
        axes[-1,j].set_xlabel('训练到达步 t')
    fig.suptitle(NAMES[ds]+' · 信号 / MSE / 全部原探针 · 各维数',fontsize=21,y=.99);fig.text(.5,.95,'探针蓝=S1、橙=S2、绿=S3固定池；实线μ、虚线z。不同组分开画，不用新任务难度代替旧任务遗忘。',ha='center',fontsize=11);fig.text(.5,.018,'显示从700步开始，绝对BA保持0–1；新阶段任务在到达前缺测，不补零。所有seed、所有任务和数值异常保留。',ha='center',fontsize=10,color='#555555');fig.subplots_adjust(top=.92,bottom=.075,hspace=.30,wspace=.44,left=.055,right=.945);export(fig,f'全任务_{ds}_信号_MSE_各维数')
def transition_table(cases):
    result=[]
    for c in cases:
        if c['dataset']=='controlled':continue
        tasks=sorted({r['task'] for r in c['branches']['changing']['probes'] if r['probe_group']=='S1'})
        for task in tasks:
            for kind in ['fixed','frozen']:
                for cut,pre,post in [(1000,[900,950,1000],[1001,1050,1100]),(2000,[1900,1950,2000],[2001,2050,2100])]:
                    a=dict(get(c,'probe',kind=kind,task=task));b=dict(get(c,'probe',b='stable',kind=kind,task=task))
                    if not all(t in a and t in b for t in pre+post):continue
                    da=np.mean([a[t] for t in post])-np.mean([a[t] for t in pre]);db=np.mean([b[t] for t in post])-np.mean([b[t] for t in pre]);result.append(dict(dataset=c['dataset'],dim=c['dim'],seed=c['seed'],task=task,readout='mu',head_kind=kind,cut=cut,changing_delta_pp=100*da,stable_delta_pp=100*db,paired_increment_pp=100*(da-db),pre_points=3,post_points=3))
    if result:
        with (ROOT/'reports/transition_effects.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
def metric_figure():
    fig,axes=plt.subplots(1,3,figsize=(16,4.8));axes[0].bar(['普通准确率','平衡准确率'],[.9,.5],color=['#2563eb','#d97706']);axes[0].set_ylim(0,1);axes[0].set_title('90张A、10张B，全猜A');axes[0].text(.0,.93,'90%',ha='center');axes[0].text(1,.53,'50%',ha='center')
    rr=json.loads((ROOT/'reports/existing_probe_summary.json').read_text());selected=[r for r in rr if (r['dataset']=='cifar' and r['dim']==32 and r['group']=='S1' and r['readout']=='mu') or (r['dataset']=='dsprites' and r['dim']==32 and r['task']=='orientation_shape2_bin8' and r['readout']=='mu')]
    axes[1].bar(np.arange(len(selected)),[r['mean'] for r in selected],color='#0f766e');axes[1].scatter(np.arange(len(selected)),[r['chance'] for r in selected],color='#dc2626',marker='_',s=200,label='机会水平');axes[1].set_xticks(range(len(selected)),['CIFAR 10类','dSprites方向8类']);axes[1].set_ylim(0,1);axes[1].set_title('旧32维、3seed：低分要与机会水平比较');axes[1].legend()
    rr=[r for r in rr if r['dataset']=='shapes3d' and r['dim']==64 and r['group']=='S1' and r['task']=='floor_hue'];rr=sorted(rr,key=lambda r:r['readout']);axes[2].bar(['μ均值读出','z采样读出'],[r['mean'] for r in rr],color=['#7c3aed','#d97706']);axes[2].set_ylim(0,1);axes[2].set_title('Shapes3D64维：同一地面颜色任务');fig.text(.5,.02,'BA=各类别召回率的等权平均；潜维数2/4不代表类别数。右图提示采样噪声/有限样本拟合影响，不能说模型完全没有颜色信息。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.08,1,1]);export(fig,'解释_平衡准确率_低分与采样噪声')
def master(cases):
    fig,axes=plt.subplots(12,5,figsize=(30,35),squeeze=False)
    for k,(ds,dims) in enumerate(DIMS.items()):
        off=3*k
        for j,d in enumerate(dims):
            cs=[c for c in cases if c['dataset']==ds and c['dim']==d];axes[off,j].set_title(f'{NAMES[ds]} · z={d} · n={len(cs)}/3',fontsize=13)
            for ax in axes[off:off+3,j]:decorate(ax)
            if not cs:
                for ax in axes[off:off+3,j]:ax.text(.5,.5,'补充实验运行中',transform=ax.transAxes,ha='center',color='#777777')
                continue
            signal(axes[off,j],cs);line(axes[off+1,j],cs,'mse',COL['mse'],'变化流MSE');axes[off+1,j].set_ylabel('当前批 MSE')
            if ds=='controlled':line(axes[off+2,j],cs,'probe',COL['fixed'],'历史当前阶段θ（普通准确率）')
            else:
                line(axes[off+1,j],cs,'mse',COL['stable'],'S1稳定流',b='stable')
                for kind,color,label in [('fixed',COL['fixed'],'重拟合/固定S1'),('current',COL['current'],'重拟合/当前阶段'),('frozen',COL['frozen'],'冻结头/固定S1')]:line(axes[off+2,j],cs,'probe',color,label,kind=kind)
                line(axes[off+2,j],cs,'probe',COL['stable'],'重拟合/S1稳定',b='stable');axes[off+2,j].axhline(chance(cs),ls=':',color='#999999',lw=.8)
            decorate(axes[off+2,j],True);axes[off+2,j].set_ylabel('探针准确率' if ds=='controlled' else '原S1任务均值 BA · μ');axes[off+2,j].legend(loc='lower left',fontsize=7,ncol=2)
    fig.suptitle('四数据集 × 全部维数 · 新旧信号 / MSE / 下游探针',fontsize=27,y=.995)
    fig.text(.5,.977,'每数据集三排：信号（新Ψ ref0橙/ref1000紫，旧Ξ红右轴）、当前流MSE、探针。细分任务、z采样读出与切换校正另见各数据集清晰图。',ha='center',fontsize=13)
    fig.text(.5,.012,'从700步显示，训练仍为1000+1000+1000；虚线为结构切换。曲线为已完成seed均值，阴影为seed范围，未完成不补零。\n固定S1评估旧任务；当前阶段更换测试支持，下降不等于遗忘；900步冻结头可能反映坐标漂移。controlled仍为历史θ回测口径，不能与图像独立探针直接比较。',ha='center',fontsize=12,color='#555555');fig.subplots_adjust(top=.96,bottom=.045,hspace=.48,wspace=.43,left=.055,right=.945);export(fig,'总览_四数据集_全部维数_新旧信号_MSE_探针',dpi=170)
def main():
    FROZEN.clear();INPUTS.clear();cases=load()
    for ds in DIMS:
        cs=[c for c in cases if c['dataset']==ds];overview(ds,cs);window_figure(ds,cs);all_tasks(ds,cs)
    master(cases);transition_table(cases);metric_figure();write(ROOT/'reports/figure_inputs.json',INPUTS);write(ROOT/'reports/figure_status.json',dict(status='complete' if len(cases)==60 and len(FROZEN)==45 else 'partial',training_conditions=len(cases),planned_conditions=60,new_training_complete=len(cases)-42,new_training_planned=18,frozen_replays=len(FROZEN),planned_frozen_replays=45,figure_start=700,plot_code_sha256=sha(Path(__file__))))
    lines=['# 结构切换聚焦图与2/4维补充','',f'完整父条件{len(cases)}/60，新增低维{len(cases)-42}/18，冻结头重算{len(FROZEN)}/45。','',
       '概览图将固定S1、当前阶段、900步冻结头分开；切换窗口图报告稳定对照校正后的百分点；全任务图保留S2/S3新增任务。所有原始轨迹仍为3000步，只将显示起点设为700。','',
       '当前阶段换了评估支持，分数跳变不等于旧任务遗忘；冻结头下降也可能是可逆坐标漂移。原重拟合S1探针和稳定对照共同用于解释。','']
    for p in sorted(FIG.glob('*.png')):lines.append(f'- [{p.stem}](../figures/{p.name})')
    (ROOT/'reports/GALLERY.md').write_text('\n'.join(lines)+'\n')
    for p,h in INPUTS.items():
        if sha(p)!=h:raise ValueError('Completed source changed during plotting')
    print(json.dumps(dict(status='rendered',parents=len(cases),frozen=len(FROZEN),pngs=len(list(FIG.glob('*.png'))))),flush=True)
    return cases
if __name__=='__main__':main()
