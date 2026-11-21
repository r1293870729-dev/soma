"""Real metrics, locally cropped y axes, every measured step, legends in each panel."""
from pathlib import Path
import csv,json,sys,math
import numpy as np
from dense_common import ROOT,LOW,PLAN,wanted,write,sha
sys.path.insert(0,str(LOW/'tools'))
import plots_switch as old
plt=old.plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
FIG=ROOT/'figures';COL=old.COL
DATASETS=['controlled','dsprites','shapes3d','cifar'];NAMES=old.NAMES
INPUTS={}
def csvread(p):
    p=Path(p);INPUTS[str(p)]=sha(p)
    with p.open() as f:return list(csv.DictReader(f))
def jsonlines(p):
    if not p.exists():return []
    INPUTS[str(p)]=sha(p);return [json.loads(v) for v in p.read_text().splitlines() if v]
def load():
    old.FROZEN.clear();old.INPUTS.clear();INPUTS.clear();cases=old.load();INPUTS.update(old.INPUTS)
    for c in cases:
        folder=ROOT/'results'/f'{c["dataset"]}_z{c["dim"]}_seed{c["seed"]}';summary=folder/'summary.json';c['dense']=None
        if not summary.exists():continue
        INPUTS[str(summary)]=sha(summary);s=json.loads(summary.read_text())
        if s['status']!='complete':continue
        c['dense']={}
        for branch in (['changing'] if c['dataset']=='controlled' else ['changing','stable']):
            base=folder if c['dataset']=='controlled' else folder/branch
            c['dense'][branch]=dict(metrics=csvread(base/'dense_metrics.csv'),probes=csvread(base/'dense_probe.csv'),frozen=csvread(base/'dense_frozen_probe.csv') if (base/'dense_frozen_probe.csv').exists() else [],signals={a:jsonlines(base/f'psi_ref{a}.jsonl') for a in [0,1000]})
            expected=[t for t in wanted() if branch=='changing' or t>1000]
            if sorted({int(r['step']) for r in c['dense'][branch]['metrics']})!=expected:raise ValueError('Incomplete verified dense metrics')
            if sorted({int(r['step']) for r in c['dense'][branch]['probes']})!=expected:raise ValueError('Incomplete verified dense probes')
    return cases
def records(c,branch,key,age=None):
    d=c['dense'];rr=d[branch]['signals'][age] if key=='signals' else d[branch][key]
    if branch=='stable':
        before=d['changing']['signals'][age] if key=='signals' else d['changing'][key];rr=[r for r in before if int(r['step'])<=1000]+rr
    return rr
def points(c,metric,branch='changing',kind='fixed',readout='mu',task=None,group=None):
    if c['dense'] is None:
        if group is None:return old.get(c,metric,branch,readout,kind,task)
        rr=c['branches'][branch]['probes']
        if branch=='stable':rr=[r for r in c['branches']['changing']['probes'] if int(r['step'])<=1000]+rr
        return [(int(r['step']),float(r['balanced_accuracy'])) for r in rr if r['probe_group']==group and r['task']==task and r['readout']==readout]
    if metric in ['ref0','ref1000']:return [(int(r['step']),float(r['psi'])) for r in records(c,branch,'signals',int(metric[3:]))]
    if metric in ['old','mse']:
        return [(int(r['step']),float(r['xi' if metric=='old' else 'pixel_mse'])) for r in records(c,branch,'metrics') if metric!='old' or old.old.truth(r['eligible'])]
    if c['dataset']=='controlled':return [(int(r['step']),float(r['accuracy'])) for r in records(c,'changing','probes') if r['task']=='theta']
    rr=records(c,branch,'frozen' if kind=='frozen' else 'probes');tasks={task} if task else {r['task'] for r in c['branches']['changing']['probes'] if r['probe_group']=='S1'};buckets={}
    for r in rr:
        t=int(r['step']);desired='S1' if branch=='stable' or kind in ['fixed','frozen'] else 'S'+str((t-1)//1000+1)
        if r['readout']!=readout or r['task'] not in tasks:continue
        if group is not None and r['probe_group']!=group:continue
        if group is None and kind!='frozen' and r['probe_group']!=desired:continue
        buckets.setdefault(t,[]).append(float(r['balanced_accuracy']))
    return sorted((t,float(np.mean(v))) for t,v in buckets.items() if len(v)==len(tasks))
def decorate(ax,cut):
    ax.set_xlim(cut-30,cut+30);ax.xaxis.set_major_locator(MultipleLocator(5));ax.xaxis.set_minor_locator(MultipleLocator(1));ax.grid(alpha=.20,which='major');ax.grid(alpha=.05,which='minor',axis='x');ax.axvline(cut+.5,color='#555555',lw=1,ls=':');ax.yaxis.set_major_locator(MaxNLocator(nbins=4));ax.tick_params(labelsize=9);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
def curve(ax,cs,cut,metric,color,label,branch='changing',kind='fixed',rd='mu',style='-',task=None,group=None):
    values=[[(t,y) for t,y in points(c,metric,branch,kind,rd,task,group) if cut-30<=t<=cut+30] for c in cs]
    line=old.old.meanline(ax,values,color,label,style,shade=True,marker='o' if cs and cs[0]['dense'] else 'x')
    if line is not None:line.set_markersize(2.6 if cs[0]['dense'] else 5);line.set_markevery(1)
    return line,[v for vv in values for _,v in vv]
def probability(ax,values,empty='当前窗口无原始探针点\n逐步补测中'):
    if values:
        lo=max(0.,math.floor((min(values)-.012)*10)/10);hi=min(1.,math.ceil((max(values)+.012)*10)/10)
        if hi-lo<.099:lo=max(0.,lo-.05);hi=min(1.,hi+.05)
        ax.set_ylim(lo,hi)
    else:ax.set_ylim(0,1);ax.text(.5,.48,empty,ha='center',va='center',transform=ax.transAxes,color='#777777',fontsize=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
def legend(ax,lines):
    handles=[v for v in lines if v is not None]
    if handles:ax.legend(handles=handles,loc='best',fontsize=7.8,framealpha=.92,ncol=1)
def signal(ax,cs,cut):
    decorate(ax,cut);right=ax.twinx();right.spines['right'].set_visible(True);lines=[]
    for metric,col,label in [('ref0',COL['ref0'],'新Ψ ref0 · 左轴'),('ref1000',COL['ref1000'],'新Ψ ref1000 · 左轴')]:lines.append(curve(ax,cs,cut,metric,col,label)[0])
    lines.append(curve(right,cs,cut,'old',COL['old'],'旧Ξ · 右轴')[0]);ax.set_ylabel('新Ψ（左）',fontsize=10);right.set_ylabel('旧Ξ（右）',fontsize=10,color=COL['old']);right.tick_params(axis='y',labelsize=9,colors=COL['old']);right.yaxis.set_major_locator(MaxNLocator(nbins=4));right.ticklabel_format(axis='y',style='plain',useOffset=False);ax.set_ylim(bottom=0);right.set_ylim(bottom=0);legend(ax,lines)
def mse(ax,cs,cut):
    decorate(ax,cut);lines=[curve(ax,cs,cut,'mse',COL['mse'],'当前变化流MSE')[0]]
    if cs and cs[0]['dataset']!='controlled':lines.append(curve(ax,cs,cut,'mse',COL['stable'],'S1稳定流MSE',branch='stable')[0])
    ax.set_ylabel('原始MSE',fontsize=10);legend(ax,lines)
def probes(ax,cs,cut,frozen=False):
    decorate(ax,cut);lines=[];vv=[]
    if cs[0]['dataset']=='controlled':
        line,v=curve(ax,cs,cut,'probe',COL['fixed'],'当前阶段θ普通准确率');lines.append(line);vv+=v;ax.set_ylabel('θ准确率 · 局部纵轴',fontsize=10)
    else:
        defs=[('changing','fixed',COL['fixed'],'固定S1 · 重拟合μ'),('changing','current',COL['current'],'当前阶段 · 重拟合μ'),('stable','fixed',COL['stable'],'S1稳定流 · 重拟合μ')]
        if frozen:defs=[('changing','frozen',COL['frozen'],'固定S1 · 冻结μ头'),('stable','frozen',COL['stable'],'S1稳定流 · 冻结μ头'),('changing','fixed',COL['fixed'],'固定S1 · 重拟合μ')]
        for branch,kind,col,label in defs:
            line,v=curve(ax,cs,cut,'probe',col,label,branch,kind);lines.append(line);vv+=v
        ax.set_ylabel('真实BA · 局部纵轴',fontsize=10)
    probability(ax,vv);legend(ax,lines)
def selected_cases(cases,ds,dim):
    cs=[c for c in cases if c['dataset']==ds and c['dim']==dim];dense=[c for c in cs if c['dense'] is not None]
    return dense or cs,len(dense)
def footer(fig,ds,y=.012):
    a='窗口内每步画真实点，主刻度5步、副刻度1步；圆点=已验证逐步补测，叉号=原日志已测点，缺测不补点。\n'
    b='各面板保留原始量纲，探针纵轴按本面板实际范围截取，请按刻度判断变化；阴影为所示seed最小—最大范围，不是置信区间。\n'
    c=('controlled为历史当前阶段θ训练池80/20回测，split/拟合seed随step变化，更新后测量；没有本批稳定分叉。' if ds=='controlled' else '图像探针更新前测量；固定S1与当前阶段的测试支持不同。稳定流共用前1000步；冻结头下降可含坐标漂移。')
    fig.text(.5,y,a+b+c,ha='center',fontsize=10,color='#555555')
def export(fig,name):
    for ext in ['png','svg']:fig.savefig(FIG/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)
def panel(ds,cases,cuts):
    dims=PLAN['controlled_dims'] if ds=='controlled' else PLAN['image_dims'][ds];nr=3 if ds=='controlled' else 4;fig,axes=plt.subplots(nr*len(cuts),5,figsize=(28,3*nr*len(cuts)+2.5),squeeze=False)
    for j,dim in enumerate(dims):
        cs,n=selected_cases(cases,ds,dim)
        for k,cut in enumerate(cuts):
            offset=k*nr;signal(axes[offset,j],cs,cut);axes[offset,j].set_title(f'z={dim} · 切换{cut} · 逐步验证{n}/3',fontsize=12,pad=10);mse(axes[offset+1,j],cs,cut);probes(axes[offset+2,j],cs,cut)
            if nr==4:probes(axes[offset+3,j],cs,cut,True)
            axes[offset+nr-1,j].set_xlabel('原实验步 t（切换后首批 '+str(cut+1)+'）',fontsize=10)
    fig.suptitle(NAMES[ds]+' · 切换前后各30步 · 新旧信号 / MSE / 真实探针指标',fontsize=23,y=.993);fig.text(.5,.961,'每条曲线在本面板标注图例。ref1000从1001步起；列中完成数不足3表示逐步补测未全部完成，0/3只显示原始日志。',ha='center',fontsize=11);footer(fig,ds);fig.subplots_adjust(top=.91 if len(cuts)==1 else .94,bottom=.13 if len(cuts)==1 else .075,left=.055,right=.947,wspace=.50,hspace=.43)
    export(fig,ds+'_逐步切换_'+('全部维数' if len(cuts)==2 else str(cuts[0])))
def detail(ds,dim,cases):
    cs,n=selected_cases(cases,ds,dim);tasks=[] if ds=='controlled' else sorted({r['task'] for c in cs for r in c['branches']['changing']['probes']});nr=3 if ds=='controlled' else len(tasks)+2;fig,axes=plt.subplots(nr,2,figsize=(14,2.7*nr+2.2),squeeze=False)
    for j,cut in enumerate(PLAN['cuts']):
        signal(axes[0,j],cs,cut);axes[0,j].set_title(f'切换{cut} · {cut-30}–{cut+30}步',fontsize=13);mse(axes[1,j],cs,cut)
        if ds=='controlled':probes(axes[2,j],cs,cut)
        else:
            for i,task in enumerate(tasks,2):
                ax=axes[i,j];decorate(ax,cut);lines=[];vv=[];groups=sorted({r['probe_group'] for c in cs for r in c['branches']['changing']['probes'] if r['task']==task})
                for group in groups:
                    col=COL['fixed'] if group=='S1' else COL['current'] if group=='S2' else '#15803d'
                    for rd,st in [('mu','-'),('sampled_z','--')]:
                        line,v=curve(ax,cs,cut,'probe',col,group+' · '+('μ' if rd=='mu' else 'z'),rd=rd,style=st,task=task,group=group);lines.append(line);vv+=v
                future=all(g.startswith('S3') for g in groups) and cut==1000
                probability(ax,vv,'该任务尚未到达\n不定义，不补零' if future else '当前窗口无探针实测点\n逐步补测中');ax.set_ylabel(old.old.TASKS.get(task,task)+'\n真实BA · 局部纵轴',fontsize=10);legend(ax,lines)
        axes[-1,j].set_xlabel('原实验步 t',fontsize=11)
    fig.suptitle(f'{NAMES[ds]} · z={dim} · 逐步验证{n}/3 · 全部任务',fontsize=20,y=.995);footer(fig,ds);fig.subplots_adjust(top=.94,bottom=.13 if nr==3 else .07,left=.12,right=.90,hspace=.45,wspace=.42);export(fig,f'{ds}_z{dim}_两次切换_全部任务')
def point_table(cases):
    file=ROOT/'reports/plot_points_by_seed.csv';field=['dataset','dim','seed','cut','metric','branch','kind','readout','step','value','origin']
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=field);w.writeheader()
        for ds in DATASETS:
            for dim in (PLAN['controlled_dims'] if ds=='controlled' else PLAN['image_dims'][ds]):
                cs,_=selected_cases(cases,ds,dim)
                curves=[(m,'changing','fixed','mu') for m in ['ref0','ref1000','old','mse']]+[('probe','changing','fixed','mu')]
                if ds!='controlled':curves += [('mse','stable','fixed','mu'),('probe','changing','current','mu'),('probe','stable','fixed','mu'),('probe','changing','frozen','mu'),('probe','stable','frozen','mu'),('probe','changing','fixed','sampled_z')]
                for c in cs:
                    for metric,branch,kind,rd in curves:
                        for cut in PLAN['cuts']:
                            for t,y in points(c,metric,branch,kind,rd):
                                if cut-30<=t<=cut+30:w.writerow(dict(dataset=ds,dim=dim,seed=c['seed'],cut=cut,metric=metric,branch=branch,kind=kind,readout=rd,step=t,value=y,origin='verified_dense_replay' if c['dense'] else 'archived_measured_point'))
def main():
    cases=load();count=sum(c['dense'] is not None for c in cases)
    for ds in DATASETS:
        panel(ds,cases,PLAN['cuts'])
        for cut in PLAN['cuts']:panel(ds,cases,[cut])
        for dim in (PLAN['controlled_dims'] if ds=='controlled' else PLAN['image_dims'][ds]):detail(ds,dim,cases)
    point_table(cases)
    for file,h in INPUTS.items():
        if sha(file)!=h:raise ValueError('Complete plot source changed during rendering')
    write(ROOT/'reports/figure_manifest.json',dict(script_sha256=sha(Path(__file__)),source_files=INPUTS,figures={str(f.relative_to(ROOT)):sha(f) for f in FIG.glob('*') if f.suffix in ['.png','.svg']},point_table_sha256=sha(ROOT/'reports/plot_points_by_seed.csv')))
    write(ROOT/'reports/figure_status.json',dict(status='complete' if count==60 else 'partial',dense_conditions=count,planned=60,windows=PLAN['windows'],points_each_window=61,interpolation=False,script_sha256=sha(Path(__file__))))
    lines=['# 四数据集逐步切换图','',f'逐步补测完整通过{count}/60条件。窗口为970–1030与1970–2030，各61个真实时点。','', '探针纵轴为真实准确率/BA，使用局部纵轴；原始逐步点不平滑、不插值。每个面板标图例。未完成时仅展示原日志实测点并标完成数，不能当作补测已完成。','']
    for ds in DATASETS:lines.append(f'- [{NAMES[ds]} 全部维数](../figures/{ds}_逐步切换_全部维数.png)')
    lines+=['','## 每次切换与单维数任务明细','']
    for f in sorted(FIG.glob('*.png')):
        if not f.name.endswith('全部维数.png'):lines.append(f'- [{f.stem}](../figures/{f.name})')
    (ROOT/'reports/GALLERY.md').write_text('\n'.join(lines)+'\n');print('Rendered dense figures, conditions='+str(count),flush=True)
if __name__=='__main__':main()
