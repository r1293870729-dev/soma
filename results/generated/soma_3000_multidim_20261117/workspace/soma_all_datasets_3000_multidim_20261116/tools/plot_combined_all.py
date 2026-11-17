"""Combine unchanged3000-step signals, MSE and every recorded probe task."""
from pathlib import Path
import csv,json,sys,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import render as source
from paths import sha,write,verify,PLAN

OUT=ROOT/'figures/合图_信号_MSE_全部探针_20261117'
REPORT=ROOT/'reports/combined_20261117'
ORDER={
    'controlled':['theta'],
    'dsprites':['x','y','scale','shape','orientation_shape2_bin8','orientation_raw'],
    'shapes3d':['floor_hue','wall_hue','object_hue','scale','shape','orientation_shape0','orientation_raw'],
    'cifar':['class'],
}
COLORS={0:'#d97706',1000:'#7c3aed','old':'#dc2626',1:'#2563eb',2:'#d97706',3:'#15803d','mse':'#0f766e','theta':'#334155'}
SIGNAL_HANDLES=[Line2D([],[],color=COLORS[0],lw=2,label='新 Ψ ref0（左轴）'),Line2D([],[],color=COLORS[1000],lw=2,label='新 Ψ ref1000（左轴）'),Line2D([],[],color=COLORS['old'],lw=2,label='旧 Ξ（右轴）')]
PROBE_HANDLES=[Line2D([],[],color=COLORS[s],ls=style,lw=1.6,label=f'固定 S{s} '+('μ' if readout=='mu' else 'z')) for s in [1,2,3] for readout,style in [('mu','-'),('sampled_z','--')]]
TABLE={};AUDIT=[];EXPORTS=[]

def aggregate(cases,metric,task='',stage=0,readout=''):
    ds=cases[0]['dataset'];dim=cases[0]['dim'];seeds=tuple(c['seed'] for c in cases);key=(ds,dim,seeds,metric,task,stage,readout)
    if key in TABLE:return TABLE[key]
    values={}
    for c in cases:
        if metric=='probe':
            if ds=='controlled':points=source.series(c,'changing','probe_mu')
            else:points=source.probe_points(c['branches']['changing']['probes'],task,stage,readout)
        else:points=source.series(c,'changing',metric)
        seen=set()
        for t,v in points:
            assert 0<=t<=3000 and np.isfinite(v) and t not in seen;seen.add(t)
            values.setdefault(t,[]).append(float(v))
        AUDIT.append(dict(dataset=ds,dimension=dim,seed=c['seed'],metric=metric,task=task,probe_stage=stage,readout=readout,source_points=len(points),first_step=min(seen) if seen else None,last_step=max(seen) if seen else None))
    t=sorted(values)
    result=[dict(dataset=ds,dimension=dim,seeds=','.join(map(str,seeds)),metric=metric,task=task,probe_stage=stage,readout=readout,step=x,n=len(values[x]),mean=float(np.mean(values[x])),minimum=min(values[x]),maximum=max(values[x])) for x in t]
    if result:assert all(r['n']==len(cases) for r in result)
    TABLE[key]=result;return result

def line(ax,records,color,label='',style='-',shade=True,marker=None):
    if not records:return
    t=[r['step'] for r in records];v=[r['mean'] for r in records]
    ax.plot(t,v,color=color,ls=style,lw=1.35,label=label,marker=marker,ms=2.5,markevery=max(1,len(t)//20),zorder=3)
    if shade and records[0]['n']>1:ax.fill_between(t,[r['minimum'] for r in records],[r['maximum'] for r in records],color=color,alpha=.10,lw=0,zorder=1)

def axis(ax,probe=False):
    ax.set_xlim(0,3000);ax.set_xticks([0,500,1000,1500,2000,2500,3000]);ax.tick_params(axis='both',labelsize=9);ax.grid(alpha=.15)
    for t in [1000,2000]:ax.axvline(t,color='#666666',lw=.8,ls=':',zorder=0)
    if probe:ax.set_ylim(0,1.02)
    else:ax.set_ylim(bottom=0)

def cell_signal(ax,cases):
    right=ax.twinx();right.spines['right'].set_visible(True);right.tick_params(axis='y',colors=COLORS['old'],labelsize=9)
    for age in [0,1000]:line(ax,aggregate(cases,f'ref{age}'),COLORS[age],marker='o')
    line(right,aggregate(cases,'old'),COLORS['old'],shade=True)
    axis(ax);right.set_ylim(bottom=0);ax.set_ylabel('新 Ψ（左轴）',fontsize=10,color='#6d28d9');right.set_ylabel('旧 Ξ（右轴）',fontsize=10,color=COLORS['old']);right.ticklabel_format(axis='y',style='sci',scilimits=(-3,4));ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4))
    return right

def cell_mse(ax,cases,upper):
    line(ax,aggregate(cases,'mse'),COLORS['mse']);axis(ax);ax.set_ylim(0,upper);ax.set_ylabel('重构 MSE（线性轴）',fontsize=10);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4))

def cell_probe(ax,cases,task):
    ds=cases[0]['dataset'];axis(ax,True)
    if ds=='controlled':
        line(ax,aggregate(cases,'probe',task='theta',readout='mu'),COLORS['theta'],marker='o');ax.axhline(1/8,color='#999999',lw=.7,ls='--');ax.set_ylabel('当前阶段 θ\n普通准确率',fontsize=10);return
    classes=set()
    for stage in [1,2,3]:
        for readout,style in [('mu','-'),('sampled_z','--')]:line(ax,aggregate(cases,'probe',task,stage,readout),COLORS[stage],style=style,shade=readout=='mu')
    for c in cases:
        classes.update(int(r['classes']) for r in c['branches']['changing']['probes'] if r['task']==task)
    assert len(classes)==1,(ds,task,classes)
    ax.axhline(1/next(iter(classes)),color='#999999',lw=.7,ls='--');ax.set_ylabel(source.TASKS.get(task,task)+'\n平衡准确率',fontsize=10)

def dataset_axes(fig,grid,ds,cases,start=0):
    dims=source.DIMS[ds];nr=2+len(ORDER[ds]);upper=max(r['maximum'] for d in dims for r in aggregate([c for c in cases if c['dim']==d],'mse'))*1.04
    for j,dim in enumerate(dims):
        selected=sorted([c for c in cases if c['dim']==dim],key=lambda c:c['seed']);assert len(selected) in [1,3]
        axes=[fig.add_subplot(grid[start+i,j]) for i in range(nr)];axes[0].set_title(f'z={dim} · n={len(selected)} seed'+('s' if len(selected)>1 else ''),fontsize=12,pad=10)
        cell_signal(axes[0],selected);cell_mse(axes[1],selected,upper)
        for i,task in enumerate(ORDER[ds]):cell_probe(axes[i+2],selected,task)
        for a in axes[:-1]:a.tick_params(axis='x',labelbottom=False)
        axes[-1].set_xlabel('训练步 t（S1 / S2 / S3 各1000步）',fontsize=10)

def export(fig,stem,dpi=210):
    for ext in ['png','svg']:
        path=OUT/(stem+'.'+ext);fig.savefig(path,dpi=dpi,facecolor='white');EXPORTS.append(path)
    plt.close(fig);print('Exported '+stem,flush=True)

def dataset_figure(ds,cases,single=False):
    ncols=len(source.DIMS[ds]);nr=2+len(ORDER[ds]);w=5.7*ncols;h=2.5*nr+3.1;fig=plt.figure(figsize=(w,h));grid=fig.add_gridspec(nr,ncols,left=.065,right=.945,bottom=1.15/h,top=1-2.0/h,hspace=.31,wspace=.44)
    dataset_axes(fig,grid,ds,cases)
    seed=cases[0]['seed'];mode=f'原 seed{seed} 轨迹' if single else '三 seed 均值与范围'
    fig.suptitle(f'{source.NAMES[ds]} · 信号 + MSE + 全部下游探针 · 各潜维数',fontsize=20,y=1-.23/h)
    fig.text(.5,1-.67/h,mode+'  |  每列一个潜维数；从上到下为信号、MSE、各项原探针',ha='center',fontsize=11)
    fig.legend(handles=SIGNAL_HANDLES,loc='upper center',bbox_to_anchor=(.5,1-.95/h),ncol=3,frameon=False,fontsize=10)
    if ds!='controlled':fig.legend(handles=PROBE_HANDLES,loc='upper center',bbox_to_anchor=(.5,1-1.35/h),ncol=6,frameon=False,fontsize=10)
    else:fig.text(.5,1-1.53/h,'下排：历史当前阶段 θ 八分类，μ读出；灰虚线为1/8机会水平。',ha='center',fontsize=10,color='#334155')
    foot='ref0从第1批记录；ref1000从第1001批起；旧Ξ预热无效区不补零。MSE为原始观测/像素平方误差均值，线性轴。\n'
    foot+=('controlled保留历史训练与当前阶段训练池80/20回测，原15个θ点不替换，0/1/3000为离线补测。' if ds=='controlled' else '固定S1/S2/S3探针池：实线μ、虚线固定潜噪声z；新阶段到达后才评估。灰虚线为机会水平；方向主任务与原始辅助分别保留。')
    foot+='\n'+('阴影为三seed最小—最大范围，不是置信区间；新旧信号使用不同坐标轴，峰高不能直接排名。' if not single else '本图保留单seed实际测点；三seed均值另图。未平滑、未重训、未补造早期探针。')
    fig.text(.5,.21/h,foot,ha='center',va='bottom',fontsize=10,color='#555555')
    export(fig,f'{"原seed" if single else "三seed"}_{ds}_信号_MSE_全部探针_各维度')

def master_figure(cases):
    datasets=list(ORDER);heights=[2+len(ORDER[ds])+.55 for ds in datasets];h=2.5*sum(heights)+2.1;fig=plt.figure(figsize=(28.5,h));outer=fig.add_gridspec(4,1,height_ratios=heights,left=.055,right=.945,bottom=1.05/h,top=1-1.8/h,hspace=.065)
    for i,ds in enumerate(datasets):
        nr=2+len(ORDER[ds]);grid=outer[i].subgridspec(nr+1,len(source.DIMS[ds]),height_ratios=[.45]+[1]*nr,hspace=.28,wspace=.38);header=fig.add_subplot(grid[0,:]);header.axis('off')
        description='原当前阶段 θ 普通准确率' if ds=='controlled' else '固定S1/S2/S3全部因子，μ / z平衡准确率'
        header.text(.5,.50,source.NAMES[ds]+'  |  '+description,ha='center',va='center',fontsize=18,weight='bold',transform=header.transAxes)
        dataset_axes(fig,grid,ds,[c for c in cases if c['dataset']==ds],start=1)
    fig.suptitle('四个数据集 · 全部14种潜维数 · 信号 + MSE + 全部下游探针',fontsize=25,y=1-.22/h)
    fig.text(.5,1-.68/h,'全部42个父条件已完成；每条实线为3seed均值，阴影为seed最小—最大范围（不是置信区间）。各数据集按列排列潜维数。',ha='center',fontsize=12)
    fig.legend(handles=SIGNAL_HANDLES,loc='upper center',bbox_to_anchor=(.5,1-.93/h),ncol=3,frameon=False,fontsize=12)
    fig.legend(handles=PROBE_HANDLES,loc='upper center',bbox_to_anchor=(.5,1-1.29/h),ncol=6,frameon=False,fontsize=11)
    fig.text(.5,.19/h,'所有主轨迹均为1000/1000/1000步；固定参考0从第1批测量，参考1000从第1001批起，不使用未来参考倒填。\nMSE采用线性坐标；新旧信号各读其轴，不能按峰高直接排名。controlled保留原历史θ回测；其他数据集使用固定独立探针角色。\n未重新训练、未改权重、未换seed或探针；原始测点和输入SHA另存。可打开各数据集PNG/SVG大图查看细节。',ha='center',va='bottom',fontsize=11,color='#555555')
    export(fig,'00_四数据集_全部维度_信号_MSE_全部探针_总合图',dpi=155)

def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);verify();frozen=json.loads((ROOT/'configs/analysis_frozen.json').read_text());assert sha(Path(source.__file__))==frozen['tools/render.py'];status=json.loads((ROOT/'reports/FINAL_STATUS.json').read_text());assert status['status']=='complete' and status['completed_parents']==42 and not status['failures']
    cases=source.load_cases();assert len(cases)==42
    for ds in ORDER:
        selected=[c for c in cases if c['dataset']==ds];expected=set(ORDER[ds]);actual={'theta'} if ds=='controlled' else {r['task'] for c in selected for r in c['branches']['changing']['probes']};assert actual==expected,(ds,actual,expected)
        assert {c['dim'] for c in selected}==set(source.DIMS[ds]);assert all(sum(c['dim']==d for c in selected)==3 for d in source.DIMS[ds]);dataset_figure(ds,selected)
        seed=0 if ds=='controlled' else 10;dataset_figure(ds,[c for c in selected if c['seed']==seed],single=True)
    master_figure(cases)
    values=[r for points in TABLE.values() for r in points]
    with (REPORT/'plotted_values.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    write(REPORT/'curve_audit.json',AUDIT);write(REPORT/'input_manifest.json',source.inputs)
    for p,h in source.inputs.items():assert sha(p)==h,'Source changed '+p
    write(REPORT/'verification.json',dict(status='complete',parents=42,latent_conditions=14,stage_steps=[1000]*3,all_recorded_probe_tasks_included=True,probe_tasks=ORDER,probe_averaging='three seeds at exact same measured step; raw observations not smoothed',uncertainty='min-max over three seeds, not confidence interval',MSE_axis='linear, same limits within each dataset',signals_dual_axes=True,new_training_updates=0,code_sha256=sha(Path(__file__)),source_loader_sha256=sha(Path(source.__file__)),png_count=9,svg_count=9,plotted_rows=len(values),matplotlib=matplotlib.__version__,numpy=np.__version__))
    lines=['# 信号、MSE与全部下游探针合图','', '全部42/42个父条件已经完成，无失败。本次仅重新排图，未训练或改变原始结果。','',
       '每个数据集一组：每列一个维数；第一排新旧信号，第二排原始MSE，后续每排一个原探针任务。四数据集总合图包含同样的全部任务，便于在一张图片中查看。','',
       '三seed版为所有seed的均值与最小—最大范围；原seed版保留controlled seed0、其他seed10的实际曲线。数据点未经平滑；阴影不是置信区间。MSE是线性坐标，因此视觉形状会与此前对数坐标图不同，数值保持一致。','',
       '探针颜色：蓝=S1固定池，橙=S2固定池，绿=S3固定池；实线μ，虚线固定潜噪声z。自定义方向探针也按其原阶段S3完整纳入。controlled仍为原当前阶段θ普通准确率，不能等同独立旧任务保持率。','',
       '信号：橙=新Ψ参考0（左轴），紫=新Ψ参考1000（左轴），红=旧Ξ（右轴）。参考1000从1001批才可使用；旧Ξ预热无效区不补零。灰虚线为探针机会水平。','',
       '本次只展示阶段变化主轨迹。已运行稳定分支对照仍保留在原figures/稳定流对照_*，不混为独立seed。','',
       '## 图像','']
    for p in EXPORTS:
        if p.suffix=='.png':lines.append(f'- [{p.stem}](../../{p.relative_to(ROOT).as_posix()})')
    lines+=['','代码重画需要本工程的原始结果与历史依赖：`.venv/bin/python tools/plot_combined_all.py`。打包图片可独立查看。','']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    package=ROOT/'信号_MSE_全部下游探针_各维度_20261117.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in EXPORTS+list(REPORT.glob('*'))+[Path(__file__),ROOT/'tools/render.py',ROOT/'src/paths.py',ROOT/'configs/plan.json',ROOT/'configs/frozen.json',ROOT/'configs/analysis_frozen.json',ROOT/'docs/SIGNALS_AND_PROBES.md']:z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(package) as z:assert z.testzip() is None
    print(json.dumps(dict(status='complete',parents=42,pngs=9,svgs=9,package=str(package)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
