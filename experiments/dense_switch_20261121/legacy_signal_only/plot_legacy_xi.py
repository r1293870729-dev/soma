"""Read-only legacy Xi plots: every recorded step, all dimensions and seeds."""
from pathlib import Path
import argparse, csv, hashlib, json, math, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parent
NAMES = {'controlled':'controlled','dsprites':'dSprites','shapes3d':'Shapes3D','cifar':'CIFAR / CIFAR-C'}
DIMS = {'controlled':[2,4,8,16,32],'dsprites':[2,4,8,16,32],'shapes3d':[2,4,16,32,64],'cifar':[2,4,16,32,64]}
SEEDS = {'controlled':[0,1,2],**{k:[10,11,12] for k in ['dsprites','shapes3d','cifar']}}
VIEWS = [(1,3000,'完整 3000 步'),(970,1030,'第一次切换 · 前后各 30 步'),(1970,2030,'第二次切换 · 前后各 30 步')]
INPUTS = {}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def pin(p):
    p=Path(p).resolve();h=sha(p)
    if str(p) in INPUTS and INPUTS[str(p)]!=h:raise ValueError('Input changed: '+str(p))
    INPUTS[str(p)]=h;return p

def rows(p):
    with pin(p).open() as f:return list(csv.DictReader(f))

def read_json(p):return json.loads(pin(p).read_text())
def truth(v):return str(v).lower() in {'true','1','1.0'}
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def relocated(original,source,prior):
    p=Path(original)
    for base in [source,prior]:
        for i,part in enumerate(p.parts):
            if part.startswith('soma_'):
                candidate=base.joinpath(*p.parts[i:])
                if candidate.exists():return candidate
    if p.exists():return p
    raise FileNotFoundError(original)

def load(source,prior):
    dense=source/'soma_dense_switch_windows_20261119'
    low=source/'soma_lowdim_switch_probe_20261117'
    old=prior/'soma_all_datasets_3000_multidim_20261116'
    ds_plan=read_json(dense/'configs/plan.json');assert ds_plan['planned_conditions']==60
    assert read_json(dense/'reports/figure_status.json')['dense_conditions']==60
    result={};checks=0;original_checks=0
    for ds,dims in DIMS.items():
        for dim in dims:
            for seed in SEEDS[ds]:
                key=(ds,dim,seed);case=f'{ds}_z{dim}_seed{seed}';df=dense/'results'/case
                assert read_json(df/'summary.json')['status']=='complete'
                if ds=='controlled':
                    raw=rows(df/'native/train_history.csv')
                    values=[(int(r['step']),float(r['xi_pw']),truth(r['baseline_ready']) and truth(r['window_ready'])) for r in raw]
                    archived=read_json(old/'results/all_3000_001'/case/'summary.json')
                    original=rows(relocated(archived['source_dir'],source,prior)/'native/train_history.csv')
                    native={t:(v,e) for t,v,e in values}
                    for r in original:
                        assert native[int(r['step'])]==(float(r['xi_pw']),truth(r['baseline_ready']) and truth(r['window_ready']))
                        original_checks+=1
                    dense_rows=rows(df/'dense_metrics.csv')
                else:
                    parent=(low/'results/lowdim_switch_001' if dim in [2,4] else old/'results/all_3000_001')/case
                    assert read_json(parent/'summary.json')['status']=='complete'
                    raw=rows(parent/'changing/legacy_every_batch.csv')
                    values=[(int(r['step']),float(r['xi']),truth(r['eligible'])) for r in raw]
                    dense_rows=rows(df/'changing/dense_metrics.csv')
                assert [t for t,_,_ in values]==list(range(1,3001)),case
                assert all(math.isfinite(v) and v>=0 for _,v,_ in values),case
                native={t:(v,e) for t,v,e in values}
                expected=[*range(970,1031),*range(1970,2031)]
                assert [int(r['step']) for r in dense_rows]==expected,case
                for r in dense_rows:
                    assert native[int(r['step'])]==(float(r['xi']),truth(r['eligible'])),case
                    checks+=1
                result[key]=values
    return result,dict(conditions=60,source_rows=180000,dense_overlap_exact_rows=checks,controlled_archived_overlap_exact_rows=original_checks)

def aggregate(data):
    result={}
    for ds,dims in DIMS.items():
        for dim in dims:
            a=np.array([[v if e else np.nan for _,v,e in data[ds,dim,seed]] for seed in SEEDS[ds]],dtype=float)
            mask=np.isfinite(a).all(axis=0)
            mean=np.full(3000,np.nan);lo=mean.copy();hi=mean.copy()
            mean[mask]=a[:,mask].mean(axis=0);lo[mask]=a[:,mask].min(axis=0);hi[mask]=a[:,mask].max(axis=0)
            result[ds,dim]=(mean,lo,hi,mask)
    return result

def setup():
    p=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if p.exists():font_manager.fontManager.addfont(str(p));plt.rcParams['font.family']='Arial Unicode MS'
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})

def decorate(ax,view):
    lo,hi,_=view;ax.set_xlim(0 if lo==1 else lo,hi)
    if lo==1:
        ax.xaxis.set_major_locator(MultipleLocator(500))
        for t in [1000.5,2000.5]:ax.axvline(t,color='#6b7280',ls='--',lw=1)
    else:
        ax.xaxis.set_major_locator(MultipleLocator(10));ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.axvline((1000.5 if hi==1030 else 2000.5),color='#6b7280',ls='--',lw=1)
        ax.grid(axis='x',which='minor',alpha=.07)
    ax.grid(which='major',alpha=.19);ax.yaxis.set_major_locator(MaxNLocator(4))
    f=ScalarFormatter(useOffset=False,useMathText=True);f.set_powerlimits((-3,4));ax.yaxis.set_major_formatter(f)
    ax.tick_params(labelsize=10);ax.set_xlabel('训练步 t');ax.set_ylabel('旧信号 Ξ · 原始值')

def curve(ax,agg,ds,dim,view,color='#c92d39',label='旧Ξ：3 seed 均值',shade=True,zero_note=True):
    lo,hi,_=view;t=np.arange(lo,hi+1);mean,mn,mx,mask=agg[ds,dim];ix=t-1
    line,=ax.plot(t,mean[ix],color=color,label=label,lw=1.65,marker='o' if lo!=1 else None,ms=2.4)
    if shade:ax.fill_between(t,mn[ix],mx[ix],color=color,alpha=.17,lw=0)
    good=mask[ix]
    for start in np.flatnonzero((~good)&np.r_[True,good[:-1]]):
        end=start
        while end+1<len(t) and not good[end+1]:end+=1
        ax.axvspan(t[start]-.5,t[end]+.5,color='#d1d5db',alpha=.45,lw=0)
    ax.set_ylim(bottom=0)
    if not good.any() or np.nanmax(mx[ix])==0:
        ax.set_ylim(0,1)
        if zero_note:ax.text(.5,.56,'本窗口有效旧Ξ均为 0' if good.any() else '旧信号尚未就绪',transform=ax.transAxes,ha='center',color='#6b7280',fontsize=10)
    else:ax.set_ylim(0,float(np.nanmax(mx[ix]))*1.09)
    return line

def export(fig,directory,name):
    for ext in ['png','svg']:fig.savefig(directory/(name+'.'+ext),dpi=250,facecolor='white')
    plt.close(fig)

def figures(agg,directory):
    setup()
    for ds,dims in DIMS.items():
        fig,axes=plt.subplots(5,3,figsize=(18,16.2),squeeze=False)
        for i,dim in enumerate(dims):
            for j,view in enumerate(VIEWS):
                ax=axes[i,j];decorate(ax,view);line=curve(ax,agg,ds,dim,view)
                ax.set_title(f'z={dim} · {view[2]}',fontsize=13,pad=8)
                ax.legend(handles=[line,Patch(facecolor='#c92d39',alpha=.17,label='3 seed 最小–最大范围')],loc='upper right' if view[0]==1 else 'upper left',fontsize=8.4,framealpha=.9)
        fig.suptitle(f'{NAMES[ds]} · 旧信号 Ξ · 全部维数',fontsize=24,y=.991)
        fig.text(.5,.964,'每行一个潜维数；左：1–3000步；中：970–1030步；右：1970–2030步。只展示历史旧Ξ。',ha='center',fontsize=12)
        fig.text(.5,.015,'每个条件使用全部3 seed；均值与最小–最大范围，非置信区间。各面板独立线性纵轴，保留真实量纲。\n虚线在1000/1001、2000/2001之间；灰区为预热未就绪，不将占位0当有效信号。逐步实测、不平滑。\n旧Ξ保留历史已知stage基线路由；切换尖峰受fallback及基线更新影响，不作为未知切换盲检测证明。',ha='center',fontsize=10.5,color='#4b5563')
        fig.subplots_adjust(left=.075,right=.98,top=.925,bottom=.095,hspace=.52,wspace=.31)
        export(fig,directory,ds+'_old_xi_full_and_windows')
    colors=['#c92d39','#e48a0a','#14805e','#2465bd','#7c3eb5']
    fig,axes=plt.subplots(4,3,figsize=(20,14),squeeze=False)
    for i,(ds,dims) in enumerate(DIMS.items()):
        for j,view in enumerate(VIEWS):
            ax=axes[i,j];decorate(ax,view)
            # Overlay uses all five means, with one scale shared by dimensions in each panel.
            maxima=[]
            for dim,color in zip(dims,colors):
                curve(ax,agg,ds,dim,view,color,f'z={dim}',shade=False,zero_note=False)
                a=agg[ds,dim][0][view[0]-1:view[1]]
                if np.isfinite(a).any():maxima.append(float(np.nanmax(a)))
            ax.set_ylim(0,max(maxima)*1.10 if maxima and max(maxima)>0 else 1)
            ax.set_title(NAMES[ds]+' · '+view[2],fontsize=14)
            ax.legend(loc='upper right' if view[0]==1 else 'upper left',fontsize=9,ncol=2)
    fig.suptitle('四数据集 · 仅旧信号 Ξ · 3000步与两次切换放大',fontsize=24,y=.992)
    fig.text(.5,.02,'颜色=潜维数；每线为全部3 seed均值，原始量纲、线性轴、不平滑。各面板独立纵轴。\n详细单维数图另含seed范围；灰区=预热无效。旧Ξ含已知stage基线路由，切换后第21批更换当前stage基线。',ha='center',fontsize=11,color='#4b5563')
    fig.subplots_adjust(left=.075,right=.98,top=.945,bottom=.09,hspace=.43,wspace=.30)
    export(fig,directory,'all_four_datasets_old_xi_overview')

def main(a):
    source=Path(a.source_workspace).resolve();prior=Path(a.prior_workspace).resolve() if a.prior_workspace else source
    output=Path(a.output).resolve()
    if (output/'manifest.json').exists() and not a.overwrite:raise ValueError('Choose a new output directory, or explicitly --overwrite these derived plots')
    script_hash=sha(Path(__file__))
    output.mkdir(parents=True,exist_ok=True);figdir=output/'figures';figdir.mkdir(exist_ok=True)
    data,validation=load(source,prior);agg=aggregate(data)
    fields=['dataset','latent_dim','seed','step','xi_raw','eligible','xi_plotted']
    for ds in DIMS:
        with (output/(ds+'_legacy_xi_by_seed.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for dim in DIMS[ds]:
                for seed in SEEDS[ds]:
                    for t,v,e in data[ds,dim,seed]:w.writerow(dict(dataset=ds,latent_dim=dim,seed=seed,step=t,xi_raw=v,eligible=e,xi_plotted=v if e else ''))
    with (output/'plot_aggregates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','latent_dim','step','mean','min','max','n_valid_seeds'])
        for (ds,dim),(mean,lo,hi,mask) in agg.items():
            for ix in range(3000):w.writerow([ds,dim,ix+1,*([mean[ix],lo[ix],hi[ix],3] if mask[ix] else ['','','',0])])
    figures(agg,figdir)
    for p,h in INPUTS.items():
        if sha(p)!=h:raise ValueError('Input changed during plotting: '+p)
    validation.update(status='pass',new_training_updates=0,old_signal_only=True,interpolation=False,steps_per_condition=3000,window_points=61,
                      invalid_points_shown_as_gaps=True,raw_signal_formula='relu(lambda_max - (1 + sqrt(d/W))**2)',
                      full_source_sha256=INPUTS,script_sha256=script_hash,dimensions=DIMS,seeds=SEEDS)
    write(output/'validation.json',validation)
    lines=['# 四数据集：仅旧信号Ξ','',
           '每条变化流均覆盖1–3000步，使用全部60条件、每种维数全部三个seed。每个数据集一张图：五行对应五种潜维数，三列依次是完整3000步、970–1030步、1970–2030步。两次切换后第一批分别为1001、2001，局部各61个真实测点。','',
           '红线为3seed均值，浅红为最小–最大范围；不是置信区间。各面板独立线性纵轴，按刻度比较；不平滑、不归一化、不画新信号。预热无效值留空并标灰区，不把日志占位0当有效观测。','',
           '旧Ξ=relu(最大白化协方差特征值−MP边缘)，使用历史EMA/收缩白化及已知stage路由。前20个新stage批次可用上一stage基线fallback，第21批转为当前stage基线；尖峰不等于盲检测证据。','',
           'controlled完整逐步数据来自已完成且已核验的确定性重放native/train_history.csv；图像来自原逐批legacy_every_batch.csv。全部7320个窗口重叠点逐项一致；controlled另核对原稀疏日志。没有重新训练、修改信号或改动历史结果。','',
           '- [四数据集叠加总览](figures/all_four_datasets_old_xi_overview.png)','']
    for ds in DIMS:lines.append(f'- [{NAMES[ds]}：全部维数，完整与局部](figures/{ds}_old_xi_full_and_windows.png) · [SVG](figures/{ds}_old_xi_full_and_windows.svg)')
    lines+=['','每数据集CSV保留全部原始旧Ξ、有效标记和绘图值；plot_aggregates.csv记录绘图聚合值；validation.json保存源文件哈希和一致性检查。旧7月源码缺失部分仍按历史协议重建标签保留。','',
            '复画命令（仅读取日志，依赖numpy和matplotlib）：','',
            '```bash','python plot_legacy_xi.py --source-workspace /path/to/outputs --output /path/to/new_output',
            '# GitHub分开存放两个归档时，另传 --prior-workspace /path/to/prior_archive/workspace','```','']
    (output/'README.md').write_text('\n'.join(lines))
    if sha(Path(__file__))!=script_hash:raise ValueError('Plot source changed during rendering')
    if (output/'plot_legacy_xi.py').resolve()!=Path(__file__).resolve():(output/'plot_legacy_xi.py').write_bytes(Path(__file__).read_bytes())
    payload=[p for p in output.rglob('*') if p.is_file() and p.name not in ['manifest.json','legacy_xi_four_datasets.zip'] and '__pycache__' not in p.parts]
    write(output/'manifest.json',dict(schema=1,files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(payload)},validation='validation.json'))
    with zipfile.ZipFile(output/'legacy_xi_four_datasets.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [*sorted(payload),output/'manifest.json']:z.write(p,p.relative_to(output))
    print(json.dumps(dict(status='complete',conditions=60,rows=180000,pngs=5,svgs=5,output=str(output)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-workspace',default=str(ROOT.parent));p.add_argument('--prior-workspace');p.add_argument('--output',default=str(ROOT));p.add_argument('--overwrite',action='store_true');main(p.parse_args())
