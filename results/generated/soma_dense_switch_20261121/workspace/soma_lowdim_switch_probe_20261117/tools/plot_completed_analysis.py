"""Scientific summary figures from all completed conditions; no training imports."""
from pathlib import Path
import csv,sys,json
import numpy as np
import plots_switch as p
plt=p.plt
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/final_analysis_20261119';FIG=ROOT/'figures/final_analysis_20261119';FIG.mkdir(exist_ok=True)
BLUE='#2563eb';PURPLE='#7c3aed';GRAY='#64748b';ORANGE='#d97706';GREEN='#0f766e';PALETTE=['#dc2626','#d97706','#16a34a','#2563eb','#7c3aed']
def read(name):return list(csv.DictReader((REPORT/name).open()))
def select(rr,**keys):return [r for r in rr if all(str(r[k])==str(v) for k,v in keys.items())]
def one(rr,**keys):
    v=select(rr,**keys)
    if len(v)!=1:raise ValueError('Expected single summary '+str(keys))
    return v[0]
def error(ax,x,rr,key,color,label,scale=1,style='-o'):
    mean=np.array([float(r[key+'_mean'])*scale for r in rr]);lo=np.array([float(r[key+'_min'])*scale for r in rr]);hi=np.array([float(r[key+'_max'])*scale for r in rr]);ax.errorbar(x,mean,yerr=np.array([mean-lo,hi-mean]),fmt=style,color=color,label=label,lw=1.5,ms=4,capsize=3)
def decorate(ax):ax.grid(alpha=.16);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
def export(fig,name):
    for ext in ['png','svg']:fig.savefig(FIG/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)
def capacities():
    rr=read('final_by_dimension.csv');vv=read('S1_validation_by_dimension.csv');cc=read('controlled_by_dimension.csv');cases=p.load();fig,axes=plt.subplots(3,4,figsize=(22,13))
    for j,ds in enumerate(['controlled','dsprites','shapes3d','cifar']):
        dims=p.DIMS[ds];x=np.arange(5)
        for ax in axes[:,j]:decorate(ax);ax.set_xticks(x,dims)
        axes[0,j].set_title(p.NAMES[ds],fontsize=17)
        if ds=='controlled':
            error(axes[0,j],x,[one(cc,dim=d) for d in dims],'theta_final_accuracy',BLUE,'历史当前阶段θ',100);axes[0,j].axhline(12.5,color='#999999',ls=':',lw=.8)
            ax=axes[1,j]
            for d,col in zip(dims,PALETTE):p.old.meanline(ax,[p.get(c,'probe') for c in cases if c['dataset']==ds and c['dim']==d],col,'z='+str(d))
            ax.set_xlim(700,3000);ax.set_xticks([1000,2000,3000]);ax.set_ylim(0,1.02)
            for t in [1000,2000]:ax.axvline(t,ls=':',color='#777777',lw=.8)
            ax.set_ylabel('历史θ普通准确率');ax.set_xlabel('训练步');ax.legend(fontsize=8,ncol=2)
            series=[]
            for d in dims:
                vals=[np.mean([v for t,v in p.get(c,'mse') if t>=2900]) for c in cases if c['dataset']==ds and c['dim']==d];series.append(dict(mse_mean=np.mean(vals),mse_min=min(vals),mse_max=max(vals)))
            error(axes[2,j],x,series,'mse',GREEN,'当前训练流末101步MSE');axes[2,j].set_title('历史当前流训练MSE',fontsize=11)
        else:
            for rd,col,label in [('mu',BLUE,'μ'),('sampled_z',ORANGE,'采样z')]:error(axes[0,j],x,[one(rr,dataset=ds,dim=d,readout=rd) for d in dims],'S1_refit_3000',col,label,100)
            chance=float(one(rr,dataset=ds,dim=dims[0],readout='mu')['chance']);axes[0,j].axhline(chance*100,color='#999999',ls=':',lw=.8)
            r=[one(rr,dataset=ds,dim=d,readout='mu') for d in dims]
            for key,col,label in [('refit_final_changing_minus_stable_pp',BLUE,'重拟合头'),('frozen_final_changing_minus_stable_pp',PURPLE,'900步冻结头')]:error(axes[1,j],x,r,key,col,label)
            axes[1,j].axhline(0,color='#777777',lw=.8);axes[1,j].set_ylabel('末期固定S1：变化−稳定（百分点）');axes[1,j].set_title('负值：变化流较低；冻结头可包含坐标漂移',fontsize=10);axes[1,j].legend(fontsize=9)
            for branch,col,label in [('changing',GREEN,'变化流模型'),('stable',GRAY,'稳定流模型')]:error(axes[2,j],x,[one(vv,dataset=ds,dim=d,branch=branch) for d in dims],'pixel_mse',col,label)
            axes[2,j].set_title('同一固定S1验证集上的像素MSE',fontsize=11)
        axes[0,j].set_ylim(0,102);axes[0,j].set_ylabel('末次探针准确率（%）');axes[0,j].legend(fontsize=9);axes[2,j].set_yscale('log');axes[2,j].set_ylabel('MSE（对数轴）');axes[2,j].set_xlabel('潜维数');axes[2,j].legend(fontsize=8)
        if ds!='controlled':axes[1,j].set_xlabel('潜维数')
    fig.suptitle('四数据集：容量、任务读出与重构的关系',fontsize=23,y=.99);fig.text(.5,.018,'均值与误差线=3个seed的均值和最小—最大；不是置信区间。图像上排=固定S1原任务等权平均BA，controlled为历史当前θ普通准确率。\n图像末次探针对应更新2999次；验证在更新3000次后。同S1验证避免比较不同输入分布的MSE；Shapes3D16/64稳定流各有1条严重logvar裁剪记录，均保留。',ha='center',fontsize=11,color='#555555');fig.tight_layout(rect=[0,.075,1,.96],h_pad=2);export(fig,'01_四数据集_维数_探针_重构')
def task_heatmap():
    rr=list(csv.DictReader((ROOT/'reports/final_probe_summary.csv').open()));fig,axes=plt.subplots(2,3,figsize=(20,11))
    defs={'dsprites':[('S1','x'),('S1','y'),('S1','scale'),('S3','shape'),('S3_observable_orientation','orientation_shape2_bin8')], 'shapes3d':[('S1','floor_hue'),('S1','wall_hue'),('S1','object_hue'),('S1','scale'),('S3','shape'),('S3_observable_orientation','orientation_shape0')], 'cifar':[('S1','class'),('S2','class'),('S3','class')]}
    for j,(ds,tasks) in enumerate(defs.items()):
        dims=p.DIMS[ds]
        for i,rd in enumerate(['mu','sampled_z']):
            ax=axes[i,j];values=np.array([[100*float(one(rr,dataset=ds,dim=d,group=g,task=t,readout=rd)['mean']) for d in dims] for g,t in tasks]);im=ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto');labels=[]
            for g,t in tasks:
                name={'orientation_shape2_bin8':'心形方向（8类）','orientation_shape0':'形状0方向（15类）'}.get(t,p.old.TASKS.get(t,t))
                r=one(rr,dataset=ds,dim=dims[0],group=g,task=t,readout=rd);label=('S3' if g.startswith('S3') else g)+' '+name+f"\n随机{100*float(r['chance']):.1f}%";labels.append(label)
            ax.set_yticks(range(len(tasks)),labels);ax.set_xticks(range(5),dims);ax.set_xlabel('潜维数');ax.set_title(p.NAMES[ds]+' · '+('μ' if rd=='mu' else '采样z'),fontsize=15)
            for a in range(len(tasks)):
                for b in range(5):ax.text(b,a,f'{values[a,b]:.1f}',ha='center',va='center',color='white' if values[a,b]<55 else '#111111',fontsize=12)
            ax.grid(False)
    fig.suptitle('全部维数：哪些因子被读出来，哪些仍接近机会水平？',fontsize=23,y=.99);fig.text(.5,.025,'每格=全部3seed的末次BA均值。S1与S3使用各自固定、独立的探针角色；S3任务在到达前不定义。不同任务机会水平不同，颜色不是机会校正后的分数。\n本图保留全部主任务；原始方向辅助任务及逐时点曲线见原GALLERY。μ与z均展示，不按表现选择读出。',ha='center',fontsize=11,color='#555555');fig.subplots_adjust(top=.92,bottom=.13,left=.115,right=.90,wspace=.58,hspace=.42)
    color_ax=fig.add_axes([.935,.17,.012,.65]);fig.colorbar(im,cax=color_ax,label='平衡准确率（%）');export(fig,'02_各维数_全部主任务_mu与z')
def cuts():
    rr=read('transition_macro_by_dimension.csv');fig,axes=plt.subplots(2,3,figsize=(18,9.5))
    for i,cut in enumerate([1000,2000]):
        for j,ds in enumerate(['dsprites','shapes3d','cifar']):
            ax=axes[i,j];dims=p.DIMS[ds];x=np.arange(5)
            for kind,col,label,offset in [('fixed',BLUE,'原重拟合头',-.07),('frozen',PURPLE,'900步冻结头',.07)]:error(ax,x+offset,[one(rr,dataset=ds,dim=d,readout='mu',kind=kind,cut=cut) for d in dims],'paired_increment_pp',col,label)
            ax.axhline(0,color='#777777',lw=.8);ax.set_xticks(x,dims);ax.set_xlabel('潜维数');ax.set_ylabel('固定S1 BA配对变化（百分点）');ax.set_title(f'{p.NAMES[ds]} · 切换{cut}');decorate(ax);ax.legend(fontsize=9)
    fig.suptitle('切换附近的额外探针变化：扣除稳定流后还剩多少？',fontsize=22,y=.99);fig.text(.5,.025,'效应=（变化流后−前）−（稳定流后−前）。预窗口900/950/1000及1900/1950/2000；后窗口1001/1050/1100及2001/2050/2100。\n均值与误差线为全部3seed均值和范围；负值表示相对下降。冻结头效应可含坐标漂移，不等同可读信息消失；本图不作显著性声明。',ha='center',fontsize=11,color='#555555');fig.tight_layout(rect=[0,.10,1,.95],h_pad=2);export(fig,'03_两次切换_稳定流校正_探针变化')
def signal_vs_probe():
    ss=read('signals_by_seed.csv');ee=read('transition_macro_by_seed.csv');fig,axes=plt.subplots(2,3,figsize=(18,10))
    for i,cut in enumerate([1000,2000]):
        for j,ds in enumerate(['dsprites','shapes3d','cifar']):
            ax=axes[i,j]
            for d,col in zip(p.DIMS[ds],PALETTE):
                x=[];y=[]
                for seed in PLAN['image_seeds']:
                    x.append(float(one(ss,dataset=ds,dim=d,seed=seed,cut=cut,metric='ref1000')['post_changing_minus_stable']));y.append(float(one(ee,dataset=ds,dim=d,seed=seed,cut=cut,readout='mu',kind='fixed')['paired_increment_pp']))
                ax.scatter(x,y,c=col,s=25,alpha=.45);ax.scatter([np.mean(x)],[np.mean(y)],c=col,s=95,label='z='+str(d),edgecolors='white');ax.annotate(str(d),(np.mean(x),np.mean(y)),xytext=(5,6),textcoords='offset points',color=col,fontsize=9)
            decorate(ax);ax.axhline(0,color='#777777',lw=.8);ax.axvline(0,color='#aaaaaa',lw=.6);ax.set_ylim(-5,10);ax.set_title(f'{p.NAMES[ds]} · 切换{cut}');ax.set_xlabel('新Ψ ref1000：后窗口变化流−稳定流');ax.set_ylabel('重拟合S1 BA配对变化（百分点）');ax.legend(fontsize=8,ncol=3)
    fig.suptitle('新信号升高，是否同时出现旧任务线性读出下降？',fontsize=22,y=.99);fig.text(.5,.025,'小点=每个seed，大点=维数均值。信号取原定后窗口3点平均，探针按同一预/后窗口计算稳定流校正效应。\n只作描述性对照；不同数据集/维数的固定白化参考不同，不把横轴当通用容量刻度，不合并计算显著性或预测AUROC。',ha='center',fontsize=11,color='#555555');fig.tight_layout(rect=[0,.10,1,.95]);export(fig,'04_新信号与固定S1探针变化')
if __name__=='__main__':
    from study_paths import PLAN,sha,write
    capacities();task_heatmap();cuts();signal_vs_probe();write(REPORT/'summary_figure_manifest.json',dict(script_sha256=sha(Path(__file__)),inputs={str(f.relative_to(ROOT)):sha(f) for f in REPORT.glob('*.csv')},final_task_table_sha256=sha(ROOT/'reports/final_probe_summary.csv'),figures={str(f.relative_to(ROOT)):sha(f) for f in FIG.iterdir() if f.suffix in ['.png','.svg']}));print('Saved4 summary PNG/SVG figures.')
