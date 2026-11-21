"""Controlled switch windows from immutable historical logs; no new probe points."""
from pathlib import Path
import csv,json
import numpy as np
import plots_switch as p
from study_paths import ROOT,OLD,verify,sha,write
plt=p.plt
FIG=ROOT/'figures/controlled_switch_windows_20261119';FIG.mkdir(exist_ok=True)
REPORT=ROOT/'reports/controlled_switch_windows_20261119';REPORT.mkdir(exist_ok=True)
DIMS=[2,4,8,16,32];WINDOWS=[(1000,(700,1300),'S1圆 → S2环面'),(2000,(1700,2300),'S2环面 → S3局部分支')]
def load():
    cases=[]
    for file in sorted((OLD/'results/all_3000_001').glob('controlled_z*/summary.json')):
        s=json.loads(p.pin(file).read_text());assert s['status']=='complete'
        base=Path(s['source_dir']);tr=p.read(base/'native/train_history.csv');pr=[r for r in p.read(base/'native/probe_history.csv') if r['task']=='theta'];ext=p.read(base/'online_theta_extensions.csv');pr=sorted(pr+ext,key=lambda r:int(r['step']))
        assert len(pr)==18 and len({int(r['step']) for r in pr})==18 and int(tr[-1]['step'])==3000
        if s['seed']==0:
            original=[r for r in p.read(p.old.HIST/'results/reconstructed_B_001'/f'B_z{s["latent_dim"]}_loss_plus_xi/probe_history.csv') if r['task']=='theta']
            native=[r for r in pr if int(r['step']) not in [0,1,3000]]
            assert [(int(r['step']),float(r['accuracy'])) for r in native]==[(int(r['step']),float(r['accuracy'])) for r in original]
        sig={age:p.jl(base/f'ref{age}/signals.jsonl') for age in [0,1000]}
        assert sig[0][0]['step']==1 and sig[1000][0]['step']==1001
        series=dict(mse=p.points(tr,'reconstruction'),old=p.points(tr,'xi_pw',lambda r:p.old.truth(r['baseline_ready']) and p.old.truth(r['window_ready'])),ref0=p.points(sig[0],'psi'),ref1000=p.points(sig[1000],'psi'),probe_mu=p.points(pr,'accuracy'))
        cases.append(dict(dataset='controlled',dim=s['latent_dim'],seed=s['seed'],branches={'changing':dict(series=series)},summary=s))
    assert len(cases)==15
    for d in DIMS:assert sorted(c['seed'] for c in cases if c['dim']==d)==[0,1,2]
    return cases
def subset(c,key,window):return [(t,y) for t,y in p.get(c,key) if window[0]<=t<=window[1]]
def draw_rows(axes,cases,item):
    cut,window,phase=item
    for j,d in enumerate(DIMS):
        cs=[c for c in cases if c['dim']==d]
        for ax in axes[:,j]:
            p.decorate(ax,window=window);ax.tick_params(labelsize=10)
        axes[0,j].set_title(f'z={d} · {phase}',fontsize=13,pad=12)
        p.signal(axes[0,j],cs,window)
        p.line(axes[1,j],cs,'mse',p.COL['mse'],'当前流观测MSE');axes[1,j].set_ylabel('观测重构 MSE',fontsize=11);axes[1,j].set_ylim(bottom=0)
        probes=[subset(c,'probe',window) for c in cs]
        expected=list(range(window[0],window[1]+1,200))
        assert all([t for t,y in q]==expected for q in probes)
        probe_line=p.old.meanline(axes[2,j],probes,p.COL['fixed'],'原当前阶段θ',style='--',marker='o')
        probe_line.set_markersize(5.2);probe_line.set_markeredgecolor('white');probe_line.set_markeredgewidth(.5)
        axes[2,j].axhline(1/8,color='#9ca3af',ls=':',lw=.8);axes[2,j].set_ylim(0,1.02);axes[2,j].set_ylabel('θ普通准确率',fontsize=11);axes[2,j].set_xlabel('训练步',fontsize=11)
def legend(fig,y):
    from matplotlib.lines import Line2D
    items=[Line2D([0],[0],color=p.COL[k],lw=2,label=v) for k,v in [('ref0','新Ψ ref0（左轴）'),('ref1000','新Ψ ref1000（左轴）'),('old','旧Ξ（右轴）'),('mse','当前流MSE')]]
    items.append(Line2D([0],[0],color=p.COL['fixed'],lw=1.5,ls='--',marker='o',label='原θ探针：真实测点'))
    fig.legend(handles=items,loc='upper center',bbox_to_anchor=(.5,y),ncol=5,frameon=False,fontsize=12)
def export(fig,name):
    for ext in ['png','svg']:fig.savefig(FIG/(name+'.'+ext),dpi=220,facecolor='white')
    plt.close(fig)
def foot(fig,combined=False):
    text='全部3seed均值；阴影为seed最小—最大范围，不是置信区间。新旧信号读各自纵轴；ref1000从1001步起，建立前留空。\n'
    text+='θ沿用历史当前阶段训练池80/20回测、普通准确率。仅画真实200步间隔测点，虚线只连点；切换瞬间没有θ测量，不补点。\n'
    text+='此controlled历史轨迹没有本批图像实验的同父稳定分叉；本图不作稳定流校正，也不作为独立固定S1遗忘检验。'
    fig.text(.5,.018,text,ha='center',va='bottom',fontsize=11,color='#555555')
def main():
    verify();p.INPUTS.clear();cases=load()
    for item in WINDOWS:
        cut,window,phase=item;fig,axes=plt.subplots(3,5,figsize=(27,11.5),squeeze=False);draw_rows(axes,cases,item)
        fig.suptitle(f'controlled · {cut}步切换窗口（{window[0]}–{window[1]}步）',fontsize=24,y=.99);legend(fig,.95);foot(fig);fig.subplots_adjust(left=.055,right=.95,top=.85,bottom=.15,wspace=.46,hspace=.35);export(fig,f'controlled_切换{cut}_全部维数_信号_MSE_探针')
    fig,axes=plt.subplots(6,5,figsize=(27,20),squeeze=False)
    for k,item in enumerate(WINDOWS):draw_rows(axes[k*3:k*3+3,:],cases,item)
    fig.suptitle('controlled · 两次结构切换窗口 · 全部潜维数',fontsize=25,y=.993);legend(fig,.967);foot(fig,True);fig.subplots_adjust(left=.055,right=.95,top=.91,bottom=.092,wspace=.46,hspace=.40);export(fig,'controlled_两次切换窗口_全部维数_合图')
    rr=[]
    for c in cases:
        for cut,window,phase in WINDOWS:
            for metric in ['ref0','ref1000','old','mse','probe']:
                for t,y in subset(c,metric,window):rr.append(dict(dim=c['dim'],seed=c['seed'],cut=cut,metric=metric,step=t,value=y))
    with (REPORT/'window_points_by_seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    for file,h in p.INPUTS.items():assert sha(file)==h,file
    write(REPORT/'manifest.json',dict(status='complete',conditions=15,dimensions=DIMS,seeds=[0,1,2],windows=[list(w) for _,w,_ in WINDOWS],source_files=p.INPUTS,script_sha256=sha(Path(__file__)),figures={str(f.relative_to(ROOT)):sha(f) for f in FIG.iterdir()},probe_interpolation=False,stable_control_available=False,new_training_updates=0))
    lines=['# controlled切换窗口','', '15个完整条件：z=2/4/8/16/32，seed0/1/2。上排信号、中排MSE、下排当前阶段θ普通准确率。','', '原θ只有700/900/1100/1300和1700/1900/2100/2300等实际测点；虚线仅连接测点，不构造切换瞬间的准确率。ref1000建立前留空。没有本批同父稳定分叉，不套用图像数据的稳定流校正效应。','']
    for f in sorted(FIG.glob('*.png')):lines.append(f'- [{f.stem}](../../figures/controlled_switch_windows_20261119/{f.name})')
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    print('Saved3 controlled window PNG/SVG figures; original logs unchanged.')
if __name__=='__main__':main()
