"""Analyze predeclared reconstruction; never imports or alters training models."""
import argparse,csv,hashlib,json,math,datetime,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/reconstructed_B_001'
REPORT=ROOT/'reports';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def savejson(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def savecsv(name,rows):
    if not rows:return
    with (REPORT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def val(rows,key):return np.array([float(r[key]) for r in rows])
def theta(rows,t):return next(float(r['accuracy']) for r in rows if int(r['step'])==t)
def fixed(rows,view,t):return next(float(r['accuracy']) for r in rows if r['view']==view and int(r['step'])==t)
def decorate(ax):
    for t in [1000,2000]:ax.axvline(t,color='black',ls=':',lw=.6)
    ax.set_xlim(0,3000);ax.grid(alpha=.15)

def figures(curves):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(15,10),sharex=True,gridspec_kw={'height_ratios':[1.4,.7,1]})
    for col,d in enumerate([2,4]):
        if (d,'loss_plus_xi') not in curves:continue
        old,new,op,np_,fix=curves[d,'loss_plus_xi'];ax=axes[0,col];right=ax.twinx()
        for tr,style,alpha,prefix in [(old,'--',.55,'Archived'),(new,'-',1,'Fresh')]:
            t=val(tr,'step');ax.plot(t,val(tr,'reconstruction'),ls=style,alpha=alpha,color='#2674b8',label=prefix+' reconstruction')
            right.plot(t,val(tr,'xi_pw'),ls=style,alpha=alpha,color='#df3434',label=prefix+' Xi')
            axes[1,col].plot(t,val(tr,'spike_rank'),ls=style,alpha=alpha,color='#9467bd',label=prefix)
        ax.set_title(f'Scheme B | controlled | z={d} | loss_plus_xi');ax.set_ylabel('Reconstruction MSE',color='#2674b8');right.set_ylabel('Absolute MP excess Xi',color='#df3434')
        ax.legend(loc='upper left',fontsize=7);right.legend(loc='upper right',fontsize=7);axes[1,col].set_ylabel('Spike rank k');axes[1,col].legend(fontsize=7)
        for pr,style,alpha,label in [(op,'--',.6,'Archived theta'),(np_,'-',1,'Fresh theta')]:axes[2,col].plot(val(pr,'step'),val(pr,'accuracy'),ls=style,marker='.',alpha=alpha,color='#269b43',label=label)
        axes[2,col].set_ylim(0,1.02);axes[2,col].set_ylabel('Current-stage theta accuracy');axes[2,col].set_xlabel('Formal training step');axes[2,col].legend(fontsize=8)
        for ax in axes[:,col]:decorate(ax)
    fig.suptitle('Historical protocol reconstruction vs original archived logs | seed 0\nDashed = archived July run; solid = fresh CPU reconstruction. Original source/environment unavailable.\nTheta uses the historical changing-stage pool and changing 80/20 split; not a fixed old-task test.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(FIG/'reconstructed_vs_archive_z2_z4.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(20,10),sharex=True)
    for col,d in enumerate([2,4,8,16,32]):
        for mode,color in [('loss_only','#777777'),('loss_plus_xi','#2467b3')]:
            if (d,mode) not in curves:continue
            old,new,op,np_,fix=curves[d,mode]
            for tr,pr,style,alpha,label in [(old,op,'--',.55,'archived'),(new,np_,'-',1,'fresh')]:
                axes[0,col].plot(val(tr,'step'),val(tr,'reconstruction'),color=color,ls=style,alpha=alpha,label=f'{mode}, {label}')
                axes[1,col].plot(val(tr,'step'),val(tr,'xi_pw'),color=color,ls=style,alpha=alpha)
                axes[2,col].plot(val(pr,'step'),val(pr,'accuracy'),color=color,ls=style,alpha=alpha)
        axes[0,col].set_title(f'Latent z={d}');axes[0,col].set_ylabel('Reconstruction MSE');axes[1,col].set_ylabel('Legacy Xi');axes[2,col].set_ylabel('Current-stage theta accuracy');axes[2,col].set_ylim(0,1.02);axes[2,col].set_xlabel('Formal training step');axes[0,col].legend(fontsize=6)
        for ax in axes[:,col]:decorate(ax)
    fig.suptitle('Scheme B | all five dimensions, both objectives | archived dashed vs fresh solid | seed 0\nAll predeclared cases shown; no parameter/seed search. This is protocol reconstruction, not original-source exact reproduction.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(FIG/'reconstructed_vs_archive_all_dimensions.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,sharey=True)
    for col,d in enumerate([2,4]):
        for row,mode in enumerate(['loss_plus_xi','loss_only']):
            if (d,mode) not in curves:continue
            old,new,op,np_,fix=curves[d,mode];ax=axes[row,col]
            for view,color,label in [('fixed_S1','#2467b3','Fixed S1 pool, fixed split/head seed'),('current_stage','#e97c22','Current-stage pool, fixed split/head seed')]:
                pr=[r for r in fix if r['view']==view];ax.plot(val(pr,'step'),val(pr,'accuracy'),marker='o',color=color,label=label)
            ax.plot(val(np_,'step'),val(np_,'accuracy'),ls=':',color='#777777',alpha=.65,label='Historical online protocol (changing split)')
            ax.set_title(f'z={d} | {mode}');ax.set_ylim(0,1.02);ax.set_ylabel('Theta accuracy');ax.set_xlabel('Snapshot after formal model updates');decorate(ax);ax.legend(fontsize=7)
    fig.suptitle('Offline data-view diagnostic on frozen fresh-training snapshots\nSame historical classifier + fixed seed 10900; original first2048 training-pool rows, 80/20. Not an independent final test.\nThe two views coincide with original online theta at step900; the diagnostic never feeds back into model training.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.89));fig.savefig(FIG/'reconstructed_fixed_S1_vs_current_stage.png',dpi=170);plt.close(fig)

def main(partial=False):
    frozen=load(RUN/'frozen.json');m=load(REPORT/'source_manifest.json');bad=[n for n,h in frozen['files'].items() if sha(ROOT/n)!=h]
    if bad:raise ValueError('Frozen source changed: '+repr(bad))
    for item in m['original_run_logs']:
        for n,meta in item['files'].items():
            if sha(ROOT/'archive_logs'/f'B_z{item["dimension"]}_{item["mode"]}'/n)!=meta['sha256']:raise ValueError('Archived input changed')
    curves={};stats=[];audit=[];matches=[];missing=[];pairhash={};routing=[]
    for d in [2,4,8,16,32]:
        for mode in ['loss_plus_xi','loss_only']:
            name=f'B_z{d}_{mode}';out=RUN/name
            if not (out/'execution.json').exists() or not (out/'fixed_probe_summary.json').exists():
                missing.append(dict(case=name,training_failure=load(out/'FAILED.json') if (out/'FAILED.json').exists() else None));continue
            e=load(out/'execution.json');a=load(out/'reconstruction_args.json');q=load(out/'fixed_probe_summary.json')
            if e['status']!='complete' or q['status']!='complete':raise ValueError('Incomplete case '+name)
            old=read(ROOT/'archive_logs'/name/'train_history.csv');new=read(out/'train_history.csv')
            op=[r for r in read(ROOT/'archive_logs'/name/'probe_history.csv') if r['task']=='theta'];np_=[r for r in read(out/'probe_history.csv') if r['task']=='theta'];fix=read(out/'fixed_probe_diagnostic.csv')
            if [r['step'] for r in old]!=[r['step'] for r in new] or [r['step'] for r in op]!=[r['step'] for r in np_]:raise ValueError('Clock mismatch '+name)
            for r in new:
                if not all(math.isfinite(float(r[k])) for k in ['objective','reconstruction','kl','xi_pw','lambda_max_batch','lambda_edge','grad_norm']):raise ValueError('Nonfinite '+name)
            xierr=float(np.max(np.abs(val(new,'xi_pw')-np.maximum(val(new,'lambda_max_batch')-val(new,'lambda_edge'),0))))
            objerr=float(np.max(np.abs(val(new,'objective')-val(new,'reconstruction')-.001*val(new,'kl')-(.001*val(new,'xi_pw') if mode=='loss_plus_xi' else 0))))
            if xierr>1e-4 or objerr>1e-5 or not np.all(val(new,'lambda_edge')==4):raise ValueError('Restored equation violated '+name)
            if e['formal_model_updates']!=3000 or e['separate_calibration_model_updates']!=3000 or len(np_)!=15 or not q['anchor_roundtrip_passed']:raise ValueError('Budget/round-trip failure '+name)
            for k,v in a['archived'].items():
                if k not in ['device','output_dir','cifar10_root','cifarc_root'] and a['actual'][k]!=v:raise ValueError('Archived hyperparameter changed '+name+' '+k)
            pairhash[d,mode]=e['initial_model_sha256'];curves[d,mode]=(old,new,op,np_,fix)
            st=dict(dimension=d,mode=mode,seed=0,formal_updates=3000,calibration_updates=3000,archived_final_rec=float(old[-1]['reconstruction']),fresh_final_rec=float(new[-1]['reconstruction']),rec_RMSE=float(np.sqrt(np.mean((val(new,'reconstruction')-val(old,'reconstruction'))**2))),archived_peak_Xi=float(max(val(old,'xi_pw'))),fresh_peak_Xi=float(max(val(new,'xi_pw'))),Xi_RMSE=float(np.sqrt(np.mean((val(new,'xi_pw')-val(old,'xi_pw'))**2))),archived_online_theta_2900=theta(op,2900),fresh_online_theta_2900=theta(np_,2900),theta_MAE_pp=float(100*np.mean(np.abs(val(np_,'accuracy')-val(op,'accuracy')))))
            for label,pre,post in [('S1_to_S2',900,1100),('S2_to_S3',1900,2100)]:
                st[f'archived_drop_{label}_pp']=100*(theta(op,pre)-theta(op,post));st[f'fresh_drop_{label}_pp']=100*(theta(np_,pre)-theta(np_,post));st[f'fresh_fixed_S1_drop_{label}_pp']=100*(fixed(fix,'fixed_S1',pre)-fixed(fix,'fixed_S1',post))
            for t in [900,1100,1900,2100,2900]:
                st[f'fresh_fixed_S1_theta_{t}']=fixed(fix,'fixed_S1',t);st[f'fresh_fixed_current_theta_{t}']=fixed(fix,'current_stage',t)
            stats.append(st)
            for provenance,tr in [('archived',old),('fresh',new)]:
                for r in tr:
                    if 1000<=int(r['step'])<=1040 or 2000<=int(r['step'])<=2040:
                        routing.append(dict(dimension=d,mode=mode,source=provenance,step=int(r['step']),Xi=float(r['xi_pw']),baseline_source=r['baseline_source'],signal_key=r['signal_key'],target_signal_key=r['target_signal_key']))
            for o,n in zip(old,new):
                r=dict(dimension=d,mode=mode,step=int(n['step']))
                for k in ['reconstruction','kl','xi_pw','spike_rank','objective']:r[k+'_archived']=float(o[k]);r[k+'_fresh']=float(n[k]);r[k+'_delta']=float(n[k])-float(o[k])
                matches.append(r)
            head=read(out/'fixed_probe_diagnostic.csv');online=read(out/'probe_loss_history.csv')
            audit.append(dict(case=name,formal_updates=3000,separate_calibration_updates=3000,online_theta_evaluations=len(np_),offline_theta_evaluations=len(head),offline_classifier_epochs=sum(int(r['epochs']) for r in head),online_classifier_loss_rows=len(online),Xi_equation_max_error=xierr,objective_equation_max_error=objerr,roundtrip_passed=q['anchor_roundtrip_passed'],initial_model_sha256=e['initial_model_sha256'],wall_seconds=e['wall_seconds'],fresh_source='report protocol reconstruction, exact historical source unavailable'))
    for d in [2,4,8,16,32]:
        if (d,'loss_only') in pairhash and (d,'loss_plus_xi') in pairhash and pairhash[d,'loss_only']!=pairhash[d,'loss_plus_xi']:raise ValueError('Paired model initialization mismatch')
    if not partial and missing and not (RUN/'job_failures.txt').exists():raise ValueError('Workers still running; use --partial')
    savecsv('reconstruction_metrics.csv',stats);savecsv('reconstruction_audits.csv',audit);savecsv('reconstruction_matched_steps.csv',matches);savecsv('reconstruction_stage_key_routing.csv',routing)
    if curves:figures(curves)
    status='complete' if not missing else ('partial' if partial else 'complete_with_missing_cases')
    result=dict(status=status,planned_cases=10,completed_cases=len(stats),missing=missing,original_source_available=False,curve_tuning=False,seed_search=False,formal_updates=sum(s['formal_updates'] for s in stats),calibration_updates=sum(s['calibration_updates'] for s in stats),paired_initialization_verified=True,frozen_hashes_verified=True,archive_hashes_verified=True,analysis_sha256=sha(Path(__file__)),single_seed_only=True)
    savejson(REPORT/'reconstruction_analysis.json',result)
    lines=['# 方案 B 历史协议重建结果','',f'生成时间：{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}。状态：{status}；完成 {len(stats)}/10 预定组。','',
    '这是从报告公式、原始参数和后续存留源码重建的全新训练。原始 7 月源码与环境不可得，因此不能宣称原版源码精确复现。原日志重绘位于 original_log_*，本次新训练位于 reconstructed_*，两者明确分开。','',
    '## 恢复范围与核验','',
    '- 恢复未迹归一化白化协方差、Ξ=max(λmax−λMP,0)、λMP=(1+√(D/W))²=4。保留 state-keyed EMA/shrinkage、fallback 和现存 runner 的窗口细节；这些继承细节无法逐字核对原源码。',
    '- 五个潜维 2/4/8/16/32，各做 loss_only 与 loss_plus_xi，原 seed 0；目标 rec+0.001 KL，后者另加 0.001 Ξ。原学习率、数据生成、批量、阶段与探针参数不改，不追求曲线贴合而调参。',
    f'- 已完成正式模型 {result["formal_updates"]} 次更新、独立校准模型 {result["calibration_updates"]} 次更新；每个完成组分别 3000 次。模型和信号核心源码 SHA、原始输入 SHA、配对初始化、Ξ/目标恒等式及探针快照回放均已核验。',
    '- CPU 单线程；缺失原环境/原版源码。新环境及额外后续中性参数见各 reconstruction_args.json 和 environment.json。失败/缺失组保留分母，不替换 seed。','',
    '## 与原始在线曲线比较','',
    '初始校准第 1 步和正式第 1 步（Ξ 尚未激活）即存在数值差异，因此新旧轨迹差异不能仅归因于 Ξ 公式。原硬件/RNG 路径、依赖版本和后续源码修订均无法完整分离；不据新结果反推或调换初始状态。','',
    '仅比较相同时点、同定义的 online theta；原报告另一个 heldout_all 最终评分缺少对应入口，不拿它与 online@2900 混比。下降量为“切换前准确率−切换后准确率”，单位为百分点；负值表示上升。','',
    '| z | 目标 | 最终 MSE 原→新 | Ξ峰值 原→新 | θ@2900 原→新 | θ均绝对差/百分点 | S1→S2下降 原→新/百分点 | S2→S3下降 原→新/百分点 |',
    '|---|---|---|---|---|---|---|---|']
    for s in stats:lines.append(f'| {s["dimension"]} | {s["mode"]} | {s["archived_final_rec"]:.5f}→{s["fresh_final_rec"]:.5f} | {s["archived_peak_Xi"]:.1f}→{s["fresh_peak_Xi"]:.1f} | {s["archived_online_theta_2900"]:.3f}→{s["fresh_online_theta_2900"]:.3f} | {s["theta_MAE_pp"]:.2f} | {s["archived_drop_S1_to_S2_pp"]:.2f}→{s["fresh_drop_S1_to_S2_pp"]:.2f} | {s["archived_drop_S2_to_S3_pp"]:.2f}→{s["fresh_drop_S2_to_S3_pp"]:.2f} |')
    lines+=['','## 阶段尖峰的解释','',
    '旧方案 B 使用已知 stage key：新阶段前约 20 个 batch 借用上一阶段 baseline，然后转用当前阶段 baseline。日志显示的切换尖峰及其快速归零，必须结合这种参考切换解释，不能把归零直接解释为旧任务已恢复。路由明细见 reconstruction_stage_key_routing.csv。它与手册中新固定参考信号不是同一个估计量，绝对数值不能直接比较。','',
    '## 数据视图诊断','',
    '原在线探针在每阶段当前训练池前 2048 行上重新随机 80/20 拟合与评分；换阶段也换数据，不能单凭其下降判断遗忘。新增诊断在训练完成后加载固定快照，用同一历史线性分类器和固定 split/head seed 10900，比对固定 S1 池与当前阶段池。900 步与原在线探针严格回放一致；之后诊断不影响模型训练。','',
    '这批样本仍来自历史训练池，20% 又用于 early stopping，并非独立最终 test；诊断只能区分数据视图变化与固定旧数据可读性变化，不能作为新协议独立测试结论。','',
    '| z | 目标 | 固定 S1 θ900→1100 | 当前池 θ1100（固定划分） | 固定 S1 θ1900→2100 | 当前池 θ2100（固定划分） |',
    '|---|---|---|---|---|---|']
    for s in stats:lines.append(f'| {s["dimension"]} | {s["mode"]} | {s["fresh_fixed_S1_theta_900"]:.3f}→{s["fresh_fixed_S1_theta_1100"]:.3f} | {s["fresh_fixed_current_theta_1100"]:.3f} | {s["fresh_fixed_S1_theta_1900"]:.3f}→{s["fresh_fixed_S1_theta_2100"]:.3f} | {s["fresh_fixed_current_theta_2100"]:.3f} |')
    if missing:lines+=['','## 未完成组','',*[f'- {x["case"]}；训练错误记录：{x["training_failure"]}' for x in missing]]
    lines+=['','## 图与可追溯文件','',
    '- `../figures/reconstructed_vs_archive_z2_z4.png`：截图维数新旧叠图；虚线为旧日志，实线为新训练。',
    '- `../figures/reconstructed_vs_archive_all_dimensions.png`：全部维数和目标。',
    '- `../figures/reconstructed_fixed_S1_vs_current_stage.png`：固定 S1 与当前池诊断。',
    '- `reconstruction_metrics.csv`、`reconstruction_matched_steps.csv`：同钟点误差；`reconstruction_audits.csv`：预算与恒等式审计。',
    '- `../results/reconstructed_B_001/`：原始新训练日志、模型快照、探针拟合历史、环境、冻结清单。',
    '- 每条件仅原 seed 0；不声称统计显著、方法优越性或完整论文验证。图像实验保持原运行配置。','']
    (REPORT/'RECONSTRUCTION_RESULTS.md').write_text('\n'.join(lines))
    if status=='complete':
        files=[]
        for folder in ['candidate_handoff_20260903','reconstruction','configs','scripts','tests','archive_logs','reports','figures','vendor_deps']:
            files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and '.tmp' not in p.suffixes and p.name!='RECONSTRUCTION_PROGRESS.md']
        files += [p for p in RUN.rglob('*') if p.is_file() and p.suffix!='.partial']
        files += [ROOT/'README.md',ROOT/'docs/RECONSTRUCTION_PROTOCOL.md',ROOT/'docs/source/scheme_ab_report.pdf']
        manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
        manifest.pop('reports/reconstruction_delivery_manifest.json',None)
        savejson(REPORT/'reconstruction_delivery_manifest.json',manifest)
        files=list(sorted(set(files)|{REPORT/'reconstruction_delivery_manifest.json'}))
        dest=ROOT/'scheme_B_fresh_protocol_reconstruction.zip';tmp=dest.with_suffix('.partial')
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for p in files:z.write(p,p.relative_to(ROOT))
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None:raise ValueError('Delivery zip CRC failure')
            for name,h in manifest.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Delivery hash mismatch '+name)
        tmp.replace(dest)
        print(json.dumps(dict(**result,zip=str(dest),zip_bytes=dest.stat().st_size,zip_sha256=sha(dest)),ensure_ascii=False,indent=2))
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
