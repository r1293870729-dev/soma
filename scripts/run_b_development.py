"""Train a small fixed-topology VAE and record FINAL B from pre-update residuals.

Development only: illustrative S1 -> T2 -> locally split Fourier stream, not the
manual's frozen, moment-matched multi-generator benchmark. No structural edits,
no probe endpoint, no formal Q1/Q2/Q3 performance claim. A/C are diagnostic-only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from soma_b import BConfig,BMonitor,validate_main_signal

def main():
    import torch
    from torch import nn
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/dev_b_cpu.json')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text(encoding='utf8'))
    validate_main_signal(cfg['signal'])
    if cfg.get('status')!='development-only' or cfg.get('formal_test_unlocked'):
        p.error('This entry point is development-only, not a final-test runner')
    if a.out.exists():p.error('Output path already exists; use a new run ID')
    a.out.mkdir(parents=True)
    (a.out/'resolved_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\\n')
    status={'status':'running','claim_scope':'development-only','structural_edits':False}
    (a.out/'status.json').write_text(json.dumps(status))
    started=time.perf_counter()
    try:
        torch.set_num_threads(1);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
        children=np.random.SeedSequence(cfg['seed']).spawn(5)
        render_rng=np.random.default_rng(children[0]);train_rng=np.random.default_rng(children[1])
        white_rng=np.random.default_rng(children[2]);edge_rng=np.random.default_rng(children[3])
        d=cfg['dimension'];batch=cfg['batch_size'];zdim=cfg['latent_dimension']
        if d!=8:raise ValueError('This development generator is explicitly eight-dimensional')
        renderer=np.linalg.qr(render_rng.normal(size=(d,d)))[0]
        def sample(n,phase,rng):
            u=rng.uniform(0,2*np.pi,n)
            v=np.zeros(n) if phase==0 else rng.uniform(0,2*np.pi,n)
            f=np.column_stack([np.cos(u),np.sin(u),np.cos(v),np.sin(v),
                               np.cos(u+v),np.sin(u+v),np.cos(2*u),np.sin(2*u)])
            if phase==2:
                idx=(u<np.pi/2);branch=rng.integers(0,4,n);ang=branch*np.pi/2
                f[idx,6]+=0.6*np.cos(ang[idx]);f[idx,7]+=0.6*np.sin(ang[idx])
            return (np.tanh(f@renderer)+rng.normal(0,.02,(n,d))).astype('float32')
        class VAE(nn.Module):
            def __init__(self):
                super().__init__();self.encoder=nn.Sequential(nn.Linear(d,32),nn.Tanh())
                self.mu=nn.Linear(32,zdim);self.logvar=nn.Linear(32,zdim)
                self.decoder=nn.Sequential(nn.Linear(zdim,32),nn.Tanh(),nn.Linear(32,d))
            def stats(self,x):
                h=self.encoder(x);return self.mu(h),self.logvar(h).clamp(-10,5)
            def mean_prediction(self,x):
                m,_=self.stats(x);return self.decoder(m)
            def loss(self,x):
                m,l=self.stats(x);z=m+torch.randn_like(m)*torch.exp(.5*l)
                rec=self.decoder(z);mse=torch.mean((x-rec)**2)
                kl=-.5*torch.mean(1+l-m*m-l.exp())
                return mse+cfg['beta_kl']*kl
        model=VAE();opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        for _ in range(cfg['warmup_steps']):
            x=torch.from_numpy(sample(batch,0,train_rng));opt.zero_grad();loss=model.loss(x)
            loss.backward();opt.step()
        def residual(x):
            with torch.no_grad():return (x-model.mean_prediction(x)).numpy().astype('float64')
        white=residual(torch.from_numpy(sample(cfg['white_samples'],0,white_rng)))
        bc=BConfig(**cfg['numeric_parameters'])
        edge_n=cfg['edge_windows']*bc.window
        edge=residual(torch.from_numpy(sample(edge_n,0,edge_rng)))
        roles={'white':[f'dev-white:{i}' for i in range(len(white))],
               'edge':[f'dev-edge:{i}' for i in range(len(edge))]}
        monitor=BMonitor.fit(white,edge,bc,roles);monitor.save(a.out/'reference.npz')
        np.savez_compressed(a.out/'calibration_residuals.npz',white=white,edge=edge)
        all_residuals=[];valid=[];step=0
        with (a.out/'signal_log.jsonl').open('x') as signal_out,(a.out/'training_log.jsonl').open('x') as train_out:
            for phase,steps in enumerate(cfg['stage_steps']):
                for _ in range(steps):
                    step+=1;x=torch.from_numpy(sample(batch,phase,train_rng))
                    eps=residual(x);all_residuals.append(eps.copy());s=monitor.update(eps)
                    signal_out.write(json.dumps(s,allow_nan=False)+'\\n')
                    opt.zero_grad();loss=model.loss(x)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite model objective')
                    loss.backward();opt.step()
                    train_out.write(json.dumps({'step':step,'train_loss':float(loss.detach()),
                         'pre_update_mse':float(np.mean(eps**2))},allow_nan=False)+'\\n')
                    if s['psi_b'] is not None:valid.append(s)
        np.save(a.out/'residuals.npy',np.concatenate(all_residuals))
        monitor.save(a.out/'monitor_final.npz',include_window=True)
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                    'torch_rng':torch.get_rng_state(),'cfg':cfg},a.out/'development_model.pt')
        summary={'scope':'development-only; not a paper experiment','steps':step,
                 'samples':step*batch,'valid_diagnostic_steps':len(valid),
                 'multi_mode_steps':sum(s['multi_mode_active'] for s in valid),
                 'low_energy_steps':sum(s['status']=='low-energy' for s in valid),
                 'mean_B':float(np.mean([s['psi_b'] for s in valid])),
                 'mean_B_minus_A':float(np.mean([s['b_minus_a'] for s in valid])),
                 'lambda_plus_0':monitor.edge,'reference_id':monitor.reference_id(),
                 'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),
                 'device':'cpu','wall_seconds':time.perf_counter()-started,
                 'formal_Q1_test':False,'formal_Q2_test':False,'formal_Q3_test':False}
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\\n')
        (a.out/'status.json').write_text(json.dumps({**status,'status':'complete'},indent=2)+'\\n')
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    except Exception as e:
        (a.out/'status.json').write_text(json.dumps({**status,'status':'failed','error':repr(e)},indent=2)+'\\n')
        raise
if __name__=='__main__':main()
