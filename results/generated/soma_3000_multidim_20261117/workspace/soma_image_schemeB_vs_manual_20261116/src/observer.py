"""Detached July-report observer; known stage only enters this historical auxiliary."""
from dataclasses import asdict
import torch
from runtime import PLAN
from legacy_signal import RestoredAbsoluteMPSignal
from losses_pw import ResidualFeatureProjector

class AuditedSignal(RestoredAbsoluteMPSignal):
    def _shape_eigenvalues(self,x):
        v=super()._shape_eigenvalues(x);self.last_spectrum=v.detach().cpu().tolist();return v

class SchemeBObserver:
    def __init__(self,seed):
        self.signals={};self.projector=ResidualFeatureProjector(PLAN['legacy']['feature_dim'],seed+777)
    def make(self):
        keys=['feature_dim','window_samples','gamma','alpha','epsilon','threshold_mode','spectral_readout','baseline_warmup_batches']
        return AuditedSignal(**{k:PLAN['legacy'][k] for k in keys})
    @torch.no_grad()
    def step_features(self,features,stage):
        before_rng=torch.get_rng_state().clone();target=int(stage);sig=self.signals.get(target)
        source=target;route=f'warming:{target+1}'
        if sig is not None and sig.readout_ready():route=f'state:{target+1}'
        else:
            fallback=next((k for k in range(target-1,-1,-1) if k in self.signals and self.signals[k].readout_ready()),None)
            if fallback is not None:source=fallback;route=f'fallback:{fallback+1}->target:{target+1}'
        if source not in self.signals:self.signals[source]=self.make()
        # Same current-batch window readout as reconstructed loss_plus_xi,
        # but under no_grad and never added to the model objective.
        value,log=self.signals[source].differentiable_window_signal(features)
        eligible=bool(log.baseline_ready and log.window_ready)
        spectrum=list(self.signals[source].last_spectrum) if eligible else []
        row=dict(xi=float(value),eligible=eligible,baseline_source=route,target_stage=target+1,source_stage=source+1,**asdict(log))
        row['whitened_covariance_trace']=sum(spectrum) if spectrum else None
        if target not in self.signals:self.signals[target]=self.make()
        self.signals[target].update_baseline(features.detach());self.signals[target].update_window(features.detach())
        if source!=target and self.signals[source].baseline_ready():self.signals[source].update_window(features.detach())
        if not torch.equal(before_rng,torch.get_rng_state()):raise ValueError('Observer changed training RNG')
        return row,spectrum
    @torch.no_grad()
    def __call__(self,residual,stage):return self.step_features(self.projector(residual.detach()),stage)

