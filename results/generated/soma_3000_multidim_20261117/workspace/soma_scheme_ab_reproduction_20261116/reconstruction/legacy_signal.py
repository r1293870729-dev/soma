"""Report-specified July Xi math, on top of the available pinned September class.

Only the three mathematically evidenced differences are overridden. This is a
protocol reconstruction, not a claim to possess the missing original source.
"""
import math
import torch
from losses_pw import PrecisionWeightedResidualSignal as AvailableSignal

class RestoredAbsoluteMPSignal(AvailableSignal):
    def mp_edge(self):
        return (1.0+math.sqrt(self.feature_dim/self.window_samples))**2

    def _shape_eigenvalues(self,whitened_window):
        if whitened_window.ndim!=2:raise ValueError('Expected samples x residual features')
        centered=whitened_window-whitened_window.mean(dim=0,keepdim=True)
        covariance=centered.T@centered/float(max(1,len(centered)-1))
        return torch.linalg.eigvalsh(covariance).clamp_min(0.)

    def _spectral_readout(self,eigenvalues,threshold):
        if self.spectral_readout!='edge':raise ValueError('Only the archived absolute-edge Xi is reconstructed')
        return torch.relu(eigenvalues[-1]-threshold)
