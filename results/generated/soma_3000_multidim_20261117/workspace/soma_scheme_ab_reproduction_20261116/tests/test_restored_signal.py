import sys,unittest,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'reconstruction'))
import runtime
from legacy_signal import RestoredAbsoluteMPSignal

class SignalContract(unittest.TestCase):
    def test_report_MP_edge_and_absolute_excess(self):
        s=RestoredAbsoluteMPSignal(64,64,.005,.1,threshold_mode='mp_edge',baseline_warmup_batches=20)
        self.assertEqual(s.mp_edge(),4.)
        self.assertEqual(float(s._spectral_readout(torch.tensor([1.,3.,5.]),torch.tensor(4.))),1.)
    def test_raw_covariance_scales_not_trace_normalized(self):
        s=RestoredAbsoluteMPSignal(4,64,.005,.1);g=torch.Generator().manual_seed(2);x=torch.randn(64,4,generator=g)
        a=s._shape_eigenvalues(x);b=s._shape_eigenvalues(2*x)
        torch.testing.assert_close(b,4*a)
    def test_current_batch_Xi_has_gradient(self):
        s=RestoredAbsoluteMPSignal(4,64,.005,.1,threshold_mode='mp_edge');g=torch.Generator().manual_seed(3);s.initialize(torch.randn(512,4,generator=g))
        x=(torch.randn(64,4,generator=g)*6).requires_grad_();xi,log=s.differentiable_window_signal(x);xi.backward();self.assertGreater(float(xi),0);self.assertGreater(float(x.grad.norm()),0);self.assertTrue(torch.isfinite(x.grad).all())
    def test_archived_parameters_are_not_retuned(self):
        r=runtime.load_runner();args,old,extra=runtime.case_args(r,2,'loss_plus_xi',ROOT/'results/test_unused')
        for key,value in old.items():
            if key not in ['output_dir','device','cifar10_root','cifarc_root']:self.assertEqual(getattr(args,key),value)
        self.assertEqual(args.xi_signal,'pw');self.assertEqual(args.pw_readout_mode,'current')

if __name__=='__main__':torch.set_num_threads(1);unittest.main()
