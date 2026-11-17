import sys,unittest,importlib.util,json,types
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from runtime import PLAN
from observer import SchemeBObserver,AuditedSignal

class ObserverContract(unittest.TestCase):
    def test_original_current_readout_and_fallback_sequence(self):
        # Independent routing reference copied from the preserved native runner.
        observer=SchemeBObserver(10);signals={};g=torch.Generator().manual_seed(42)
        for stage in range(3):
            for batch in range(25):
                x=torch.randn(64,128,generator=g)*(stage+1)
                target=stage
                if target in signals and signals[target].readout_ready():source=target
                else:source=next((k for k in range(stage-1,-1,-1) if k in signals and signals[k].readout_ready()),target)
                if source not in signals:signals[source]=observer.make()
                value,log=signals[source].differentiable_window_signal(x)
                if target not in signals:signals[target]=observer.make()
                signals[target].update_baseline(x);signals[target].update_window(x)
                if source!=target and signals[source].baseline_ready():signals[source].update_window(x)
                row,spectrum=observer.step_features(x,stage)
                self.assertEqual(row['xi'],float(value));self.assertEqual(row['source_stage'],source+1)
                torch.testing.assert_close(observer.signals[target].cov,signals[target].cov,rtol=0,atol=0) if signals[target].cov is not None else None
                if stage>0 and batch<20:self.assertTrue(row['baseline_source'].startswith('fallback:'))
    def test_raw_eigenvalues_scale_and_absolute_peak(self):
        s=SchemeBObserver(10).make();x=torch.randn(64,128,generator=torch.Generator().manual_seed(3));a=s._shape_eigenvalues(x);b=s._shape_eigenvalues(2*x)
        torch.testing.assert_close(b,4*a);self.assertAlmostEqual(s.mp_edge(),(1+2**.5)**2);self.assertEqual(float(s._spectral_readout(torch.tensor([1.,5.,6.]),torch.tensor(4.))),2.)
    def test_no_global_rng_change_from_projector_and_observer(self):
        g=torch.Generator().manual_seed(4);x=torch.randn(64,1,64,64,generator=g);before=torch.get_rng_state().clone();o=SchemeBObserver(10)
        for _ in range(21):o(x,0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
