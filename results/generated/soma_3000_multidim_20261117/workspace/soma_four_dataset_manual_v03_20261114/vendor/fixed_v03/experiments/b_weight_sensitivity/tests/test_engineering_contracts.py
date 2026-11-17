import csv
from dataclasses import asdict
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from experiments.b_weight_sensitivity.weighting import (
    FixedWeightController, InitialScaleController, UncertaintyController,
    GradNormController, assert_disjoint_optimizers)
from experiments.b_weight_sensitivity.gradient_diagnostics import collect_gradient_vectors
from experiments.b_weight_sensitivity.historical_loader import HistoricalSource
from experiments.b_weight_sensitivity.legacy_b_monitor import LegacyBMonitor

class ControllerContracts(unittest.TestCase):
    def test_nonpositive_or_nonfinite_references_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), -float("inf")):
            for ctor in (lambda x: InitialScaleController({"rec": x}),
                         lambda x: GradNormController(["rec"], {"rec": x})):
                with self.subTest(value=value, ctor=ctor):
                    with self.assertRaises(ValueError):
                        ctor(value)

    def test_fixed_matches_original_two_term_objective(self):
        rec = torch.tensor(0.1234, requires_grad=True)
        kl = torch.tensor(3.258, requires_grad=True)
        result = FixedWeightController({"rec": 1, "kl": 0.001}).objective({"rec": rec, "kl": kl})
        self.assertTrue(torch.equal(result, rec + 0.001*kl))

    def test_calibration_reference_is_copied_and_frozen(self):
        values={"a":2.0,"b":4.0}
        c=InitialScaleController(values)
        values["a"]=100
        weights=c.weights({"a":torch.tensor(10.),"b":torch.tensor(20.)})
        self.assertAlmostEqual(float(weights["a"]),.5)
        self.assertAlmostEqual(float(weights["b"]),.25)

    def test_gradnorm_target_has_no_weight_gradient(self):
        c=GradNormController(["a","b"],{"a":1.,"b":1.})
        losses={"a":torch.tensor(1.,requires_grad=True),"b":torch.tensor(2.,requires_grad=True)}
        norms=torch.tensor([1.,3.],requires_grad=True)
        objective=c.gradnorm_loss(losses,norms)
        actual=torch.autograd.grad(objective,c.raw)[0]
        w=c.weights_tensor()
        rates=torch.tensor([2./3.,4./3.])
        target=(w.detach()*norms.detach()).mean()*rates.pow(1.5)
        expected=torch.autograd.grad(torch.abs(w*norms.detach()-target).sum(),c.raw)[0]
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=1e-6))
        w=c.weights_tensor()
        wrong_target=(w*norms.detach()).mean()*rates.pow(1.5)
        wrong=torch.autograd.grad(torch.abs(w*norms.detach()-wrong_target).sum(),c.raw)[0]
        self.assertFalse(torch.allclose(actual,wrong,atol=1e-6,rtol=1e-6))
        self.assertTrue(all(x.grad is None for x in losses.values()))
        self.assertIsNone(norms.grad)

    def test_model_objective_cannot_update_gradnorm_weights(self):
        c=GradNormController(["a","b"],{"a":1.,"b":1.})
        p=torch.nn.Parameter(torch.tensor(2.))
        objective=c.model_objective({"a":p.square(),"b":3*p.square()})
        objective.backward()
        self.assertIsNone(c.raw.grad)
        self.assertIsNotNone(p.grad)
        self.assertAlmostEqual(float(c.weights_tensor().sum()),2.,places=6)
        self.assertTrue(bool((c.weights_tensor()>0).all()))

    def test_optimizer_parameter_sets_must_be_disjoint(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        a=torch.optim.SGD([p],lr=.1)
        b=torch.optim.SGD([p],lr=.1)
        with self.assertRaises(ValueError):
            assert_disjoint_optimizers(a,b)

    def test_uncertainty_negative_loss_is_rejected(self):
        c=UncertaintyController(["rec"])
        with self.assertRaises(ValueError):
            c.objective({"rec":torch.tensor(-1.)})

    def test_weight_overflow_and_collapse_are_reported(self):
        c=UncertaintyController(["rec"])
        for value in (-1000.,1000.):
            with torch.no_grad():c.log_vars["rec"].fill_(value)
            with self.assertRaises(FloatingPointError):c.current_weights()
        g=GradNormController(["a","b"],{"a":1.,"b":1.})
        with torch.no_grad():g.raw.fill_(3e38)
        with self.assertRaises(FloatingPointError):g.weights_tensor()

class GradientContracts(unittest.TestCase):
    def test_finite_coefficient_can_still_overflow_weighted_gradient(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        with self.assertRaises(FloatingPointError):
            collect_gradient_vectors({"a":1e20*p},[p],{"a":1e20})

    def test_known_gradients_scaling_unused_and_zero_norm(self):
        p=torch.nn.Parameter(torch.tensor(2.))
        q=torch.nn.Parameter(torch.tensor(7.))
        rows,pairs=collect_gradient_vectors({"a":p.square(),"b":3*p.square(),"zero":torch.tensor(0.)},
                                             [p,q],{"a":1.,"b":.001,"zero":1.})
        by_name={r["loss_name"]:r for r in rows}
        self.assertEqual(by_name["a"]["raw_norm"],4.)
        self.assertEqual(by_name["a"]["unused_parameter_elements"],1)
        self.assertAlmostEqual(by_name["b"]["weighted_norm"],.012,places=6)
        self.assertEqual(by_name["zero"]["gradient_status"],"constant_loss")
        for pair in pairs:
            if pair["loss_b"]=="zero":
                self.assertIsNone(pair["cosine"])
                self.assertEqual(pair["status"],"undefined_zero_norm")
            else:
                self.assertAlmostEqual(pair["cosine"],1.)
        self.assertIsNone(p.grad)
        self.assertIsNone(q.grad)

    def test_negative_positive_weight_does_not_rotate_gradient(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        _,pairs=collect_gradient_vectors({"a":p.sum(),"b":-2*p.sum()},[p],{"a":.1,"b":100.})
        self.assertEqual(len(pairs),2)
        self.assertAlmostEqual(pairs[0]["cosine"],-1.,places=6)
        self.assertAlmostEqual(pairs[1]["cosine"],-1.,places=6)

@unittest.skipUnless(os.environ.get("BWS_LEGACY_ROOT"),"Original 03 snapshot is required")
class HistoricalContracts(unittest.TestCase):
    def setUp(self):
        self.root=Path(os.environ["BWS_LEGACY_ROOT"])
        self.config=dict(feature_dim=4,window_samples=8,gamma=.05,alpha=.1,warmup_batches=2,seed=777,epsilon=1e-5)

    def test_exact_original_readouts_no_rng_or_import_collision(self):
        original_path=list(sys.path)
        previous=sys.modules.get("utils")
        dummy=types.ModuleType("utils")
        dummy.make_generator=lambda *_: (_ for _ in ()).throw(AssertionError("Wrong utils module used"))
        sys.modules["utils"]=dummy
        try:
            loader=HistoricalSource(self.root)
            original=loader.load("signals_xib").XiBSignal(**self.config)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/"b.csv"
                monitor=LegacyBMonitor(self.root/"signals_xib.py",csv_path=path,**self.config)
                gen=torch.Generator().manual_seed(13)
                step=0
                for stage in ("s1","t2","local_split"):
                    for _ in range(6):
                        step+=1
                        x=torch.randn(4,5,generator=gen,requires_grad=True)
                        before=torch.get_rng_state().clone()
                        observed=monitor.update(x,stage,step,"unit-fixture","checkpoint-"+str(step))
                        expected=original.update_and_measure(x.detach(),stage)
                        self.assertEqual(asdict(observed),asdict(expected))
                        self.assertTrue(torch.equal(before,torch.get_rng_state()))
                        self.assertIsNone(x.grad)
                with path.open() as f:rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),18)
                self.assertTrue(all(r["checkpoint_id"] and r["source_sha256"] and r["config_sha256"] for r in rows))
                with self.assertRaises(ValueError):
                    monitor.update(x,"s1",18,"unit-fixture","duplicate")
                with self.assertRaises(ValueError):
                    monitor.update(x,"s1",19,"another-run","cross-run")
                with self.assertRaises(FileExistsError):
                    LegacyBMonitor(self.root/"signals_xib.py",csv_path=path,**self.config)
            self.assertIs(sys.modules["utils"],dummy)
            self.assertEqual(sys.path,original_path)
            loader.verify()
        finally:
            if previous is None:sys.modules.pop("utils",None)
            else:sys.modules["utils"]=previous

    def test_source_mismatch_is_rejected(self):
        loader=HistoricalSource(self.root)
        pin=dict(loader.hashes)
        pin["signals_xib.py"]="0"*64
        with self.assertRaises(ValueError):
            HistoricalSource(self.root,pin)

    def test_monitors_do_not_share_window_or_baseline(self):
        a=LegacyBMonitor(self.root/"signals_xib.py",**self.config)
        b=LegacyBMonitor(self.root/"signals_xib.py",**self.config)
        self.assertIsNot(a.signal.window,b.signal.window)
        a.update(torch.ones(4,5),"s1",1,"a","c1")
        self.assertEqual(len(b.signal.baselines),0)

if __name__=="__main__":
    unittest.main()
