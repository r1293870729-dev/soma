import tempfile
import unittest
import numpy as np
from soma_b import BConfig,BMonitor,SIGNAL_ID,spectrum_diagnostics,validate_main_signal
from soma_protocol.signal import CalibrationError

POLICY={'signal_id':SIGNAL_ID,'family':'B','readout':'mass','reference_policy':'fixed',
        'edge_policy':'empirical_order_statistic','normalize_by_mode_count':False,
        'backpropagate_signal':False}

class FinalBTests(unittest.TestCase):
    def setUp(self):
        g=np.random.default_rng(12)
        self.white=g.normal(size=(128,4)); self.edge=g.normal(size=(16*12,4))
        self.x=g.normal(size=(80,4))*[3,1,1,.5]
        self.cfg=BConfig(window=16,alpha=.1,beta=.1,energy_floor=1e-12,negative_tolerance=1e-10)
        self.ids={'white':[f'w:{i}' for i in range(len(self.white))],
                  'edge':[f'e:{i}' for i in range(len(self.edge))]}
    def monitor(self):return BMonitor.fit(self.white,self.edge,self.cfg,self.ids)
    def test_mass_source_formula(self):
        r=spectrum_diagnostics([2,1.8,.1,.1],1.5)
        self.assertAlmostEqual(r['psi_b'],(2/1.5-1)+(1.8/1.5-1))
        self.assertEqual(r['active_modes'],2)
    def test_same_peak_different_mass(self):
        x=spectrum_diagnostics([2,1.8,.1,.1],1.5);y=spectrum_diagnostics([2,1,.5,.5],1.5)
        self.assertEqual(x['a_edge_diagnostic'],y['a_edge_diagnostic'])
        self.assertGreater(x['psi_b'],y['psi_b'])
    def test_single_spike_B_equals_A(self):
        r=spectrum_diagnostics([3,.5,.3,.2],1.5)
        self.assertEqual(r['psi_b'],r['a_edge_diagnostic'])
    def test_zero_and_boundary_no_contribution(self):
        self.assertEqual(spectrum_diagnostics([2,2,0,0],2)['psi_b'],0)
    def test_mass_not_probability(self):
        self.assertGreater(spectrum_diagnostics([4,0,0,0],1)['psi_b'],1)
    def test_bad_spectrum_or_edge(self):
        for v in [[],[1,-.01],[np.nan],[np.inf]]:
            with self.assertRaises(CalibrationError):spectrum_diagnostics(v,1.)
        for e in [0,-1,np.inf,np.nan]:
            with self.assertRaises(CalibrationError):spectrum_diagnostics([1,2],e)
    def test_B_policy_only(self):validate_main_signal(POLICY)
    def test_reject_other_main_signals(self):
        for key,val in [('family','A'),('readout','edge'),('readout','gap'),('readout','entropy'),
                        ('edge_policy','mp'),('reference_policy','ema'),
                        ('normalize_by_mode_count',True),('backpropagate_signal',True)]:
            with self.subTest(key=key,val=val),self.assertRaises(CalibrationError):
                validate_main_signal({**POLICY,key:val})
    def test_warming_is_not_zero(self):
        r=self.monitor().update(self.x[:2]);self.assertIsNone(r['psi_b'])
        self.assertEqual(r['status'],'warming-window')
    def test_reference_never_changes_online(self):
        m=self.monitor();h=m.reference_id()
        for chunk in np.array_split(self.x,5):m.update(chunk)
        self.assertEqual(h,m.reference_id())
    def test_future_stage_argument_rejected(self):
        with self.assertRaises(TypeError):self.monitor().update(self.x,stage_name='T2')
    def test_global_calibration_id_overlap(self):
        bad={**self.ids,'edge':self.ids['edge'].copy()};bad['edge'][0]=self.ids['white'][0]
        with self.assertRaises(ValueError):BMonitor.fit(self.white,self.edge,self.cfg,bad)
    def test_reference_serialization(self):
        m=self.monitor()
        with tempfile.TemporaryDirectory() as td:
            p=__import__('pathlib').Path(td)/'r.npz';m.save(p);n=BMonitor.load(p)
            self.assertEqual(m.reference_id(),n.reference_id())
            self.assertEqual(m.update(self.x),n.update(self.x))
    def test_raw_residual_scale_invariance(self):
        a=self.monitor().update(self.x)
        for k in [-2,.3,3.]:
            b=self.monitor().update(k*self.x)
            self.assertAlmostEqual(a['psi_b'],b['psi_b'],places=11)
    def test_nonfinite_not_silently_dropped(self):
        x=self.x.copy();x[0,0]=np.nan
        with self.assertRaises(CalibrationError):self.monitor().update(x)

if __name__=='__main__':unittest.main()
