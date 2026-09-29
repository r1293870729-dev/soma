import itertools
import unittest
import numpy as np
from soma_protocol.signal import (CalibrationError, FixedReference, calibrate_edge,
                                  psi_window, ordered_shape_spectrum)
from soma_protocol.metrics import exact_alignment, restricted_recovery_burden
from soma_protocol.validation import assert_disjoint_roles, assert_arrived

class SignalTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(1).normal(size=(20,7))
        self.kw = dict(edge=2.0, floor=0.0, negative_tolerance=1e-10)
    def test_analytic_rank_one(self):
        r = psi_window(np.array([[-1.,0],[1.,0]]), **self.kw)
        self.assertEqual(r['psi'],0)
        r = psi_window(np.array([[-1.,0],[1.,0]]),1.,0.,1e-12)
        self.assertEqual(r['psi'],1)
    def test_all_active_modes(self):
        x = np.array([[2.,0],[-2.,0],[0,2],[0,-2]])
        r = psi_window(x,.5,0.,1e-12)
        self.assertEqual(r['active_modes'],2)
        self.assertAlmostEqual(r['psi'],2)
    def test_scale(self):
        for k in (-4.,.1,2.):
            self.assertAlmostEqual(psi_window(self.x,**self.kw)['psi'],
                                   psi_window(k*self.x,**self.kw)['psi'],places=12)
    def test_shift(self):
        self.assertAlmostEqual(psi_window(self.x,**self.kw)['psi'],
                               psi_window(self.x+3,**self.kw)['psi'])
    def test_floor_equal(self):
        x = np.array([[-1.,0],[1.,0]])
        self.assertEqual(psi_window(x,1.,2.,1e-12)['status'],'low-energy')
    def test_zero_window(self):
        self.assertEqual(psi_window(np.zeros((4,3)),**self.kw)['psi'],0)
    def test_edge_rejection(self):
        for edge in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(CalibrationError):
                psi_window(self.x,edge,0.,1e-12)
    def test_nonfinite(self):
        for v in (np.nan,np.inf):
            x=self.x.copy(); x[0,0]=v
            with self.assertRaises(CalibrationError): psi_window(x,**self.kw)
    def test_small_invalid(self):
        with self.assertRaises(CalibrationError): psi_window(np.zeros((1,3)),**self.kw)
    def test_backend_equivalence(self):
        for shape in ((7,20),(20,7),(7,7)):
            x=np.random.default_rng(8).normal(size=shape)
            a=psi_window(x,**self.kw,backend='covariance')
            b=psi_window(x,**self.kw,backend='gram')
            np.testing.assert_allclose(a['spectrum'],b['spectrum'],atol=1e-12)
            self.assertEqual(len(a['spectrum']),min(shape[1],shape[0]-1))
    def test_reference_ridge(self):
        r=FixedReference.fit(self.x,.2)
        c=np.cov(self.x,rowvar=False); reg=c+.2*np.trace(c)/7*np.eye(7)
        np.testing.assert_allclose(r.inverse_sqrt@reg@r.inverse_sqrt,np.eye(7),atol=1e-12)
        self.assertFalse(r.mean.flags.writeable)
    def test_zero_reference_fails(self):
        with self.assertRaises(CalibrationError): FixedReference.fit(np.zeros((4,2)),.1)
    def test_dimension_change(self):
        with self.assertRaises(CalibrationError): FixedReference.fit(self.x,.1).transform(self.x[:,:3])
    def test_edge_order_statistic(self):
        x=np.array([[0.,0],[0,0],[-1,0],[1,0],[-2,0],[2,0]])
        r=calibrate_edge(x,2,.5,0.,1e-12)
        self.assertEqual((r['k'],r['n_windows'],r['low_energy_windows']),(2,3,1))
        self.assertEqual(r['edge'],2.)
    def test_empty_edge_mass(self):
        with self.assertRaises(CalibrationError): calibrate_edge(np.zeros((6,2)),2,.05,0,1e-12)
    def test_edge_leftover_fails(self):
        with self.assertRaises(CalibrationError): calibrate_edge(self.x,3,.05,0,1e-12)
    def test_invalid_backend(self):
        with self.assertRaises(CalibrationError): psi_window(self.x,**self.kw,backend='approx')

class MetricTests(unittest.TestCase):
    def test_greedy_counterexample(self):
        r=exact_alignment([0,4],[3,7],3)
        self.assertEqual(r['matches'],[(0,3),(4,7)])
        self.assertEqual(r['taef'],1)
    def test_no_edit(self):
        self.assertEqual(exact_alignment([],[3],2)['taef'],0)
    def test_no_boundary(self):
        self.assertIsNone(exact_alignment([1],[],2)['boundary_hit_rate'])
    def test_duplicate_time_requires_policy(self):
        with self.assertRaises(ValueError): exact_alignment([1,1],[2],3)
    def test_match_bruteforce(self):
        rng=np.random.default_rng(2)
        for _ in range(30):
            a=sorted(rng.choice(12,4,replace=False).tolist())
            b=sorted(rng.choice(12,3,replace=False).tolist())
            expected=(0,0.)
            for k in range(1,4):
                for aa in itertools.combinations(a,k):
                    for bb in itertools.permutations(b,k):
                        ds=[abs(x-y) for x,y in zip(aa,bb)]
                        if max(ds)<=3 and (-k,sum(ds))<(-expected[0],expected[1]):
                            expected=(k,sum(ds))
            r=exact_alignment(a,b,3)
            self.assertEqual((r['n_matched'],r['total_absolute_delay']),expected)
    def test_no_drop(self): self.assertEqual(restricted_recovery_burden(False,None,100),0)
    def test_censored(self): self.assertEqual(restricted_recovery_burden(True,None,100),100)
    def test_recovered(self): self.assertEqual(restricted_recovery_burden(True,20,100),20)
    def test_recovery_cap(self): self.assertEqual(restricted_recovery_burden(True,200,100),100)
    def test_invalid_delay(self):
        with self.assertRaises(ValueError): restricted_recovery_burden(True,-1,100)

class IsolationTests(unittest.TestCase):
    def test_valid_roles(self): assert_disjoint_roles({'white':['a'],'edge':['b']})
    def test_overlap(self):
        with self.assertRaises(ValueError): assert_disjoint_roles({'white':['a'],'edge':['a']})
    def test_valid_arrival(self): assert_arrived([1,2,3],3)
    def test_future(self):
        with self.assertRaises(ValueError): assert_arrived([1001],290)

if __name__=='__main__': unittest.main()
